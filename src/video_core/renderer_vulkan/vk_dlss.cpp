// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <fmt/ranges.h>

#include "common/div_ceil.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/host_shaders/dlss_motion_vectors_comp_spv.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

namespace Vulkan {
namespace {

// NGX identifies applications without an NVIDIA-issued id by a project id and engine type.
constexpr const char* NGX_PROJECT_ID = "8c3c5b1e-6d0f-4a52-9e1b-5f2a7d4e0b31";
constexpr const char* NGX_ENGINE_VERSION = "2";

// Halton (2, 3) points per jitter cycle; NVIDIA suggests at least 8 per unit of upscale ratio.
constexpr u32 JITTER_PHASES = 16;

constexpr VkFormat MOTION_VECTOR_FORMAT = VK_FORMAT_R16G16_SFLOAT;
constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

using Matrix = std::array<double, 16>; // column-major: element (row r, column c) at c * 4 + r

const char* Env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

bool EnvFlag(const char* name, bool fallback) {
    const char* value = Env(name);
    return value ? *value != '0' : fallback;
}

/// Test switches for the DLSS integration, read once.
struct Options {
    NVSDK_NGX_DLSS_Hint_Render_Preset preset = NVSDK_NGX_DLSS_Hint_Render_Preset_K;
    char preset_name = 'K';
    bool jitter = true;
    bool motion_vectors = true;
    bool hdr = false;
    bool depth_inverted = true;
    float jitter_sign_x = 1.0f;
    float jitter_sign_y = 1.0f;
};

const Options& GetOptions() {
    static const Options options = [] {
        Options result;
        if (const char* preset = Env("EDEN_DLSS_PRESET")) {
            switch (*preset | 0x20) {
            case 'j':
                result.preset = NVSDK_NGX_DLSS_Hint_Render_Preset_J;
                result.preset_name = 'J';
                break;
            case 'l':
                result.preset = NVSDK_NGX_DLSS_Hint_Render_Preset_L;
                result.preset_name = 'L';
                break;
            case 'm':
                result.preset = NVSDK_NGX_DLSS_Hint_Render_Preset_M;
                result.preset_name = 'M';
                break;
            default:
                break;
            }
        }
        result.jitter = EnvFlag("EDEN_DLSS_JITTER", true);
        result.motion_vectors = EnvFlag("EDEN_DLSS_MV", true);
        result.hdr = EnvFlag("EDEN_DLSS_HDR", false);
        result.depth_inverted = EnvFlag("EDEN_DLSS_DEPTH_INVERTED", true);
        if (const char* flip = Env("EDEN_DLSS_FLIP_JITTER")) {
            result.jitter_sign_x = std::strchr(flip, 'x') ? -1.0f : 1.0f;
            result.jitter_sign_y = std::strchr(flip, 'y') ? -1.0f : 1.0f;
        }
        return result;
    }();
    return options;
}

float Halton(u32 index, u32 base) {
    float fraction = 1.0f;
    float result = 0.0f;
    while (index > 0) {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(index % base);
        index /= base;
    }
    return result;
}

Matrix Identity() {
    Matrix m{};
    m[0] = m[5] = m[10] = m[15] = 1.0;
    return m;
}

Matrix Multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += a[k * 4 + row] * b[column * 4 + k];
            }
            result[column * 4 + row] = sum;
        }
    }
    return result;
}

std::optional<Matrix> Inverse(const Matrix& m) {
    Matrix inv;
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (!std::isfinite(det) || std::abs(det) < 1e-30) {
        return std::nullopt;
    }
    for (double& value : inv) {
        value /= det;
    }
    return inv;
}

VkImageSubresourceRange SubresourceRange(VkImageAspectFlags aspect) {
    return VkImageSubresourceRange{
        .aspectMask = aspect,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

VkImageMemoryBarrier ImageBarrier(VkImage image, VkImageAspectFlags aspect, VkAccessFlags src,
                                  VkAccessFlags dst, VkImageLayout old_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src,
        .dstAccessMask = dst,
        .oldLayout = old_layout,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = SubresourceRange(aspect),
    };
}

const std::wstring& ApplicationDataPath() {
    // NGX keeps its logs and temporary files here.
    static const std::wstring path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir).wstring();
    return path;
}

void NVSDK_CONV LogFromNgx(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    std::string line{message};
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    LOG_INFO(Render_Vulkan, "NGX: {}", line);
}

NVSDK_NGX_FeatureCommonInfo MakeCommonInfo() {
    NVSDK_NGX_FeatureCommonInfo info{};
    info.LoggingInfo.LoggingCallback = LogFromNgx;
    info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    info.LoggingInfo.DisableOtherLoggingSinks = false;
    return info;
}

NVSDK_NGX_FeatureDiscoveryInfo MakeDiscoveryInfo(const NVSDK_NGX_FeatureCommonInfo& common) {
    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    info.Identifier.v.ProjectDesc.ProjectId = NGX_PROJECT_ID;
    info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    info.Identifier.v.ProjectDesc.EngineVersion = NGX_ENGINE_VERSION;
    info.ApplicationDataPath = ApplicationDataPath().c_str();
    info.FeatureInfo = &common;
    return info;
}

std::vector<std::string> ExtensionNames(u32 count, const VkExtensionProperties* properties) {
    std::vector<std::string> names;
    names.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        names.emplace_back(properties[i].extensionName);
    }
    return names;
}

struct RenderSizes {
    u32 width, height;
    u32 min_width, min_height;
    u32 max_width, max_height;
};

// What NGX_DLSS_GET_OPTIMAL_SETTINGS in nvsdk_ngx_helpers.h does, without that header's
// Direct3D and CUDA helpers.
std::optional<RenderSizes> OptimalRenderSizes(NVSDK_NGX_Parameter* capabilities, u32 output_width,
                                              u32 output_height,
                                              NVSDK_NGX_PerfQuality_Value quality) {
    void* callback = nullptr;
    NVSDK_NGX_Parameter_GetVoidPointer(capabilities, NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback,
                                       &callback);
    if (callback == nullptr) {
        return std::nullopt;
    }
    NVSDK_NGX_Parameter_SetUI(capabilities, NVSDK_NGX_Parameter_Width, output_width);
    NVSDK_NGX_Parameter_SetUI(capabilities, NVSDK_NGX_Parameter_Height, output_height);
    NVSDK_NGX_Parameter_SetI(capabilities, NVSDK_NGX_Parameter_PerfQualityValue, quality);
    NVSDK_NGX_Parameter_SetI(capabilities, NVSDK_NGX_Parameter_RTXValue, 0);
    const auto get_settings = reinterpret_cast<PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback>(callback);
    if (NVSDK_NGX_FAILED(get_settings(capabilities))) {
        return std::nullopt;
    }
    RenderSizes sizes{};
    NVSDK_NGX_Parameter_GetUI(capabilities, NVSDK_NGX_Parameter_OutWidth, &sizes.width);
    NVSDK_NGX_Parameter_GetUI(capabilities, NVSDK_NGX_Parameter_OutHeight, &sizes.height);
    sizes.min_width = sizes.max_width = sizes.width;
    sizes.min_height = sizes.max_height = sizes.height;
    NVSDK_NGX_Parameter_GetUI(capabilities, NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Width,
                              &sizes.max_width);
    NVSDK_NGX_Parameter_GetUI(capabilities, NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Height,
                              &sizes.max_height);
    NVSDK_NGX_Parameter_GetUI(capabilities, NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Width,
                              &sizes.min_width);
    NVSDK_NGX_Parameter_GetUI(capabilities, NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Height,
                              &sizes.min_height);
    return sizes;
}

const char* QualityName(NVSDK_NGX_PerfQuality_Value quality) {
    switch (quality) {
    case NVSDK_NGX_PerfQuality_Value_MaxPerf:
        return "Performance";
    case NVSDK_NGX_PerfQuality_Value_Balanced:
        return "Balanced";
    case NVSDK_NGX_PerfQuality_Value_MaxQuality:
        return "Quality";
    case NVSDK_NGX_PerfQuality_Value_UltraPerformance:
        return "Ultra Performance";
    case NVSDK_NGX_PerfQuality_Value_UltraQuality:
        return "Ultra Quality";
    case NVSDK_NGX_PerfQuality_Value_DLAA:
        return "DLAA";
    default:
        return "?";
    }
}

struct MotionVectorPushConstants {
    std::array<float, 16> reprojection;
    std::array<float, 4> viewport_xy;
    std::array<float, 4> depth_and_jitter;
    std::array<u32, 2> size;
};

constexpr std::array<VkDescriptorSetLayoutBinding, 2> MOTION_VECTOR_BINDINGS{{
    {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    },
    {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    },
}};

constexpr std::array<VkDescriptorUpdateTemplateEntry, 2> MOTION_VECTOR_TEMPLATE{{
    {
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .offset = 0,
        .stride = sizeof(DescriptorUpdateEntry),
    },
    {
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .offset = sizeof(DescriptorUpdateEntry),
        .stride = sizeof(DescriptorUpdateEntry),
    },
}};

constexpr DescriptorBankInfo MOTION_VECTOR_BANK_INFO{
    .uniform_buffers = 0,
    .storage_buffers = 0,
    .texture_buffers = 0,
    .image_buffers = 0,
    .textures = 1,
    .images = 1,
    .score = 2,
};

constexpr VkPushConstantRange MOTION_VECTOR_PUSH_CONSTANTS{
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = sizeof(MotionVectorPushConstants),
};

} // Anonymous namespace

/// Writes camera motion vectors (dlss_motion_vectors.comp) from the scene depth.
class DlssMotionVectorPass final : public ComputePass {
public:
    explicit DlssMotionVectorPass(const Device& device_, Scheduler& scheduler_,
                                  DescriptorPool& descriptor_pool_,
                                  ComputePassDescriptorQueue& compute_pass_descriptor_queue_)
        : ComputePass(device_, scheduler_, descriptor_pool_, MOTION_VECTOR_BINDINGS,
                      MOTION_VECTOR_TEMPLATE, MOTION_VECTOR_BANK_INFO, MOTION_VECTOR_PUSH_CONSTANTS,
                      DLSS_MOTION_VECTORS_COMP_SPV),
          scheduler{scheduler_}, compute_pass_descriptor_queue{compute_pass_descriptor_queue_} {}

    void Run(VkImageView depth_view, VkSampler sampler, VkImageView motion_vector_view,
             const MotionVectorPushConstants& constants) {
        compute_pass_descriptor_queue.Acquire(scheduler, 2);
        compute_pass_descriptor_queue.AddSampledImage(depth_view, sampler);
        compute_pass_descriptor_queue.AddImage(motion_vector_view);
        const void* const descriptor_data{compute_pass_descriptor_queue.UpdateData()};
        const u32 groups_x = Common::DivCeil(constants.size[0], 8U);
        const u32 groups_y = Common::DivCeil(constants.size[1], 8U);
        scheduler.Record([this, descriptor_data, constants, groups_x,
                          groups_y](vk::CommandBuffer cmdbuf) {
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptor_data);
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
            cmdbuf.Dispatch(groups_x, groups_y, 1);
        });
    }

private:
    Scheduler& scheduler;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
};

namespace Dlss {

bool Requested() {
    static const bool requested = EnvFlag("EDEN_DLSS", false);
    return requested;
}

const std::vector<std::string>& InstanceExtensions() {
    static const std::vector<std::string> extensions = [] {
        if (!Requested()) {
            return std::vector<std::string>{};
        }
        const NVSDK_NGX_FeatureCommonInfo common = MakeCommonInfo();
        const NVSDK_NGX_FeatureDiscoveryInfo discovery = MakeDiscoveryInfo(common);
        u32 count = 0;
        VkExtensionProperties* properties = nullptr;
        const NVSDK_NGX_Result result =
            NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&discovery, &count, &properties);
        if (NVSDK_NGX_FAILED(result)) {
            LOG_ERROR(Render_Vulkan, "DLSS: instance extension query failed ({:#x})",
                      static_cast<u32>(result));
            return std::vector<std::string>{};
        }
        std::vector<std::string> names = ExtensionNames(count, properties);
        LOG_INFO(Render_Vulkan, "DLSS: NGX asks for {} instance extensions: {}", names.size(),
                 fmt::join(names, ", "));
        return names;
    }();
    return extensions;
}

std::vector<std::string> DeviceExtensions(VkInstance instance, VkPhysicalDevice physical) {
    if (!Requested()) {
        return {};
    }
    const NVSDK_NGX_FeatureCommonInfo common = MakeCommonInfo();
    const NVSDK_NGX_FeatureDiscoveryInfo discovery = MakeDiscoveryInfo(common);
    u32 count = 0;
    VkExtensionProperties* properties = nullptr;
    const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
        instance, physical, &discovery, &count, &properties);
    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR(Render_Vulkan, "DLSS: device extension query failed ({:#x})",
                  static_cast<u32>(result));
        return {};
    }
    std::vector<std::string> names = ExtensionNames(count, properties);
    LOG_INFO(Render_Vulkan, "DLSS: NGX asks for {} device extensions: {}", names.size(),
             fmt::join(names, ", "));
    return names;
}

} // namespace Dlss

DlssContext::DlssContext(const Device& device_, Scheduler& scheduler_,
                         MemoryAllocator& memory_allocator_, DescriptorPool& descriptor_pool,
                         ComputePassDescriptorQueue& compute_pass_descriptor_queue)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    const auto& dld = device.GetDispatchLoader();
    const NVSDK_NGX_FeatureCommonInfo common = MakeCommonInfo();
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        NGX_PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, NGX_ENGINE_VERSION,
        ApplicationDataPath().c_str(), device.GetInstance(), device.GetPhysical(),
        *device.GetLogical(), dld.vkGetInstanceProcAddr, dld.vkGetDeviceProcAddr, &common);
    if (NVSDK_NGX_FAILED(result)) {
        LOG_ERROR(Render_Vulkan, "DLSS: NGX failed to start ({:#x})", static_cast<u32>(result));
        return;
    }
    initialized = true;

    result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&capabilities);
    if (NVSDK_NGX_FAILED(result) || capabilities == nullptr) {
        LOG_ERROR(Render_Vulkan, "DLSS: NGX capability query failed ({:#x})",
                  static_cast<u32>(result));
        capabilities = nullptr;
        return;
    }
    int supported = 0;
    int needs_newer_driver = 0;
    unsigned int min_driver_major = 0;
    unsigned int min_driver_minor = 0;
    int init_result = 0;
    NVSDK_NGX_Parameter_GetI(capabilities, NVSDK_NGX_Parameter_SuperSampling_Available, &supported);
    NVSDK_NGX_Parameter_GetI(capabilities, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                             &needs_newer_driver);
    NVSDK_NGX_Parameter_GetUI(capabilities, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,
                              &min_driver_major);
    NVSDK_NGX_Parameter_GetUI(capabilities, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,
                              &min_driver_minor);
    NVSDK_NGX_Parameter_GetI(capabilities, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
                             &init_result);
    available = supported != 0;
    LOG_INFO(Render_Vulkan,
             "DLSS Super Resolution: available={} needs_newer_driver={} min_driver={}.{} "
             "init_result={:#x}",
             available, needs_newer_driver != 0, min_driver_major, min_driver_minor,
             static_cast<u32>(init_result));
    if (!available) {
        return;
    }
    LogOptimalSettings(1920, 1080);
    LogOptimalSettings(2560, 1440);

    const Options& options = GetOptions();
    LOG_INFO(Render_Vulkan,
             "DLSS options: preset {} jitter={} motion_vectors={} hdr={} depth_inverted={} "
             "jitter_sign=({},{})",
             options.preset_name, options.jitter, options.motion_vectors, options.hdr,
             options.depth_inverted, options.jitter_sign_x, options.jitter_sign_y);

    motion_vector_pass = std::make_unique<DlssMotionVectorPass>(device, scheduler, descriptor_pool,
                                                                compute_pass_descriptor_queue);
    depth_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

DlssContext::~DlssContext() {
    if (feature != nullptr) {
        NVSDK_NGX_VULKAN_ReleaseFeature(feature);
    }
    if (feature_parameters != nullptr) {
        NVSDK_NGX_VULKAN_DestroyParameters(feature_parameters);
    }
    if (capabilities != nullptr) {
        NVSDK_NGX_VULKAN_DestroyParameters(capabilities);
    }
    if (initialized) {
        NVSDK_NGX_VULKAN_Shutdown1(*device.GetLogical());
    }
}

std::array<float, 2> DlssContext::Jitter() const noexcept {
    if (!available || feature_failed || !GetOptions().jitter) {
        return {};
    }
    const u32 index = jitter_index + 1;
    return {Halton(index, 2) - 0.5f, Halton(index, 3) - 0.5f};
}

bool DlssContext::Evaluate(const DlssFrame& frame) {
    if (!available || feature_failed || !motion_vector_pass) {
        return false;
    }
    const Options& options = GetOptions();
    const u32 render_width = static_cast<u32>(std::lround(std::abs(frame.viewport[0]) * 2.0f));
    const u32 render_height = static_cast<u32>(std::lround(std::abs(frame.viewport[1]) * 2.0f));
    if (render_width == 0 || render_height == 0 || render_width > frame.color_width ||
        render_height > frame.color_height || render_width > frame.depth_width ||
        render_height > frame.depth_height || frame.output_width == 0 ||
        frame.output_height == 0) {
        LOG_WARNING(Render_Vulkan, "DLSS: skipped a frame with render size {}x{} (color {}x{}, "
                                   "depth {}x{}, output {}x{})",
                    render_width, render_height, frame.color_width, frame.color_height,
                    frame.depth_width, frame.depth_height, frame.output_width, frame.output_height);
        return false;
    }
    EnsureImages(render_width, render_height, frame.output_width, frame.output_height);

    // From this frame's normalized device coordinates to the previous frame's clip space.
    Matrix camera;
    for (size_t i = 0; i < camera.size(); ++i) {
        camera[i] = frame.camera[i];
    }
    Matrix reprojection = Identity();
    bool reset = evaluations == 0 || !frame.has_camera || !has_previous_camera;
    if (frame.has_camera) {
        const std::optional inverse = Inverse(camera);
        if (inverse && has_previous_camera && options.motion_vectors) {
            reprojection = Multiply(previous_camera, *inverse);
        } else if (!inverse) {
            reset = true;
        }
        previous_camera = camera;
        has_previous_camera = inverse.has_value();
    } else {
        has_previous_camera = false;
    }

    const std::array<float, 2> jitter = Jitter();
    MotionVectorPushConstants constants{};
    for (size_t i = 0; i < constants.reprojection.size(); ++i) {
        constants.reprojection[i] = static_cast<float>(reprojection[i]);
    }
    constants.viewport_xy = {frame.viewport[0], frame.viewport[1], frame.viewport[3],
                             frame.viewport[4]};
    constants.depth_and_jitter = {frame.viewport[2] != 0.0f ? frame.viewport[2] : 1.0f,
                                  frame.viewport[5], jitter[0], jitter[1]};
    constants.size = {render_width, render_height};

    if (evaluations == 0) {
        LOG_INFO(Render_Vulkan,
                 "DLSS: first frame, render {}x{} -> output {}x{}, color format {} depth format {}",
                 render_width, render_height, frame.output_width, frame.output_height,
                 static_cast<u32>(frame.color_format), static_cast<u32>(frame.depth_format));
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    const bool motion_vectors_initialized = motion_vectors.initialized;
    const bool output_initialized = output.initialized;
    motion_vectors.initialized = true;
    output.initialized = true;
    scheduler.Record([motion_vector_image = *motion_vectors.image, output_image = *output.image,
                      motion_vectors_initialized, output_initialized](vk::CommandBuffer cmdbuf) {
        static constexpr VkMemoryBarrier SCENE_TO_COMPUTE{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        const std::array images{
            ImageBarrier(motion_vector_image, VK_IMAGE_ASPECT_COLOR_BIT,
                         motion_vectors_initialized ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_NONE,
                         VK_ACCESS_SHADER_WRITE_BIT,
                         motion_vectors_initialized ? VK_IMAGE_LAYOUT_GENERAL
                                                    : VK_IMAGE_LAYOUT_UNDEFINED),
            ImageBarrier(output_image, VK_IMAGE_ASPECT_COLOR_BIT,
                         output_initialized ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_NONE,
                         VK_ACCESS_SHADER_WRITE_BIT,
                         output_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, SCENE_TO_COMPUTE, {},
                               images);
    });
    motion_vector_pass->Run(frame.depth_view, *depth_sampler, *motion_vectors.view, constants);

    const std::array<float, 2> dlss_jitter{jitter[0] * options.jitter_sign_x,
                                           jitter[1] * options.jitter_sign_y};
    scheduler.Record([this, frame, render_width, render_height, dlss_jitter, reset,
                      motion_vector_image = *motion_vectors.image,
                      motion_vector_view = *motion_vectors.view, output_image = *output.image,
                      output_view = *output.view](vk::CommandBuffer cmdbuf) {
        static constexpr VkMemoryBarrier COMPUTE_TO_COMPUTE{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, COMPUTE_TO_COMPUTE);
        const std::array<u32, 4> size{render_width, render_height, frame.output_width,
                                      frame.output_height};
        if ((feature == nullptr || feature_size != size) &&
            !CreateFeature(*cmdbuf, render_width, render_height, frame.output_width,
                           frame.output_height)) {
            return;
        }
        NVSDK_NGX_Resource_VK color = NVSDK_NGX_Create_ImageView_Resource_VK(
            frame.color_view, frame.color_image, SubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT),
            frame.color_format, frame.color_width, frame.color_height, false);
        NVSDK_NGX_Resource_VK depth = NVSDK_NGX_Create_ImageView_Resource_VK(
            frame.depth_view, frame.depth_image, SubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT),
            frame.depth_format, frame.depth_width, frame.depth_height, false);
        NVSDK_NGX_Resource_VK motion = NVSDK_NGX_Create_ImageView_Resource_VK(
            motion_vector_view, motion_vector_image, SubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT),
            MOTION_VECTOR_FORMAT, render_width, render_height, false);
        NVSDK_NGX_Resource_VK result = NVSDK_NGX_Create_ImageView_Resource_VK(
            output_view, output_image, SubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT), OUTPUT_FORMAT,
            frame.output_width, frame.output_height, true);

        NVSDK_NGX_VK_DLSS_Eval_Params eval{};
        eval.Feature.pInColor = &color;
        eval.Feature.pInOutput = &result;
        eval.pInDepth = &depth;
        eval.pInMotionVectors = &motion;
        eval.InJitterOffsetX = dlss_jitter[0];
        eval.InJitterOffsetY = dlss_jitter[1];
        eval.InRenderSubrectDimensions = {render_width, render_height};
        eval.InReset = reset ? 1 : 0;
        eval.InMVScaleX = 1.0f;
        eval.InMVScaleY = 1.0f;
        const NVSDK_NGX_Result evaluated =
            NGX_VULKAN_EVALUATE_DLSS_EXT(*cmdbuf, feature, feature_parameters, &eval);
        if (NVSDK_NGX_FAILED(evaluated)) {
            LOG_ERROR(Render_Vulkan, "DLSS: evaluation failed ({:#x}), turning DLSS off",
                      static_cast<u32>(evaluated));
            feature_failed = true;
            return;
        }

        // Copy the result into the guest's final render target, converting to its format.
        const std::array to_copy{
            ImageBarrier(output_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL),
            ImageBarrier(frame.output_image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, {}, {}, to_copy);
        const s32 width = static_cast<s32>(frame.output_width);
        const s32 height = static_cast<s32>(frame.output_height);
        const VkImageBlit blit{
            .srcSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .srcOffsets{{0, 0, 0}, {width, height, 1}},
            .dstSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .dstOffsets{{0, 0, 0}, {width, height, 1}},
        };
        cmdbuf.BlitImage(output_image, VK_IMAGE_LAYOUT_GENERAL, frame.output_image,
                         VK_IMAGE_LAYOUT_GENERAL, blit, VK_FILTER_NEAREST);
        const VkImageMemoryBarrier after_copy = ImageBarrier(
            frame.output_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, after_copy);
    });

    jitter_index = (jitter_index + 1) % JITTER_PHASES;
    ++evaluations;
    return true;
}

DlssContext::OwnedImage DlssContext::CreateImage(VkFormat format, u32 width, u32 height,
                                                 VkImageUsageFlags usage) const {
    OwnedImage result;
    result.image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    result.view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *result.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components{
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = SubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT),
    });
    result.format = format;
    result.width = width;
    result.height = height;
    return result;
}

void DlssContext::EnsureImages(u32 render_width, u32 render_height, u32 output_width,
                               u32 output_height) {
    const bool motion_vectors_fit =
        motion_vectors.width == render_width && motion_vectors.height == render_height;
    const bool output_fits = output.width == output_width && output.height == output_height;
    if (motion_vectors_fit && output_fits) {
        return;
    }
    if (motion_vectors.width != 0 || output.width != 0) {
        // The old images may still be in use by recorded or submitted work.
        scheduler.Finish();
    }
    if (!motion_vectors_fit) {
        motion_vectors = CreateImage(MOTION_VECTOR_FORMAT, render_width, render_height,
                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }
    if (!output_fits) {
        output = CreateImage(OUTPUT_FORMAT, output_width, output_height,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    }
}

bool DlssContext::CreateFeature(VkCommandBuffer cmdbuf, u32 render_width, u32 render_height,
                                u32 output_width, u32 output_height) {
    if (feature != nullptr) {
        // NGX may still be using the old feature's resources.
        device.GetLogical().WaitIdle();
        NVSDK_NGX_VULKAN_ReleaseFeature(feature);
        feature = nullptr;
    }
    if (feature_parameters == nullptr &&
        NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&feature_parameters))) {
        LOG_ERROR(Render_Vulkan, "DLSS: could not allocate NGX parameters, turning DLSS off");
        feature_parameters = nullptr;
        feature_failed = true;
        return false;
    }
    const Options& options = GetOptions();
    const bool dlaa = render_width >= output_width && render_height >= output_height;
    NVSDK_NGX_DLSS_Create_Params create{};
    create.Feature.InWidth = render_width;
    create.Feature.InHeight = render_height;
    create.Feature.InTargetWidth = output_width;
    create.Feature.InTargetHeight = output_height;
    create.Feature.InPerfQualityValue =
        dlaa ? NVSDK_NGX_PerfQuality_Value_DLAA : NVSDK_NGX_PerfQuality_Value_MaxQuality;
    int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (options.depth_inverted) {
        flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    }
    if (options.hdr) {
        flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    }
    create.InFeatureCreateFlags = flags;
    create.InEnableOutputSubrects = false;
    for (const char* preset_hint : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
                                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality}) {
        NVSDK_NGX_Parameter_SetUI(feature_parameters, preset_hint, options.preset);
    }
    const NVSDK_NGX_Result result = NGX_VULKAN_CREATE_DLSS_EXT1(
        *device.GetLogical(), cmdbuf, 1, 1, &feature, feature_parameters, &create);
    if (NVSDK_NGX_FAILED(result) || feature == nullptr) {
        LOG_ERROR(Render_Vulkan, "DLSS: feature creation failed ({:#x}), turning DLSS off",
                  static_cast<u32>(result));
        feature = nullptr;
        feature_failed = true;
        return false;
    }
    feature_size = {render_width, render_height, output_width, output_height};
    LOG_INFO(Render_Vulkan, "DLSS: running {} {}x{} -> {}x{}, preset {}, flags {:#x}",
             dlaa ? "DLAA" : "Quality", render_width, render_height, output_width, output_height,
             options.preset_name, flags);
    return true;
}

void DlssContext::LogOptimalSettings(u32 output_width, u32 output_height) const {
    static constexpr NVSDK_NGX_PerfQuality_Value modes[]{
        NVSDK_NGX_PerfQuality_Value_DLAA,        NVSDK_NGX_PerfQuality_Value_UltraQuality,
        NVSDK_NGX_PerfQuality_Value_MaxQuality,  NVSDK_NGX_PerfQuality_Value_Balanced,
        NVSDK_NGX_PerfQuality_Value_MaxPerf,     NVSDK_NGX_PerfQuality_Value_UltraPerformance,
    };
    for (const NVSDK_NGX_PerfQuality_Value mode : modes) {
        const std::optional sizes = OptimalRenderSizes(capabilities, output_width, output_height, mode);
        if (!sizes) {
            LOG_INFO(Render_Vulkan, "DLSS {}x{} {}: not offered", output_width, output_height,
                     QualityName(mode));
            continue;
        }
        LOG_INFO(Render_Vulkan, "DLSS {}x{} {}: render {}x{} (dynamic {}x{} to {}x{})",
                 output_width, output_height, QualityName(mode), sizes->width, sizes->height,
                 sizes->min_width, sizes->min_height, sizes->max_width, sizes->max_height);
    }
}

} // namespace Vulkan
