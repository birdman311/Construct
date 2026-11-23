// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;
class DlssMotionVectorPass;

namespace Dlss {

/// True when NVIDIA DLSS Super Resolution was asked for with the EDEN_DLSS environment variable.
[[nodiscard]] bool Requested();

/// Instance extensions NGX needs for DLSS, empty when DLSS is not requested or NGX cannot say.
[[nodiscard]] const std::vector<std::string>& InstanceExtensions();

/// Device extensions NGX needs for DLSS on this physical device.
[[nodiscard]] std::vector<std::string> DeviceExtensions(VkInstance instance,
                                                       VkPhysicalDevice physical);

} // namespace Dlss

/// What DLSS needs from one guest frame. Images are expected in VK_IMAGE_LAYOUT_GENERAL.
struct DlssFrame {
    /// Scene camera, world to clip without jitter, column-major as the guest uses it.
    std::array<float, 16> camera{};
    /// Guest viewport transform of the scene: scale x, y, z, then translate x, y, z.
    std::array<float, 6> viewport{};
    bool has_camera = false;

    /// Input of the guest's final anti-aliasing pass (the finished, jittered frame).
    VkImage color_image = VK_NULL_HANDLE;
    VkImageView color_view = VK_NULL_HANDLE;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    u32 color_width = 0;
    u32 color_height = 0;

    /// Scene depth, viewed with only the depth aspect.
    VkImage depth_image = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    u32 depth_width = 0;
    u32 depth_height = 0;

    /// Render target of the guest's final pass, which DLSS output is copied into.
    VkImage output_image = VK_NULL_HANDLE;
    u32 output_width = 0;
    u32 output_height = 0;
};

/// NVIDIA DLSS Super Resolution on Eden's Vulkan device. Eden jitters the guest's scene by a
/// sub-pixel viewport offset each frame (Jitter), derives camera motion vectors from the scene
/// depth and camera, and runs DLSS in place of the guest's own final anti-aliasing pass.
class DlssContext {
public:
    explicit DlssContext(const Device& device, Scheduler& scheduler,
                         MemoryAllocator& memory_allocator, DescriptorPool& descriptor_pool,
                         ComputePassDescriptorQueue& compute_pass_descriptor_queue);
    ~DlssContext();

    DlssContext(const DlssContext&) = delete;
    DlssContext& operator=(const DlssContext&) = delete;

    /// True when NGX started and the driver offers DLSS Super Resolution.
    [[nodiscard]] bool IsAvailable() const noexcept {
        return available;
    }

    /// Sub-pixel offset, in render pixels, Eden adds to the scene's viewports this frame.
    [[nodiscard]] std::array<float, 2> Jitter() const noexcept;

    /// Runs DLSS on the frame, copies the result into frame.output_image and moves on to the next
    /// jitter offset. Called on the GPU thread in place of the guest's final pass. Returns false,
    /// recording nothing, when the frame does not fit; the guest's own pass should then run.
    bool Evaluate(const DlssFrame& frame);

private:
    struct OwnedImage {
        vk::Image image;
        vk::ImageView view;
        VkFormat format = VK_FORMAT_UNDEFINED;
        u32 width = 0;
        u32 height = 0;
        bool initialized = false;
    };

    void LogOptimalSettings(u32 output_width, u32 output_height) const;
    OwnedImage CreateImage(VkFormat format, u32 width, u32 height, VkImageUsageFlags usage) const;
    void EnsureImages(u32 render_width, u32 render_height, u32 output_width, u32 output_height);

    /// Creates the NGX feature for these sizes. Runs on the scheduler worker thread.
    bool CreateFeature(VkCommandBuffer cmdbuf, u32 render_width, u32 render_height,
                       u32 output_width, u32 output_height);

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& memory_allocator;
    std::unique_ptr<DlssMotionVectorPass> motion_vector_pass;
    vk::Sampler depth_sampler;

    NVSDK_NGX_Parameter* capabilities = nullptr;
    bool initialized = false;
    bool available = false;

    OwnedImage motion_vectors;
    OwnedImage output;

    // Owned by the scheduler worker thread, where NGX records its work.
    NVSDK_NGX_Parameter* feature_parameters = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    std::array<u32, 4> feature_size{};
    /// Set by the worker thread when NGX fails; DLSS then stays off (read on the GPU thread).
    std::atomic<bool> feature_failed{false};

    std::array<double, 16> previous_camera{};
    bool has_previous_camera = false;
    u32 jitter_index = 0;
    u64 evaluations = 0;
};

} // namespace Vulkan
