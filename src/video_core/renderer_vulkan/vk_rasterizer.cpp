// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <span>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/renderer_vulkan.h"

#include "common/assert.h"
#include "shader_recompiler/exception.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/launch_log.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/native_query_trace.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/native_debug_env.h"
#include "video_core/nx2_titles.h"
// Test build: the frame capture's PNG writer, compiled here because frame_dump.cpp is not in the
// CMake source list (adding it re-runs CMake, which fails on this PC at httplib's OpenSSL check).
#include "video_core/frame_dump.cpp"
#include "video_core/texture_cache/util.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;


namespace {
// Test-build diagnostic: which pipelines ran in each presented frame. A frame whose set differs
// from both neighbours (which match each other) and that is rare in the last 120 frames is logged
// with the pipelines it lacks and the extra ones it has (the cutscene movie's one-frame dropouts).
struct FrameDiag {
    std::vector<u64> current;
    u32 draws{};
    u32 dispatches{};
    struct Past {
        u64 signature;
        std::vector<u64> set;
        u32 draws;
        u32 dispatches;
    };
    std::deque<Past> history;
    std::map<u64, u32> recent_counts;
    u64 frame{};
    u32 logged{};
    std::deque<std::vector<u64>> movie_recent;
    std::map<u64, u32> movie_seen;
    u64 movie_frames{};
    u32 movie_logged{};
    u64 PipelineId(const std::array<u64, 6>& hashes) const {
        u64 h = 0xcbf29ce484222325ULL;
        for (const u64 value : hashes) {
            h = (h ^ value) * 0x100000001b3ULL;
        }
        return h;
    }
    void Note(u64 id) {
        current.push_back(id);
    }
    void EndFrame() {
        std::sort(current.begin(), current.end());
        current.erase(std::unique(current.begin(), current.end()), current.end());
        u64 signature = 0xcbf29ce484222325ULL;
        for (const u64 value : current) {
            signature = (signature ^ value) * 0x100000001b3ULL;
        }
        history.push_back({signature, current, draws, dispatches});
        recent_counts[signature]++;
        if (history.size() > 120) {
            const u64 old = history.front().signature;
            if (--recent_counts[old] == 0) {
                recent_counts.erase(old);
            }
            history.pop_front();
        }
        ++frame;
        // While a movie plays: a frame that draws pipelines none of the previous 30 frames drew
        // (the field interface appearing over the cutscene) is logged with those pipelines.
        if (Core::LaunchLog::g_movie_timeline.load(std::memory_order_relaxed)) {
            std::vector<u64> fresh;
            for (const u64 id : current) {
                if (movie_seen.find(id) == movie_seen.end()) {
                    fresh.push_back(id);
                }
            }
            if (movie_frames >= 30 && !fresh.empty() && movie_logged < 5000) {
                ++movie_logged;
                std::string text;
                for (size_t i = 0; i < fresh.size() && i < 8; ++i) {
                    text += fmt::format(" {:08x}", static_cast<u32>(fresh[i]));
                }
                Core::LaunchLog::Write("TL frame {} draws {} dispatches {} pipelines {} NEW {}:{}", frame,
                                       draws, dispatches, current.size(), fresh.size(), text);
            }
            movie_recent.push_back(current);
            for (const u64 id : current) {
                movie_seen[id]++;
            }
            if (movie_recent.size() > 30) {
                for (const u64 id : movie_recent.front()) {
                    if (--movie_seen[id] == 0) {
                        movie_seen.erase(id);
                    }
                }
                movie_recent.pop_front();
            }
            ++movie_frames;
        } else if (movie_frames != 0) {
            movie_frames = 0;
            movie_recent.clear();
            movie_seen.clear();
        }
        if (false && history.size() >= 3) {
            const size_t n = history.size();
            const Past& before = history[n - 3];
            const Past& blip = history[n - 2];
            const Past& after = history[n - 1];
            if (blip.signature != before.signature && before.signature == after.signature &&
                recent_counts[blip.signature] <= 1 && logged < 600 && Core::LaunchLog::Active()) {
                ++logged;
                std::vector<u64> missing, extra;
                std::set_difference(before.set.begin(), before.set.end(), blip.set.begin(),
                                    blip.set.end(), std::back_inserter(missing));
                std::set_difference(blip.set.begin(), blip.set.end(), before.set.begin(),
                                    before.set.end(), std::back_inserter(extra));
                std::string miss_text, extra_text;
                for (size_t i = 0; i < missing.size() && i < 6; ++i) {
                    miss_text += fmt::format(" {:08x}", static_cast<u32>(missing[i]));
                }
                for (size_t i = 0; i < extra.size() && i < 6; ++i) {
                    extra_text += fmt::format(" {:08x}", static_cast<u32>(extra[i]));
                }
                Core::LaunchLog::Write("frame blip at presented frame {}: draws {} (neighbours {}), dispatches {} "
                                       "(neighbours {}), pipelines missing {}:{} extra {}:{}",
                                       frame - 1, blip.draws, before.draws, blip.dispatches,
                                       before.dispatches, missing.size(), miss_text, extra.size(),
                                       extra_text);
            }
        }
        current.clear();
        draws = 0;
        dispatches = 0;
    }
};
FrameDiag g_frame_diag;

// EDEN_FRAME_CAPTURE=1 (test builds): a file named DUMP_NOW in the working directory makes the
// next presented frame get captured to framecap\cap<N>: passes.txt lists every render pass (the
// shader pairs its draws used) and every compute dispatch, in order; the color targets of each
// render pass and the storage images of each dispatch are read back once written, saved as PNG
// and summarised (mean and largest value per channel, NaN and Inf pixel counts).
struct FrameCapture {
    const bool enabled = std::getenv("EDEN_FRAME_CAPTURE") != nullptr;
    bool active = false;
    u32 number = 0;
    u32 pass = 0;
    u32 dispatch = 0;
    std::filesystem::path folder;
    std::ofstream log;
    bool in_pass = false;
    u64 pass_signature = 0;
    u32 pass_draws = 0;
    VideoCommon::RenderTargets targets{};
    std::map<std::string, u32> pass_shaders;
    // Burst: DUMP_NOW containing "burst" captures this many frames in a row (statistics for every
    // target, PNGs only of bright 8-bit targets), to catch one-frame flickers.
    u32 burst_left = 0;
    u32 burst_number = 0;
    u32 burst_frame = 0;
    bool burst_full = false; // "burstfull": every image saved as PNG (6 frames)
    // Texture buffers bound since the last pass / dispatch was logged: "format:size" -> count.
    std::map<std::string, u32> texel_buffers;
    u64 invalid_texel_buffers = 0; // in the captured frame
    // Image views of the draw being prepared, and those of watched fragment shaders in the
    // current pass (described at the pass end): EDEN_CAPTURE_WATCH_PS=<hash>[,<hash>...]
    // (default: the deferred lighting shader 5c65163e4105372b).
    std::vector<u32> draw_views;
    struct TexelBinding { u64 address; u32 size; VideoCore::Surface::PixelFormat format; };
    std::vector<TexelBinding> draw_texel_buffers; // of the draw/dispatch being prepared
    bool last_gpu_modified = false; // of the last buffer read back
    std::string storage_buffers; // storage buffers bound since the last dispatch was logged
    std::vector<std::pair<std::string, TexelBinding>> watched_texel_buffers;
    std::vector<std::pair<std::string, u32>> watched_views; // "<ps hash> slot N", view index
};
FrameCapture g_capture;

bool CaptureWatchesFragment(u64 hash) {
    static const std::vector<u64> watched = [] {
        std::vector<u64> list;
        const char* env = std::getenv("EDEN_CAPTURE_WATCH_PS");
        std::string text = env ? env : "5c65163e4105372b";
        size_t start = 0;
        while (start < text.size()) {
            const size_t end = text.find(',', start);
            list.push_back(std::strtoull(text.substr(start, end - start).c_str(), nullptr, 16));
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return list;
    }();
    return std::find(watched.begin(), watched.end(), hash) != watched.end();
}
std::atomic<u64> g_invalid_texel_buffers_total{0};

std::string CaptureTakeTexelBuffers() {
    std::string text;
    for (const auto& [key, count] : g_capture.texel_buffers) {
        text += fmt::format(" {}x{}", key, count);
    }
    g_capture.texel_buffers.clear();
    return text;
}

float CaptureHalf(u32 h) {
    const u32 sign = (h >> 15) & 1, exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    float value;
    if (exponent == 31) {
        value = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else {
        value = std::ldexp(static_cast<float>(mantissa | 0x400), static_cast<int>(exponent) - 25);
    }
    return sign ? -value : value;
}

// Unsigned small floats of B10G11R11 (5 exponent bits, `bits` mantissa bits).
float CaptureSmallFloat(u32 v, u32 bits) {
    const u32 exponent = v >> bits, mantissa = v & ((1U << bits) - 1);
    if (exponent == 31) {
        return mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    }
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -14 - static_cast<int>(bits));
    }
    return std::ldexp(static_cast<float>(mantissa | (1U << bits)), static_cast<int>(exponent) - 15 - static_cast<int>(bits));
}

std::string CaptureStats(VideoCore::Surface::PixelFormat format, u32 width, u32 height, u32 row_length,
                         std::span<const u8> data, float* mean_rgb = nullptr) {
    using VideoCore::Surface::PixelFormat;
    const u32 bpp = VideoCore::Surface::BytesPerBlock(format);
    if (data.size() < static_cast<size_t>(row_length) * (height - 1) * bpp + static_cast<size_t>(width) * bpp) {
        return "too little data";
    }
    const auto load32 = [](const u8* p) { u32 v; std::memcpy(&v, p, 4); return v; };
    const auto load16 = [](const u8* p) { u16 v; std::memcpy(&v, p, 2); return static_cast<u32>(v); };
    const auto loadf = [](const u8* p) { float v; std::memcpy(&v, p, 4); return v; };
    // Decodes one pixel into out[], returns the channel count (0: format not summarised).
    const auto decode = [&](const u8* p, std::array<float, 4>& out) -> u32 {
        switch (format) {
        case PixelFormat::B10G11R11_FLOAT: {
            const u32 v = load32(p);
            out = {CaptureSmallFloat(v & 0x7ff, 6), CaptureSmallFloat((v >> 11) & 0x7ff, 6), CaptureSmallFloat(v >> 22, 5), 0};
            return 3;
        }
        case PixelFormat::R16G16B16A16_FLOAT:
            for (u32 c = 0; c < 4; ++c) out[c] = CaptureHalf(load16(p + 2 * c));
            return 4;
        case PixelFormat::R32G32B32A32_FLOAT:
            for (u32 c = 0; c < 4; ++c) out[c] = loadf(p + 4 * c);
            return 4;
        case PixelFormat::R16G16_FLOAT:
            out[0] = CaptureHalf(load16(p)); out[1] = CaptureHalf(load16(p + 2));
            return 2;
        case PixelFormat::R32G32_FLOAT:
            out[0] = loadf(p); out[1] = loadf(p + 4);
            return 2;
        case PixelFormat::R16_FLOAT:
            out[0] = CaptureHalf(load16(p));
            return 1;
        case PixelFormat::R32_FLOAT:
            out[0] = loadf(p);
            return 1;
        case PixelFormat::A8B8G8R8_UNORM: case PixelFormat::A8B8G8R8_SRGB:
            for (u32 c = 0; c < 4; ++c) out[c] = p[c] / 255.0f;
            return 4;
        case PixelFormat::B8G8R8A8_UNORM: case PixelFormat::B8G8R8A8_SRGB:
            out = {p[2] / 255.0f, p[1] / 255.0f, p[0] / 255.0f, p[3] / 255.0f};
            return 4;
        case PixelFormat::A2B10G10R10_UNORM: {
            const u32 v = load32(p);
            out = {(v & 1023) / 1023.0f, ((v >> 10) & 1023) / 1023.0f, ((v >> 20) & 1023) / 1023.0f, (v >> 30) / 3.0f};
            return 4;
        }
        case PixelFormat::R8_UNORM:
            out[0] = p[0] / 255.0f;
            return 1;
        case PixelFormat::R8G8_UNORM:
            out[0] = p[0] / 255.0f; out[1] = p[1] / 255.0f;
            return 2;
        case PixelFormat::R16_UNORM:
            out[0] = load16(p) / 65535.0f;
            return 1;
        case PixelFormat::R16G16_UNORM:
            out[0] = load16(p) / 65535.0f; out[1] = load16(p + 2) / 65535.0f;
            return 2;
        case PixelFormat::R16G16B16A16_UNORM:
            for (u32 c = 0; c < 4; ++c) out[c] = load16(p + 2 * c) / 65535.0f;
            return 4;
        case PixelFormat::R32_UINT:
            out[0] = static_cast<float>(load32(p));
            return 1;
        default:
            return 0;
        }
    };
    std::array<double, 4> sum{};
    std::array<float, 4> top{-1e30f, -1e30f, -1e30f, -1e30f};
    u64 nan_pixels = 0, inf_pixels = 0, finite_pixels = 0;
    u32 channels = 0;
    std::array<float, 4> px{};
    float brightest = -1e30f;
    u32 brightest_x = 0, brightest_y = 0;
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            channels = decode(data.data() + (static_cast<size_t>(y) * row_length + x) * bpp, px);
            if (channels == 0) {
                return "no statistics for this format";
            }
            bool nan = false, inf = false;
            for (u32 c = 0; c < channels; ++c) {
                if (std::isnan(px[c])) nan = true;
                else if (std::isinf(px[c])) inf = true;
            }
            if (nan) { ++nan_pixels; continue; }
            if (inf) { ++inf_pixels; continue; }
            ++finite_pixels;
            float brightness = 0.0f;
            for (u32 c = 0; c < channels; ++c) {
                sum[c] += px[c];
                top[c] = std::max(top[c], px[c]);
                if (c < 3) brightness += px[c];
            }
            if (brightness > brightest) {
                brightest = brightness;
                brightest_x = x;
                brightest_y = y;
            }
        }
    }
    if (mean_rgb && finite_pixels) {
        double rgb = 0;
        const u32 color_channels = std::min(channels, 3U);
        for (u32 c = 0; c < color_channels; ++c) rgb += sum[c] / finite_pixels;
        *mean_rgb = static_cast<float>(rgb / color_channels);
    }
    std::string text = "mean";
    for (u32 c = 0; c < channels; ++c) text += fmt::format(" {:.4g}", finite_pixels ? sum[c] / finite_pixels : 0.0);
    text += " max";
    for (u32 c = 0; c < channels; ++c) text += fmt::format(" {:.4g}", top[c]);
    text += fmt::format(" | NaN pixels {} Inf pixels {} | brightest {:.4g} at ({},{})", nan_pixels,
                        inf_pixels, brightest, brightest_x, brightest_y);
    if (width * height == 1 || width * height <= 16) {
        text += " | values:";
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                decode(data.data() + (static_cast<size_t>(y) * row_length + x) * bpp, px);
                text += " (";
                for (u32 c = 0; c < channels; ++c) text += fmt::format("{}{:.6g}", c ? "," : "", px[c]);
                text += ")";
            }
        }
    }
    return text;
}

// Reads back level 0 / layer 0 of the view's image and returns its summary; writes the PNG.
std::string CaptureView(TextureCache& texture_cache, TextureCacheRuntime& runtime, ImageViewId id,
                        const std::string& name) {
    if (!id) {
        return "none";
    }
    ImageView& view = texture_cache.GetImageView(id);
    if (!view.image_id) {
        return "null view";
    }
    Image& image = texture_cache.GetImage(view.image_id);
    const auto& info = image.info;
    if (info.num_samples > 1 || info.type == VideoCommon::ImageType::Buffer ||
        image.unswizzled_size_bytes == 0 || image.unswizzled_size_bytes > (256ULL << 20)) {
        return fmt::format("{} {}x{} not read back", info.format, info.size.width, info.size.height);
    }
    const auto copies = VideoCommon::FullDownloadCopies(info);
    auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
    image.DownloadMemory(map, copies);
    runtime.Finish();
    const u32 width = info.size.width, height = info.size.height;
    const u32 row_length = copies.empty() ? width : copies[0].buffer_row_length;
    const std::span<const u8> data{map.mapped_span.data(), map.mapped_span.size()};
    float mean_rgb = 0.0f;
    std::string text = fmt::format("{} {}x{}{}{}: {}", info.format, width, height,
                                   info.resources.layers > 1 ? fmt::format(" ({} layers)", info.resources.layers) : "",
                                   image.IsRescaled() ? " scaled" : "",
                                   CaptureStats(info.format, width, height, row_length, data, &mean_rgb));
    using VideoCore::Surface::PixelFormat;
    const bool eight_bit = info.format == PixelFormat::A8B8G8R8_UNORM || info.format == PixelFormat::A8B8G8R8_SRGB ||
                           info.format == PixelFormat::B8G8R8A8_UNORM || info.format == PixelFormat::B8G8R8A8_SRGB;
    const bool want_png = g_capture.burst_number == 0 || g_capture.burst_full || (eight_bit && width >= 1280 && mean_rgb > 0.6f);
    if (want_png && VideoCommon::FrameDumpSupportsFormat(info.format)) {
        const std::string file = g_capture.burst_number
            ? fmt::format("f{:02}_{}.png", g_capture.burst_frame, name) : name + ".png";
        VideoCommon::WriteFrameDumpImage(g_capture.folder / file, info.format, width, height, row_length, data);
        text += fmt::format(" [saved {}]", file);
    }
    // Layered images: level 0 holds the layers one after another; summarise each (the first line
    // covers layer 0 only).
    const u32 layers = std::min<u32>(static_cast<u32>(info.resources.layers), 8U);
    const size_t layer_bytes = static_cast<size_t>(row_length) * height *
                               VideoCore::Surface::BytesPerBlock(info.format);
    for (u32 layer = 1; layer < layers && data.size() >= layer_bytes * (layer + 1); ++layer) {
        const auto layer_data = data.subspan(layer_bytes * layer, layer_bytes);
        text += fmt::format("\n      layer {}: {}", layer,
                            CaptureStats(info.format, width, height, row_length, layer_data));
        if ((g_capture.burst_number == 0 || g_capture.burst_full) && VideoCommon::FrameDumpSupportsFormat(info.format)) {
            VideoCommon::WriteFrameDumpImage(g_capture.folder / (g_capture.burst_number ? fmt::format("f{:02}_{}_layer{}.png", g_capture.burst_frame, name, layer) : fmt::format("{}_layer{}.png", name, layer)),
                                             info.format, width, height, row_length, layer_data);
        }
    }
    return text;
}

using CaptureBufferReader = std::function<std::vector<u8>(u64 address, u32 size)>;

void CaptureFlushPass(TextureCache& texture_cache, TextureCacheRuntime& runtime,
                      const CaptureBufferReader& read_buffer) {
    if (!g_capture.in_pass) {
        return;
    }
    g_capture.in_pass = false;
    const u32 pass = g_capture.pass++;
    g_capture.log << fmt::format("pass {}: {} draws; shaders (vs/gs/ps: draws):", pass, g_capture.pass_draws);
    for (const auto& [shaders, count] : g_capture.pass_shaders) {
        g_capture.log << fmt::format(" {}: {};", shaders, count);
    }
    g_capture.log << '\n';
    if (const std::string texel = CaptureTakeTexelBuffers(); !texel.empty()) {
        g_capture.log << "  texture buffers (format:size x binds):" << texel << '\n';
    }
    for (u32 i = 0; i < g_capture.targets.color_buffer_ids.size(); ++i) {
        const ImageViewId id = g_capture.targets.color_buffer_ids[i];
        if (!id) continue;
        g_capture.log << fmt::format("  rt{} -> pass{:03}_rt{}.png: {}\n", i, pass, i,
                                     CaptureView(texture_cache, runtime, id, fmt::format("pass{:03}_rt{}", pass, i)));
    }
    // Inputs of watched fragment shaders drawn in this pass (after the targets: read back once
    // the pass is complete, outside the render pass).
    for (const auto& [label, index] : g_capture.watched_views) {
        ImageViewId id{};
        id.index = index;
        g_capture.log << fmt::format("  input {} -> {}\n", label,
                                     CaptureView(texture_cache, runtime, id,
                                                 fmt::format("pass{:03}_in_{}", pass, label.substr(label.size() - 2))));
    }
    g_capture.watched_views.clear();
    // Contents of the watched draws' texture buffers (light data), as the GPU sees them.
    for (const auto& [label, binding] : g_capture.watched_texel_buffers) {
        const u32 bytes = std::min<u32>(binding.size, 4096);
        const std::vector<u8> data = read_buffer ? read_buffer(binding.address, bytes) : std::vector<u8>{};
        std::string text = fmt::format("  {} -> fmt {} {} addr {:#x} size {:#x}:", label, binding.format,
                                       g_capture.last_gpu_modified ? "GPU-written" : "cpu",
                                       binding.address, binding.size);
        using VideoCore::Surface::PixelFormat;
        const size_t words = std::min<size_t>(data.size() / 4, 192);
        for (size_t w = 0; w < words; ++w) {
            u32 word;
            std::memcpy(&word, data.data() + w * 4, 4);
            float value;
            std::memcpy(&value, &word, 4);
            if (binding.format == PixelFormat::R32_UINT || binding.format == PixelFormat::R32_SINT) {
                text += fmt::format(" {}", word);
            } else {
                text += fmt::format(" {:.5g}", value);
            }
        }
        g_capture.log << text << '\n';
    }
    g_capture.watched_texel_buffers.clear();
    g_capture.log.flush();
}

// The render-target registers a draw would use: a change starts a new captured pass.
u64 CaptureTargetSignature(const Maxwell& regs) {
    u64 signature = regs.rt_control.count;
    for (u32 i = 0; i < regs.rt_control.count && i < 8; ++i) {
        signature = signature * 0x100000001b3ULL ^ regs.rt[regs.rt_control.Map(i)].Address();
    }
    return signature * 0x100000001b3ULL ^ (regs.zeta_enable ? regs.zeta.Address() : 0);
}
} // Anonymous namespace

void CaptureNoteTexelBuffer(VideoCore::Surface::PixelFormat format, u32 size, bool substituted, u64 address) {
    if (substituted) {
        g_invalid_texel_buffers_total.fetch_add(1, std::memory_order_relaxed);
    }
    if (!g_capture.active) {
        return;
    }
    if (substituted) {
        ++g_capture.invalid_texel_buffers;
    }
    g_capture.draw_texel_buffers.push_back({address, size, format});
    ++g_capture.texel_buffers[substituted ? fmt::format("INVALID:{:#x}", size)
                                          : fmt::format("{}:{:#x}@{:#x}", format, size, address)];
}

void CaptureNoteStorageBuffer(u64 address, u32 size, bool written) {
    if (!g_capture.active || g_capture.storage_buffers.size() >= 4000) {
        return;
    }
    g_capture.storage_buffers += fmt::format(" {}{:#x}+{:#x}", written ? "W:" : "R:", address, size);
}

bool CaptureActive() {
    return g_capture.active;
}

void CaptureNoteGraphicsViews(const u32* view_indices, size_t count) {
    g_capture.draw_views.assign(view_indices, view_indices + count);
}

namespace {
struct DrawParams {
    u32 base_instance;
    u32 num_instances;
    u32 base_vertex;
    u32 num_vertices;
    u32 first_index;
    bool is_indexed;
};

VkViewport GetViewportState(const Device& device, const Maxwell& regs, size_t index, float scale,
                             std::array<float, 2> jitter = {}) {
    const auto& src = regs.viewport_transform[index];
    const auto conv = [scale](float value) {
        float new_value = value * scale;
        if (scale < 1.0f) {
            const bool sign = std::signbit(value);
            new_value = std::round(std::abs(new_value));
            new_value = sign ? -new_value : new_value;
        }
        return new_value;
    };
    const float x = conv(src.translate_x - src.scale_x);
    const float width = conv(src.scale_x * 2.0f);
    float y = conv(src.translate_y - src.scale_y);
    float height = conv(src.scale_y * 2.0f);

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool y_negate = !device.IsNvViewportSwizzleSupported() &&
                          src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY;

    if (lower_left) {
        // Flip by surface clip height
        y += conv(static_cast<f32>(regs.surface_clip.height));
        height = -height;
    }

    if (y_negate) {
        // Flip by viewport height
        y += height;
        height = -height;
    }

    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    VkViewport viewport{
        .x = x + jitter[0] * scale,
        .y = y + jitter[1] * scale,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .minDepth = src.translate_z - src.scale_z * reduce_z,
        .maxDepth = src.translate_z + src.scale_z,
    };
    if (!device.IsExtDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::clamp(viewport.minDepth, 0.0f, 1.0f);
        viewport.maxDepth = std::clamp(viewport.maxDepth, 0.0f, 1.0f);
    }
    return viewport;
}

VkRect2D GetScissorState(const Maxwell& regs, size_t index, u32 up_scale = 1, u32 down_shift = 0) {
    const auto& src = regs.scissor_test[index];
    VkRect2D scissor{};
    const auto scale_up = [&](s32 value) -> s32 {
        if (value == 0) {
            return 0U;
        }
        const s32 upset = value * up_scale;
        s32 acumm = 0;
        if ((up_scale >> down_shift) == 0) {
            acumm = upset % 2;
        }
        const s32 converted_value = (value * up_scale) >> down_shift;
        return value < 0 ? std::min<s32>(converted_value - acumm, -1)
                         : std::max<s32>(converted_value + acumm, 1);
    };

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;

    // Flip coordinates if lower left
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();

    // Bound to render area
    min_y = (std::max)(min_y, 0);
    max_y = (std::max)(max_y, 0);

    if (src.enable) {
        scissor.offset.x = scale_up(src.min_x);
        scissor.offset.y = scale_up(min_y);
        scissor.extent.width = scale_up(src.max_x - src.min_x);
        scissor.extent.height = scale_up(max_y - min_y);
    } else {
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = (std::numeric_limits<s32>::max)();
        scissor.extent.height = (std::numeric_limits<s32>::max)();
    }
    return scissor;
}

DrawParams MakeDrawParams(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state, u32 num_instances, bool is_indexed) {
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    // 6 triangle vertices per quad, base vertex is part of the index
    // See BindQuadIndexBuffer for more details
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return false;
    default:
        return true;
    }
}

bool IsPrimitiveRestartSupported(const Device& device, VkPrimitiveTopology topology) {
    return ((topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsTopologyListPrimitiveRestartSupported()) ||
            SupportsPrimitiveRestart(topology) ||
            (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsPatchListPrimitiveRestartSupported()));
}
} // Anonymous namespace

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                                   Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                   const Device& device_, MemoryAllocator& memory_allocator_,
                                   StateTracker& state_tracker_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_},
      memory_allocator{memory_allocator_}, state_tracker{state_tracker_}, scheduler{scheduler_},
      staging_pool(device, memory_allocator, scheduler), descriptor_pool(device, scheduler),
      guest_descriptor_queue(device, UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                             device.IsExtDescriptorBufferSupported()),
      compute_pass_descriptor_queue(device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE),
      descriptor_buffer_ring(device, memory_allocator),
      blit_image(device, scheduler, state_tracker, descriptor_pool), render_pass_cache(device),
      texture_cache_runtime{
          device,     scheduler,         memory_allocator, staging_pool,
          blit_image, render_pass_cache, descriptor_pool,  compute_pass_descriptor_queue},
      texture_cache(texture_cache_runtime, device_memory),
      buffer_cache_runtime(device, memory_allocator, scheduler, staging_pool,
                           guest_descriptor_queue, compute_pass_descriptor_queue, descriptor_pool),
      buffer_cache(device_memory, buffer_cache_runtime),
      query_cache_runtime(this, device_memory, buffer_cache, device, memory_allocator, scheduler,
                          staging_pool, compute_pass_descriptor_queue, descriptor_pool, texture_cache),
      query_cache(gpu, *this, device_memory, query_cache_runtime),
      pipeline_cache(device_memory, device, scheduler, descriptor_pool, guest_descriptor_queue,
                     descriptor_buffer_ring, render_pass_cache, buffer_cache, texture_cache,
                     gpu.ShaderNotify()),
      accelerate_dma(buffer_cache, texture_cache, scheduler),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, device, scheduler),
      wfi_event(device.GetLogical().CreateEvent()) {
    scheduler.SetQueryCache(query_cache);
#ifdef HAS_DLSS
    if (Dlss::Requested()) {
        dlss = std::make_unique<DlssContext>(device, scheduler, memory_allocator, descriptor_pool,
                                             compute_pass_descriptor_queue);
    }
#endif
}

RasterizerVulkan::~RasterizerVulkan() {
    scheduler.WaitWorker();
    scheduler.Finish();
}

template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func) {

    SCOPE_EXIT {
        gpu.TickWork();
    };
    bool submitted = false;
    SCOPE_EXIT {
        auto& native = maxwell3d->GetNativeMacroEngine();
        if (native.native_turing && !submitted) {
            LOG_DEBUG(HW_GPU, "Native graphics draw skipped");
        }
    };
    FlushWork();
    gpu_memory->FlushCaching();

    GraphicsPipeline* const pipeline{pipeline_cache.CurrentGraphicsPipeline()};
    if (!pipeline) {
        return;
    }
    if (Core::LaunchLog::MovieTraceEnabled()) {
        g_frame_diag.Note(g_frame_diag.PipelineId(pipeline_cache.CurrentGraphicsShaderHashes()));
        ++g_frame_diag.draws;
    }
#ifdef HAS_DLSS
    if (dlss && DlssBeforeDraw()) {
        return;
    }
#endif
    if (const std::optional bloom = Tegra::Nx2Bloom();
        bloom && pipeline_cache.CurrentGraphicsShaderHashes()[5] == bloom->composite_fragment) {
        ScaleBloomIntensity(*bloom);
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // Frame capture: read back the previous pass's targets before this draw opens a new pass.
    if (g_capture.active && g_capture.in_pass &&
        CaptureTargetSignature(maxwell3d->regs) != g_capture.pass_signature) {
        CaptureFlushPass(texture_cache, texture_cache_runtime, [this](u64 address, u32 size) {
            g_capture.last_gpu_modified = buffer_cache.IsRegionGpuModified(address, size);
            FlushRegion(address, size);
            std::vector<u8> bytes(size);
            device_memory.ReadBlockUnsafe(address, bytes.data(), size);
            return bytes;
        });
    }
    // update engine as channel may be different.
    pipeline->SetEngine(maxwell3d, gpu_memory);
    if (!pipeline->Configure(is_indexed))
        return;
#ifdef HAS_DLSS
    if (dlss) {
        DlssAfterConfigure();
    }
#endif

    UpdateDynamicStates();

    query_cache.NotifySegment(true);
    HandleTransformFeedback();
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    draw_func();
    submitted = true;
    if (g_capture.active) {
        if (!g_capture.in_pass) {
            g_capture.in_pass = true;
            g_capture.pass_signature = CaptureTargetSignature(maxwell3d->regs);
            g_capture.targets = texture_cache.CurrentRenderTargets();
            g_capture.pass_draws = 0;
            g_capture.pass_shaders.clear();
        }
        const auto& hashes = pipeline_cache.CurrentGraphicsShaderHashes();
        ++g_capture.pass_draws;
        ++g_capture.pass_shaders[fmt::format("{:016x}/{}/{:016x}", hashes[1],
                                             hashes[4] ? fmt::format("{:016x}", hashes[4]) : std::string{"-"},
                                             hashes[5])];
        if (CaptureWatchesFragment(hashes[5])) {
            for (size_t t = 0; t < g_capture.draw_texel_buffers.size() && g_capture.watched_texel_buffers.size() < 32; ++t) {
                g_capture.watched_texel_buffers.emplace_back(fmt::format("draw{} texbuf {}", g_capture.pass_draws, t),
                                                             g_capture.draw_texel_buffers[t]);
            }
            // The fragment stage's constant buffers as the draw sees them in guest memory: nonzero
            // words of banks 0-5 (first 2 KiB each), to compare frames.
            for (u32 bank = 0; bank < 6; ++bank) {
                const auto& cbuf = maxwell3d->state.shader_stages[4].const_buffers[bank];
                if (!cbuf.enabled || cbuf.size == 0) continue;
                const size_t bytes = std::min<size_t>(cbuf.size, 2048);
                if (!gpu_memory->IsFullyMappedRange(cbuf.address, bytes)) continue;
                std::vector<u32> words(bytes / 4);
                gpu_memory->ReadBlockUnsafe(cbuf.address, words.data(), words.size() * 4);
                std::string text = fmt::format("  cbuf draw{} ps {:016x} bank {} addr {:#x} size {:#x}:",
                                               g_capture.pass_draws, hashes[5], bank, cbuf.address, cbuf.size);
                for (size_t w = 0; w < words.size(); ++w) {
                    if (words[w] == 0) continue;
                    float value;
                    std::memcpy(&value, &words[w], 4);
                    text += fmt::format(" {}={:.6g}", w, value);
                }
                g_capture.log << text << '\n';
            }
            for (size_t slot = 0; slot < g_capture.draw_views.size() && g_capture.watched_views.size() < 64; ++slot) {
                const u32 index = g_capture.draw_views[slot];
                if (index == std::numeric_limits<u32>::max()) continue;
                const std::string label = fmt::format("draw{} ps {:016x} slot {:02}", g_capture.pass_draws, hashes[5], slot);
                g_capture.watched_views.emplace_back(label, index);
            }
        }
    }
    g_capture.draw_texel_buffers.clear();
    g_capture.storage_buffers.clear(); // draws: only compute storage buffers are logged
    static u32 native_capture_count{};
    if (maxwell3d->GetNativeMacroEngine().native_turing &&
        VideoCommon::NativeDrawCapture() && native_capture_count < 16) {
        const u32 capture = native_capture_count++;
        const auto prefix = fmt::format("native-draw-{}", capture);
        const auto dump = [&](const std::string& name, GPUVAddr address, size_t size) {
            if (!size || !gpu_memory->IsFullyMappedRange(address, size)) return;
            std::vector<u8> bytes(size);
            gpu_memory->ReadBlockUnsafe(address, bytes.data(), bytes.size());
            std::ofstream file(prefix + name + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            LOG_INFO(HW_GPU, "Native draw capture {}{}: address={:#x}, bytes={}, nonzero={}",
                     capture, name, address, size,
                     std::count_if(bytes.begin(), bytes.end(), [](u8 byte) { return byte != 0; }));
        };
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const size_t size = static_cast<size_t>(rt.array_pitch) * 4;
        if (size && size <= 16 * 1024 * 1024) {
            if (const auto address = gpu_memory->GpuToCpuAddress(rt.Address(), size)) {
                LOG_INFO(HW_GPU, "Native draw {} render target physical address={:#x}", capture, *address);
                texture_cache.DownloadMemory(*address, size);
                dump("-rt0", rt.Address(), size);
            }
        }
        std::ofstream registers(prefix + "-registers.bin", std::ios::binary);
        registers.write(reinterpret_cast<const char*>(&regs), sizeof(regs));
        for (size_t stream = 0; stream < 4; ++stream) {
            if (regs.vertex_streams[stream].IsEnabled())
                dump(fmt::format("-vertex-{}", stream), regs.vertex_streams[stream].Address(), 128);
        }
        for (size_t stage : {0U, 4U}) {
            for (size_t bank : {2U, 3U}) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                if (cb.enabled) dump(fmt::format("-cbuf-{}-{}", stage, bank), cb.address, std::min<size_t>(cb.size, 128));
            }
        }
    }
}

void RasterizerVulkan::Draw(bool is_indexed, u32 instance_count) {
    PrepareDraw(is_indexed, [this, is_indexed, instance_count] {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 num_instances{instance_count};
        const DrawParams draw_params{MakeDrawParams(draw_state, num_instances, is_indexed)};

        scheduler.Record([draw_params](vk::CommandBuffer cmdbuf) {
            if (draw_params.is_indexed) {
                cmdbuf.DrawIndexed(draw_params.num_vertices, draw_params.num_instances,
                                   draw_params.first_index, draw_params.base_vertex,
                                   draw_params.base_instance);
            } else {
                cmdbuf.Draw(draw_params.num_vertices, draw_params.num_instances,
                            draw_params.base_vertex, draw_params.base_instance);
            }
        });

        // Log draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string params = is_indexed ?
                fmt::format("vertices={}, instances={}, firstIndex={}, baseVertex={}, baseInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.first_index, draw_params.base_vertex, draw_params.base_instance) :
                fmt::format("vertices={}, instances={}, firstVertex={}, firstInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.base_vertex, draw_params.base_instance);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw", params, VK_SUCCESS);
        }
    });
}

void RasterizerVulkan::DrawIndirect() {
    const auto& params = maxwell3d->draw_manager.indirect_state;
    buffer_cache.SetDrawIndirect(&params);
    PrepareDraw(params.is_indexed, [this, &params] {
        const auto indirect_buffer = buffer_cache.GetDrawIndirectBuffer();
        const auto& buffer = indirect_buffer.first;
        const auto& offset = indirect_buffer.second;
        if (params.is_byte_count) {
            scheduler.Record([buffer_obj = buffer->Handle(), offset,
                              stride = params.stride](vk::CommandBuffer cmdbuf) {
                cmdbuf.DrawIndirectByteCountEXT(1, 0, buffer_obj, offset, 0,
                                                static_cast<u32>(stride));
            });
            return;
        }
        if (params.include_count) {
            const auto count = buffer_cache.GetDrawIndirectCount();
            const auto& draw_buffer = count.first;
            const auto& offset_base = count.second;
            scheduler.Record([draw_buffer_obj = draw_buffer->Handle(),
                              buffer_obj = buffer->Handle(), offset_base, offset,
                              params](vk::CommandBuffer cmdbuf) {
                if (params.is_indexed) {
                    cmdbuf.DrawIndexedIndirectCount(
                        buffer_obj, offset, draw_buffer_obj, offset_base,
                        static_cast<u32>(params.max_draw_counts), static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirectCount(buffer_obj, offset, draw_buffer_obj, offset_base,
                                             static_cast<u32>(params.max_draw_counts),
                                             static_cast<u32>(params.stride));
                }
            });
            return;
        }
        scheduler.Record([buffer_obj = buffer->Handle(), offset, params](vk::CommandBuffer cmdbuf) {
            if (params.is_indexed) {
                cmdbuf.DrawIndexedIndirect(buffer_obj, offset,
                                           static_cast<u32>(params.max_draw_counts),
                                           static_cast<u32>(params.stride));
            } else {
                cmdbuf.DrawIndirect(buffer_obj, offset, static_cast<u32>(params.max_draw_counts),
                                    static_cast<u32>(params.stride));
            }
        });

        // Log indirect draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string log_params = fmt::format("drawCount={}, stride={}",
                params.max_draw_counts, params.stride);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect",
                log_params, VK_SUCCESS);
        }
    });
    buffer_cache.SetDrawIndirect(nullptr);
}

void RasterizerVulkan::DrawTexture() {

    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();

    std::scoped_lock l{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);

    SetViewportJitter({});
    UpdateDynamicStates();

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& sampler = texture_cache.GetSampler(draw_texture_state.src_sampler, false);
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();

    const bool src_rescaling = texture_cache.IsRescaling() && texture.IsRescaled();
    const bool dst_rescaling = texture_cache.IsRescaling() && framebuffer->IsRescaled();

    const auto ScaleSrc = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return src_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    const auto ScaleDst = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return dst_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    Region2D dst_region = {Offset2D{.x = ScaleDst(draw_texture_state.dst_x0),
                                    .y = ScaleDst(draw_texture_state.dst_y0)},
                           Offset2D{.x = ScaleDst(draw_texture_state.dst_x1),
                                    .y = ScaleDst(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = ScaleSrc(draw_texture_state.src_x0),
                                    .y = ScaleSrc(draw_texture_state.src_y0)},
                           Offset2D{.x = ScaleSrc(draw_texture_state.src_x1),
                                    .y = ScaleSrc(draw_texture_state.src_y1)}};
    Extent3D src_size = {static_cast<u32>(ScaleSrc(texture.size.width)),
                         static_cast<u32>(ScaleSrc(texture.size.height)), texture.size.depth};
    blit_image.BlitColor(framebuffer, texture.RenderTarget(), texture.ImageHandle(),
                         sampler->Handle(), dst_region, src_region, src_size);
}

void RasterizerVulkan::Clear(u32 layer_count) {
    FlushWork();
    gpu_memory->FlushCaching();

    auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    const VkExtent2D render_area = framebuffer->RenderArea();

    constexpr bool ENABLE_DEFERRED_CLEAR = true;
    const bool color_full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool stencil_partial = use_stencil && framebuffer->HasAspectStencilBit() &&
                                 regs.stencil_front_mask != 0xFF && regs.stencil_front_mask != 0;
    const bool ds_used = use_depth || use_stencil;
    const bool ds_deferrable =
        !ds_used || ((!framebuffer->HasAspectDepthBit() || use_depth) &&
                     (!framebuffer->HasAspectStencilBit() || use_stencil) && !stencil_partial);
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    VkRect2D default_scissor{};
    default_scissor.offset.x = 0;
    default_scissor.offset.y = 0;
    default_scissor.extent.width = (std::numeric_limits<s32>::max)();
    default_scissor.extent.height = (std::numeric_limits<s32>::max)();

    VkClearRect clear_rect{
        .rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0, up_scale, down_shift)
                                               : default_scissor,
        .baseArrayLayer = regs.clear_surface.layer,
        .layerCount = layer_count,
    };
    const auto clamp_rect_to_render_area = [render_area](VkRect2D& rect) -> bool {
        const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
            auto clamp_offset = [&offset, limit]() {
                if (limit == 0) {
                    offset = 0;
                    return;
                }
                offset = std::clamp(offset, 0, static_cast<s32>(limit));
            };

            if (extent == 0) {
                clamp_offset();
                return;
            }
            if (offset < 0) {
                const u32 shrink = (std::min)(extent, static_cast<u32>(-offset));
                extent -= shrink;
                offset = 0;
            }
            if (limit == 0) {
                extent = 0;
                offset = 0;
                return;
            }
            if (offset >= s32(limit)) {
                offset = s32(limit);
                extent = 0;
                return;
            }
            const u64 end_coord = u64(offset) + extent;
            if (end_coord > limit) {
                extent = limit - u32(offset);
            }
        };

        clamp_axis(rect.offset.x, rect.extent.width, render_area.width);
        clamp_axis(rect.offset.y, rect.extent.height, render_area.height);
        return rect.extent.width != 0 && rect.extent.height != 0;
    };
    if (!clamp_rect_to_render_area(clear_rect.rect)) {
        return;
    }

    const bool clear_covers_render_area =
        clear_rect.rect.offset.x == 0 && clear_rect.rect.offset.y == 0 &&
        clear_rect.rect.extent.width >= render_area.width &&
        clear_rect.rect.extent.height >= render_area.height;
    const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && (!regs.clear_control.use_scissor || clear_covers_render_area) &&
                                 regs.clear_surface.layer == 0 &&
                                 !scheduler.IsRenderPassActive() &&
                                 (!use_color || color_full_channels) && ds_deferrable;
    if (!can_defer_clear) {
        scheduler.RequestRenderpass(framebuffer);
    }

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    UpdateViewportsState(regs);

    const u32 color_attachment = regs.clear_surface.RT;
    if (use_color && framebuffer->HasAspectColorBit(color_attachment)) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        bool is_integer = IsPixelFormatInteger(format);
        bool is_signed = IsPixelFormatSignedInteger(format);
        size_t int_size = PixelComponentSizeBitsInteger(format);
        VkClearValue clear_value{};
        if (!is_integer) {
            std::memcpy(clear_value.color.float32, regs.clear_color.data(), regs.clear_color.size() * sizeof(f32));
        } else if (!is_signed) {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.uint32[i] = u32(f32(u64(int_size) << 1U) * regs.clear_color[i]);
        } else {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.int32[i] = s32(f32(s64(int_size - 1) << 1) * (regs.clear_color[i] - 0.5f));
        }

        if (color_full_channels) {
            if (can_defer_clear) {
                scheduler.DeferColorClear(framebuffer, color_attachment, clear_value);
            } else {
                scheduler.Record([color_attachment, clear_value, clear_rect](vk::CommandBuffer cmdbuf) {
                    const VkClearAttachment attachment{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .colorAttachment = color_attachment,
                        .clearValue = clear_value,
                    };
                    cmdbuf.ClearAttachments(attachment, clear_rect);
                });
            }
        } else {
            u8 color_mask = u8(regs.clear_surface.R | regs.clear_surface.G << 1 | regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
            Region2D dst_region = {
                Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
                Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                         .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
            blit_image.ClearColor(framebuffer, color_mask, regs.clear_color, dst_region);
        }
    }

    if (!use_depth && !use_stencil) {
        return;
    }
    VkImageAspectFlags aspect_flags = 0;
    if (use_depth && framebuffer->HasAspectDepthBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (use_stencil && framebuffer->HasAspectStencilBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (aspect_flags == 0) {
        return;
    }

    if (use_stencil && framebuffer->HasAspectStencilBit() && regs.stencil_front_mask != 0xFF &&
        regs.stencil_front_mask != 0) {
        Region2D dst_region = {
            Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
            Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                     .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
        blit_image.ClearDepthStencil(framebuffer, use_depth, regs.clear_depth,
                                     u8(regs.stencil_front_mask), regs.clear_stencil,
                                     regs.stencil_front_func_mask, dst_region);
    } else if (can_defer_clear) {
        VkClearValue ds_value{};
        ds_value.depthStencil.depth = regs.clear_depth;
        ds_value.depthStencil.stencil = regs.clear_stencil;
        scheduler.DeferDepthStencilClear(framebuffer, ds_value);
    } else {
        scheduler.Record([clear_depth = regs.clear_depth, clear_stencil = regs.clear_stencil,
                          clear_rect, aspect_flags](vk::CommandBuffer cmdbuf) {
            VkClearAttachment attachment;
            attachment.aspectMask = aspect_flags;
            attachment.colorAttachment = 0;
            attachment.clearValue.depthStencil.depth = clear_depth;
            attachment.clearValue.depthStencil.stencil = clear_stencil;
            cmdbuf.ClearAttachments(attachment, clear_rect);
        });
    }
}

void RasterizerVulkan::DispatchCompute() {
    FlushWork();
    gpu_memory->FlushCaching();

    ComputePipeline* const pipeline{pipeline_cache.CurrentComputePipeline()};
    if (!pipeline) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute pipeline unavailable");
        return;
    }
    const u64 compute_hash = pipeline_cache.CurrentComputeShaderHash();
    if (Core::LaunchLog::MovieTraceEnabled()) {
        g_frame_diag.Note(compute_hash ^ 0x8000000000000000ULL);
        ++g_frame_diag.dispatches;
    }
    const bool supersampling =
        Settings::values.resolution_info.active && Tegra::Nx2SupersampleEasuShader() != 0;
    const u64 rcas = Tegra::Nx2DisabledRcasShader();
    if (rcas != 0 && compute_hash == rcas) {
        DisableRcasSharpening(); // also the fallback when the supersample copy does not run
    }
    std::scoped_lock lock{texture_cache.mutex, buffer_cache.mutex};
    if (g_capture.active) {
        CaptureFlushPass(texture_cache, texture_cache_runtime, [this](u64 address, u32 size) {
            g_capture.last_gpu_modified = buffer_cache.IsRegionGpuModified(address, size);
            FlushRegion(address, size);
            std::vector<u8> bytes(size);
            device_memory.ReadBlockUnsafe(address, bytes.data(), size);
            return bytes;
        });
    }
    // The HWAI EASU/RCAS passes are replaced below (2x supersample). Their outputs used to be
    // scaled down here (blacklisted written images) and back up by the hook every frame: 4 wasted
    // 4K blits and 4 view/framebuffer invalidations per frame. Keep their scale instead.
    const bool replaceable_pass =
        supersampling && (compute_hash == Tegra::Nx2SupersampleEasuShader() ||
                          (rcas != 0 && compute_hash == rcas));
    if (!pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                             texture_cache, replaceable_pass)) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute resources unavailable");
        return;
    }
    // Frame capture: once the dispatch is recorded, read back the storage images it wrote.
    SCOPE_EXIT {
        if (!g_capture.active) {
            return;
        }
        const u32 number = g_capture.dispatch++;
        const auto& launch = kepler_compute->launch_description;
        g_capture.log << fmt::format("dispatch {}: cs {:016x} grid {}x{}x{}{}\n", number, compute_hash,
                                     launch.grid_dim_x, launch.grid_dim_y, launch.grid_dim_z,
                                     kepler_compute->GetIndirectComputeAddress() ? " (indirect)" : "");
        if (const std::string texel = CaptureTakeTexelBuffers(); !texel.empty()) {
            g_capture.log << "  texture buffers (format:size x binds):" << texel << '\n';
        }
        if (!g_capture.storage_buffers.empty()) {
            g_capture.log << "  storage buffers:" << g_capture.storage_buffers << '\n';
            g_capture.storage_buffers.clear();
        }
        // Watched compute shader (EDEN_CAPTURE_WATCH_CS, default the effect-light writer
        // 1ddca26e525da965): its constant buffers and texture buffer inputs, as the GPU sees them.
        static const u64 watched_cs = [] {
            const char* env = std::getenv("EDEN_CAPTURE_WATCH_CS");
            return env ? std::strtoull(env, nullptr, 16) : 0x1ddca26e525da965ULL;
        }();
        if (compute_hash == watched_cs) {
            const auto& qmd = kepler_compute->launch_description;
            for (u32 bank = 0; bank < 8; ++bank) {
                if ((qmd.const_buffer_enable_mask.Value() & (1U << bank)) == 0) continue;
                const GPUVAddr address = qmd.const_buffer_config[bank].Address();
                const u32 size = std::min<u32>(qmd.const_buffer_config[bank].size, 1024);
                if (size == 0 || !gpu_memory->IsFullyMappedRange(address, size)) continue;
                std::vector<u32> words(size / 4);
                gpu_memory->ReadBlockUnsafe(address, words.data(), words.size() * 4);
                std::string text = fmt::format("  cs cbuf bank {} addr {:#x} size {:#x}:", bank, address,
                                               qmd.const_buffer_config[bank].size.Value());
                for (size_t w = 0; w < words.size(); ++w) {
                    if (words[w] == 0) continue;
                    float value;
                    std::memcpy(&value, &words[w], 4);
                    text += fmt::format(" {}={:#x}/{:.6g}", w, words[w], value);
                }
                g_capture.log << text << '\n';
            }
            for (size_t t = 0; t < g_capture.draw_texel_buffers.size(); ++t) {
                const auto& binding = g_capture.draw_texel_buffers[t];
                const u32 bytes = std::min<u32>(binding.size, 2048);
                const bool gpu_written = buffer_cache.IsRegionGpuModified(binding.address, bytes);
                FlushRegion(binding.address, bytes);
                std::vector<u32> words(bytes / 4);
                device_memory.ReadBlockUnsafe(binding.address, words.data(), words.size() * 4);
                std::string text = fmt::format("  cs texbuf {} fmt {} {} addr {:#x} size {:#x}:", t, binding.format,
                                               gpu_written ? "GPU-written" : "cpu", binding.address, binding.size);
                for (size_t w = 0; w < words.size(); ++w) {
                    float value;
                    std::memcpy(&value, &words[w], 4);
                    text += fmt::format(" {:#x}/{:.5g}", words[w], value);
                }
                g_capture.log << text << '\n';
            }
        }
        g_capture.draw_texel_buffers.clear();
        const auto storages = pipeline->BoundStorageViews();
        for (size_t i = 0; i < storages.size(); ++i) {
            if (!storages[i]) continue;
            g_capture.log << fmt::format("  image{} -> dispatch{:04}_img{}.png: {}\n", i, number, i,
                                         CaptureView(texture_cache, texture_cache_runtime, storages[i],
                                                     fmt::format("dispatch{:04}_img{}", number, i)));
        }
        g_capture.log.flush();
    };
    if (supersampling) {
        if (compute_hash == Tegra::Nx2SupersampleEasuShader() && DownscaleInsteadOfEasu(*pipeline)) {
            return;
        }
        if (rcas != 0 && compute_hash == rcas && CopyInsteadOfRcas(*pipeline)) {
            return;
        }
        if (replaceable_pass) {
            // The hook declined: the game's own pass runs, so its written images must be at 1x
            // (the shader writes with 1x coordinates). Configure again with the usual rule.
            if (!pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                                     texture_cache, false)) {
                return;
            }
        }
    }

    const auto& qmd{kepler_compute->launch_description};
    auto indirect_address = kepler_compute->GetIndirectComputeAddress();
    if (indirect_address) {
        // DispatchIndirect
        static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
        const auto post_op = VideoCommon::ObtainBufferOperation::DiscardWrite;
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(*indirect_address, 12, sync_info, post_op);
        scheduler.RequestOutsideRenderPassOperationContext();
        // Same ordering as a direct dispatch, plus the indirect argument read (AMD/Intel run
        // back-to-back dispatches concurrently; NVIDIA mostly serialises and hid this).
        static constexpr VkMemoryBarrier INDIRECT_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   0, INDIRECT_BARRIER);
        });
        scheduler.Record([pipeline, indirect_buffer = buffer->Handle(),
                          indirect_offset = offset](vk::CommandBuffer cmdbuf) {
            if (!pipeline->IsBound()) {
                return;
            }
            cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
        });
        scheduler.MarkComputeWrites();
        return;
    }
    const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        // Write too: back-to-back dispatches writing the same buffer (write-after-write).
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
    };
    scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, READ_BARRIER); });
    scheduler.Record([pipeline, dim](vk::CommandBuffer cmdbuf) {
        if (!pipeline->IsBound()) {
            return;
        }
        cmdbuf.Dispatch(dim[0], dim[1], dim[2]);
    });
    scheduler.MarkComputeWrites();

    // Log compute dispatch
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string params = fmt::format("groupCountX={}, groupCountY={}, groupCountZ={}",
            dim[0], dim[1], dim[2]);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            "vkCmdDispatch", params, VK_SUCCESS);
    }
}

void RasterizerVulkan::ResetCounter(VideoCommon::QueryType type) {
    switch (type) {
    case VideoCommon::QueryType::ZPassPixelCount64:
    case VideoCommon::QueryType::StreamingByteCount:
    case VideoCommon::QueryType::StreamingPrimitivesSucceeded:
    case VideoCommon::QueryType::VtgPrimitivesOut:
        query_cache.CounterReset(type);
        return;
    default:
        LOG_DEBUG(Render_Vulkan, "Unimplemented counter reset={}", type);
        return;
    }
}

void RasterizerVulkan::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                             VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}

void RasterizerVulkan::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void Vulkan::RasterizerVulkan::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.FlushRegion(addr, size);
    }
}

bool RasterizerVulkan::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        auto area = texture_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    {
        // GPU-written buffer pages the guest CPU reads (Switch 2 titles without full copy-back):
        // a newly learned page is copied back now (one full flush), its later GPU writes at the
        // fences. try_lock: never stall the guest CPU behind a draw; it reads again next frame.
        std::unique_lock lock{buffer_cache.mutex, std::try_to_lock};
        if (lock.owns_lock() && buffer_cache.NoteCpuRead(addr, size)) {
            new_area.preemtive = false;
        }
    }
    return new_area;
}

void RasterizerVulkan::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.InvalidateRegion(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::ShaderCache))) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

void RasterizerVulkan::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
    {
        for (const auto& [addr, size] : sequences) {
            query_cache.InvalidateRegion(addr, size);
            pipeline_cache.InvalidateRegion(addr, size);
        }
    }
}

bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {
    DEBUG_ASSERT(addr != 0 || size != 0);
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

void RasterizerVulkan::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }

    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

bool RasterizerVulkan::TryCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) return true;
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) return false;
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

void RasterizerVulkan::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerVulkan::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerVulkan::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerVulkan::SignalFence(std::function<void()>&& func) {
    fence_manager.SignalFence(std::move(func));
}

void RasterizerVulkan::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerVulkan::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerVulkan::SignalReference() {
    fence_manager.SignalReference();
}

void RasterizerVulkan::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force);
}

void RasterizerVulkan::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerVulkan::WaitForIdle() {
    // Everything but wait pixel operations. This intentionally includes FRAGMENT_SHADER_BIT because
    // fragment shaders can still write storage buffers.
    VkPipelineStageFlags flags =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }

    query_cache.NotifyWFI();

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([event = *wfi_event, flags](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetEvent(event, flags);
        cmdbuf.WaitEvents(event, flags, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, {}, {}, {});
    });
    fence_manager.SignalOrdering();
}

void RasterizerVulkan::FragmentBarrier() {
    // We already put barriers when a render pass finishes
    scheduler.RequestOutsideRenderPassOperationContext();
}

void RasterizerVulkan::TiledCacheBarrier() {
    // TODO: Implementing tiled barriers requires rewriting a good chunk of the Vulkan backend
}

void RasterizerVulkan::FlushCommands() {
    if (draw_counter == 0) {
        return;
    }
    draw_counter = 0;
    scheduler.Flush();
}

void RasterizerVulkan::TickFrame() {
    if (Core::LaunchLog::MovieTraceEnabled()) {
        g_frame_diag.EndFrame();
    }
    if (g_capture.enabled) {
        std::scoped_lock lock{texture_cache.mutex};
        if (g_capture.active) {
            CaptureFlushPass(texture_cache, texture_cache_runtime, [this](u64 address, u32 size) {
            g_capture.last_gpu_modified = buffer_cache.IsRegionGpuModified(address, size);
            FlushRegion(address, size);
            std::vector<u8> bytes(size);
            device_memory.ReadBlockUnsafe(address, bytes.data(), size);
            return bytes;
        });
            g_capture.log << fmt::format("frame end: {} passes, {} dispatches, {} undecodable texture buffers "
                                         "this frame ({} since boot)\n",
                                         g_capture.pass, g_capture.dispatch, g_capture.invalid_texel_buffers,
                                         g_invalid_texel_buffers_total.load(std::memory_order_relaxed));
            g_capture.log.close();
            g_capture.active = false;
            LOG_WARNING(Render_Vulkan, "Frame capture {} written to {}", g_capture.number,
                        g_capture.folder.string());
        }
        const auto start_capture = [](std::filesystem::path folder, const std::string& file) {
            std::error_code error;
            g_capture.folder = std::move(folder);
            std::filesystem::create_directories(g_capture.folder, error);
            g_capture.log.open(g_capture.folder / file);
            g_capture.active = g_capture.log.is_open();
            g_capture.pass = 0;
            g_capture.dispatch = 0;
            g_capture.in_pass = false;
            g_capture.texel_buffers.clear();
            g_capture.invalid_texel_buffers = 0;
        };
        if (g_capture.burst_left > 0) {
            --g_capture.burst_left;
            ++g_capture.burst_frame;
            start_capture(std::filesystem::path{"framecap"} / fmt::format("burst{}", g_capture.burst_number),
                          fmt::format("frame{:02}.txt", g_capture.burst_frame));
            if (g_capture.burst_left == 0) {
                LOG_WARNING(Render_Vulkan, "Frame capture burst {} ends after this frame", g_capture.burst_number);
            }
        } else if (std::error_code error; !g_capture.active && std::filesystem::exists("DUMP_NOW", error)) {
            std::string request;
            std::ifstream("DUMP_NOW") >> request;
            std::filesystem::remove("DUMP_NOW", error);
            if (request == "burst" || request == "burstfull") {
                // 60 consecutive frames; numbered after any earlier bursts in the folder.
                u32 next = 1;
                while (std::filesystem::exists(std::filesystem::path{"framecap"} / fmt::format("burst{}", next), error)) ++next;
                g_capture.burst_number = next;
                g_capture.burst_frame = 1;
                g_capture.burst_full = request == "burstfull";
                g_capture.burst_left = g_capture.burst_full ? 5 : 59;
                start_capture(std::filesystem::path{"framecap"} / fmt::format("burst{}", next), "frame01.txt");
                LOG_WARNING(Render_Vulkan, "Frame capture burst {} started (60 frames)", next);
            } else {
                g_capture.burst_number = 0;
                u32 next = g_capture.number + 1;
                while (std::filesystem::exists(std::filesystem::path{"framecap"} / fmt::format("cap{}", next), error)) ++next;
                g_capture.number = next;
                start_capture(std::filesystem::path{"framecap"} / fmt::format("cap{}", next), "passes.txt");
                LOG_WARNING(Render_Vulkan, "Frame capture {} started", g_capture.number);
            }
        }
    }
    draw_counter = 0;
    guest_descriptor_queue.TickFrame();
    compute_pass_descriptor_queue.TickFrame();
    descriptor_buffer_ring.TickFrame();
    fence_manager.TickFrame();
    staging_pool.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
}

bool RasterizerVulkan::AccelerateConditionalRendering() {
    gpu_memory->FlushCaching();
    return query_cache.AccelerateHostConditionalRendering();
}

bool RasterizerVulkan::HasDrawTransformFeedback() {
    return device.IsTransformFeedbackDrawSupported();
}

bool RasterizerVulkan::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                             const Tegra::Engines::Fermi2D::Surface& dst,
                                             const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerVulkan::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerVulkan::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                                std::span<const u8> memory) {
    auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (gpu_memory->UsesNativeTiling()) {
        u32 first_word{};
        if (memory.size() >= sizeof(first_word)) std::memcpy(&first_word, memory.data(), sizeof(first_word));
        VideoCommon::TraceNativeQuery(11, gpu_memory->GetID(), address, cpu_addr.value_or(0),
            0, first_word, static_cast<u32>(copy_size));
    }
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    {
        std::unique_lock<std::recursive_mutex> lock{buffer_cache.mutex};
        if (!buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    {
        std::scoped_lock lock_texture{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
    query_cache.InvalidateRegion(*cpu_addr, copy_size);
}

std::optional<FramebufferTextureInfo> RasterizerVulkan::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    static u32 display_trace_count{};
    if (VideoCommon::NativeFrameCapture() && display_trace_count++ < 64) {
        LOG_INFO(HW_GPU, "Native display source: address={:#x}, offset={:#x}, dimensions={}x{}, stride={}, accelerated={}, mask={:#x}",
                 framebuffer_addr, config.offset, config.width, config.height, pixel_stride,
                 image_view != nullptr, config.layer_stack_mask);
    }
    if (!image_view) {
        return {};
    }

    query_cache.NotifySegment(false);

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.image = image_view->ImageHandle();
    info.image_view = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    return info;
}

void RasterizerVulkan::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerVulkan::DisableRcasSharpening() {
    // FsrRcasCon writes its constants to the front of constant buffer 3: con.x = exp2(-sharpness)
    // as a float, con.y the same value as two halves, con.zw zero. With both zero the pass copies
    // its input unchanged. Anything that does not look like that is left alone.
    const auto& qmd{kepler_compute->launch_description};
    if ((qmd.const_buffer_enable_mask.Value() & (1U << 3)) == 0) {
        return;
    }
    const GPUVAddr address{qmd.const_buffer_config[3].Address()};
    std::array<u32, 4> con{};
    if (!gpu_memory->IsFullyMappedRange(address, sizeof(con))) {
        return;
    }
    gpu_memory->ReadBlockUnsafe(address, con.data(), sizeof(con));
    const bool is_rcas = con[0] != 0 && (con[1] & 0xffff) == (con[1] >> 16) && con[2] == 0 &&
                         con[3] == 0;
    if (!is_rcas) {
        return;
    }
    // Written like an inline upload so a copy of the buffer already on the GPU is updated too.
    static constexpr std::array<u8, 8> off{};
    AccelerateInlineToMemory(address, off.size(), off);
}

void RasterizerVulkan::ScaleBloomIntensity(const Tegra::Nx2BloomScale& bloom) {
    // The composite pass multiplies its blurred bloom by one float of its fragment constant
    // buffer. Scale that word before the draw uploads the buffer; a word that already holds our
    // scaled value is left alone, so the factor does not compound when the game keeps the buffer.
    const float scale = VideoCommon::BloomScaleOverride().value_or(bloom.default_scale);
    if (scale == 1.0f) {
        return; // the game's own look: nothing to rewrite (was an inline upload per frame)
    }
    const auto& cbuf = maxwell3d->state.shader_stages[4].const_buffers[bloom.bank];
    const GPUVAddr address = cbuf.address + static_cast<GPUVAddr>(bloom.word) * sizeof(u32);
    if (!cbuf.enabled || cbuf.size < (bloom.word + 1) * sizeof(u32) ||
        !gpu_memory->IsFullyMappedRange(address, sizeof(u32))) {
        return;
    }
    u32 raw{};
    gpu_memory->ReadBlockUnsafe(address, &raw, sizeof(raw));
    if (address == bloom_scaled_address && raw == bloom_scaled_word) {
        return;
    }
    float value{};
    std::memcpy(&value, &raw, sizeof(value));
    if (!(value > 0.0f) || !std::isfinite(value)) {
        return;
    }
    const float scaled = value * scale;
    std::array<u8, sizeof(float)> bytes{};
    std::memcpy(bytes.data(), &scaled, sizeof(scaled));
    std::memcpy(&bloom_scaled_word, &scaled, sizeof(scaled));
    bloom_scaled_address = address;
    static u32 logged{};
    if (logged < 8) {
        ++logged;
        LOG_INFO(Render_Vulkan, "Bloom intensity {} at {:#x} (bank {} word {}) scaled by {} to {}",
                 value, address, bloom.bank, bloom.word, scale, scaled);
    }
    // Written like an inline upload so a copy of the buffer already on the GPU is updated too.
    AccelerateInlineToMemory(address, bytes.size(), bytes);
}

bool RasterizerVulkan::DownscaleInsteadOfEasu(ComputePipeline& pipeline) {
    const auto textures = pipeline.BoundTextureViews();
    const auto storages = pipeline.BoundStorageViews();
    if (textures.empty() || storages.empty() || !textures[0] || !storages[0]) {
        return false;
    }
    const ImageView& src_view = texture_cache.GetImageView(textures[0]);
    const ImageView& dst_view = texture_cache.GetImageView(storages[0]);
    const VideoCommon::ImageId src_id = src_view.image_id;
    const VideoCommon::ImageId dst_id = dst_view.image_id;
    const auto src_format = src_view.format;
    const auto dst_format = dst_view.format;
    const auto src_size = src_view.size;
    const auto dst_size = dst_view.size;
    Image& src_image = texture_cache.GetImage(src_id);
    if (!src_image.IsRescaled()) {
        return false; // Nothing gained without a scaled source.
    }
    // The game's EASU reads its sRGB input without colour conversion and writes the same encoded
    // values to a UNORM target. A blit straight from the sRGB image into the UNORM one would
    // decode to linear on the way (darker, more saturated), so the averaging goes through an sRGB
    // intermediate (decode, average in linear light, re-encode) and a raw copy finishes it.
    // The game binds both through plain (UNORM) views; what matters for the blit is the images'
    // own formats (the FXAA output is an sRGB image), since vkCmdBlitImage converts by those.
    using VideoCore::Surface::PixelFormat;
    const auto is_rgba8 = [](PixelFormat format) {
        return format == PixelFormat::A8B8G8R8_SRGB || format == PixelFormat::A8B8G8R8_UNORM;
    };
    const PixelFormat src_image_format = src_image.info.format;
    const PixelFormat dst_image_format = texture_cache.GetImage(dst_id).info.format;
    if (!is_rgba8(src_image_format) || !is_rgba8(dst_image_format) || !is_rgba8(src_format) ||
        !is_rgba8(dst_format)) {
        static bool warned{};
        if (!warned) {
            warned = true;
            LOG_WARNING(Render_Vulkan, "Supersample: unexpected EASU formats {} -> {} (views {} -> {}), leaving the game's pass",
                        src_image_format, dst_image_format, src_format, dst_format);
        }
        return false;
    }
    const VkFormat mid_format = src_image_format == PixelFormat::A8B8G8R8_SRGB
                                    ? VK_FORMAT_A8B8G8R8_SRGB_PACK32
                                    : VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    // The output keeps the full resolution: the game's 1080p target is scaled up like the scene
    // (the pipeline configure scaled it down as a compute-written image), so the picture reaches
    // the window image, the UI pass and the present at 2x. Views of the image are recreated by
    // the scale, so only image references are used from here on.
    // With dynamic resolution the game renders only the top-left viewport of its 1600x900 target
    // and EASU upscales just that part (FsrEasuCon at the front of a constant buffer: con0.xy =
    // viewport / output size, con0.zw = 0.5 * con0.xy - 0.5, con1.xy = 1 / input image size).
    // Blitting the whole target put the picture in the top-left corner, so blit the viewport.
    u32 view_width = src_size.width;
    u32 view_height = src_size.height;
    {
        const auto& qmd{kepler_compute->launch_description};
        for (u32 bank = 0; bank < 8; ++bank) {
            if ((qmd.const_buffer_enable_mask.Value() & (1U << bank)) == 0) {
                continue;
            }
            const GPUVAddr address{qmd.const_buffer_config[bank].Address()};
            std::array<float, 8> con{};
            if (!gpu_memory->IsFullyMappedRange(address, sizeof(con))) {
                continue;
            }
            gpu_memory->ReadBlockUnsafe(address, con.data(), sizeof(con));
            const bool easu = con[0] > 0.0f && con[1] > 0.0f && con[4] > 0.0f &&
                              std::abs(con[2] - (0.5f * con[0] - 0.5f)) < 1e-3f &&
                              std::abs(con[3] - (0.5f * con[1] - 0.5f)) < 1e-3f &&
                              std::abs(1.0f / con[4] - static_cast<float>(src_size.width)) < 2.0f;
            if (!easu) {
                continue;
            }
            const float width = con[0] * static_cast<float>(dst_size.width);
            const float height = con[1] * static_cast<float>(dst_size.height);
            if (width >= 16.0f && height >= 16.0f && width <= src_size.width + 0.5f &&
                height <= src_size.height + 0.5f) {
                view_width = static_cast<u32>(std::lround(width));
                view_height = static_cast<u32>(std::lround(height));
            }
            break;
        }
        static u32 last_view{};
        static u32 view_logs{};
        const u32 view = (view_width << 16) | view_height;
        if (view != last_view && view_logs < 32) {
            ++view_logs;
            last_view = view;
            LOG_INFO(Render_Vulkan, "Supersample: EASU input viewport {}x{} of {}x{}", view_width,
                     view_height, src_size.width, src_size.height);
        }
    }
    texture_cache.ForceScaleUp(dst_id);
    Image& dst_image = texture_cache.GetImage(dst_id);
    const auto& resolution = Settings::values.resolution_info;
    const s32 src_width = static_cast<s32>(resolution.ScaleUp(view_width));
    const s32 src_height = static_cast<s32>(resolution.ScaleUp(view_height));
    const s32 dst_width = static_cast<s32>(dst_image.IsRescaled() ? resolution.ScaleUp(dst_size.width) : dst_size.width);
    const s32 dst_height = static_cast<s32>(dst_image.IsRescaled() ? resolution.ScaleUp(dst_size.height) : dst_size.height);
    if (!ssaa_image || ssaa_extent.width != static_cast<u32>(dst_width) ||
        ssaa_extent.height != static_cast<u32>(dst_height) || ssaa_format != mid_format) {
        scheduler.Finish();
        ssaa_format = mid_format;
        const VkImageCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = mid_format,
            .extent = {.width = static_cast<u32>(dst_width), .height = static_cast<u32>(dst_height), .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        ssaa_image = memory_allocator.CreateImage(ci);
        ssaa_extent = {static_cast<u32>(dst_width), static_cast<u32>(dst_height)};
        ssaa_image_initialized = false;
    }
    static u32 logged{};
    if (logged < 4) {
        ++logged;
        LOG_INFO(Render_Vulkan, "Supersample: EASU dispatch replaced by a {}x{} -> {}x{} averaging blit",
                 src_width, src_height, dst_width, dst_height);
    }
    const VkImage src = src_image.Handle();
    const VkImage dst = dst_image.Handle();
    const VkImage mid = *ssaa_image;
    const bool init_mid = !ssaa_image_initialized;
    ssaa_image_initialized = true;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([src, dst, mid, init_mid, src_width, src_height, dst_width,
                      dst_height](vk::CommandBuffer cmdbuf) {
        const auto barrier = [](VkImage image, VkAccessFlags from, VkAccessFlags to,
                                VkImageLayout old_layout = VK_IMAGE_LAYOUT_GENERAL) {
            return VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = from,
                .dstAccessMask = to,
                .oldLayout = old_layout,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
        };
        const VkImageSubresourceLayers layers{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1};
        const std::array before{
            barrier(src, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            barrier(mid, init_mid ? 0 : VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    init_mid ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL),
            barrier(dst, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, {}, {}, before);
        const VkImageBlit blit{
            .srcSubresource = layers,
            .srcOffsets{{0, 0, 0}, {src_width, src_height, 1}},
            .dstSubresource = layers,
            .dstOffsets{{0, 0, 0}, {dst_width, dst_height, 1}},
        };
        cmdbuf.BlitImage(src, VK_IMAGE_LAYOUT_GENERAL, mid, VK_IMAGE_LAYOUT_GENERAL, blit,
                         VK_FILTER_LINEAR);
        const VkImageMemoryBarrier mid_ready =
            barrier(mid, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               mid_ready);
        const VkImageCopy copy{
            .srcSubresource = layers,
            .srcOffset{0, 0, 0},
            .dstSubresource = layers,
            .dstOffset{0, 0, 0},
            .extent{static_cast<u32>(dst_width), static_cast<u32>(dst_height), 1},
        };
        cmdbuf.CopyImage(mid, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, copy);
        const VkImageMemoryBarrier after =
            barrier(dst, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, after);
    });
    return true;
}

bool RasterizerVulkan::CopyInsteadOfRcas(ComputePipeline& pipeline) {
    const auto textures = pipeline.BoundTextureViews();
    const auto storages = pipeline.BoundStorageViews();
    if (textures.empty() || storages.empty() || !textures[0] || !storages[0]) {
        return false;
    }
    const ImageView& src_view = texture_cache.GetImageView(textures[0]);
    const ImageView& dst_view = texture_cache.GetImageView(storages[0]);
    const VideoCommon::ImageId src_id = src_view.image_id;
    const VideoCommon::ImageId dst_id = dst_view.image_id;
    const auto src_size = src_view.size;
    const auto dst_size = dst_view.size;
    if (src_view.format != dst_view.format || src_size.width != dst_size.width ||
        src_size.height != dst_size.height) {
        return false;
    }
    Image& src_image = texture_cache.GetImage(src_id);
    if (!src_image.IsRescaled()) {
        return false; // The EASU replacement did not run; leave the game's pass alone.
    }
    texture_cache.ForceScaleUp(dst_id);
    Image& dst_image = texture_cache.GetImage(dst_id);
    if (!dst_image.IsRescaled()) {
        return false;
    }
    const auto& resolution = Settings::values.resolution_info;
    const u32 width = resolution.ScaleUp(dst_size.width);
    const u32 height = resolution.ScaleUp(dst_size.height);
    static u32 logged{};
    if (logged < 2) {
        ++logged;
        LOG_INFO(Render_Vulkan, "Supersample: RCAS dispatch replaced by a {}x{} copy", width, height);
    }
    const VkImage src = src_image.Handle();
    const VkImage dst = dst_image.Handle();
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([src, dst, width, height](vk::CommandBuffer cmdbuf) {
        const auto barrier = [](VkImage image, VkAccessFlags from, VkAccessFlags to) {
            return VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = from,
                .dstAccessMask = to,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
        };
        const std::array before{
            barrier(src, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            barrier(dst, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, {}, {}, before);
        const VkImageSubresourceLayers layers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        const VkImageCopy copy{
            .srcSubresource = layers,
            .srcOffset{0, 0, 0},
            .dstSubresource = layers,
            .dstOffset{0, 0, 0},
            .extent{width, height, 1},
        };
        cmdbuf.CopyImage(src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, copy);
        const VkImageMemoryBarrier after =
            barrier(dst, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, after);
    });
    return true;
}

void RasterizerVulkan::SetViewportJitter(std::array<float, 2> jitter) {
    if (jitter != viewport_jitter) {
        viewport_jitter = jitter;
        state_tracker.InvalidateViewports();
    }
}

#ifdef HAS_DLSS
bool RasterizerVulkan::DlssBeforeDraw() {
    const std::optional hooks = Tegra::Nx2Dlss();
    if (!hooks || !dlss->IsAvailable()) {
        return false;
    }
    const auto& regs = maxwell3d->regs;
    const auto& hashes = pipeline_cache.CurrentGraphicsShaderHashes();
    // The scene camera, from the first draw of the frame that uses the camera vertex shader.
    if (!dlss_camera_captured && hashes[1] == hooks->camera_vertex) {
        const auto& cbuf = maxwell3d->state.shader_stages[0].const_buffers[hooks->camera_bank];
        constexpr u32 camera_size = sizeof(dlss_frame.camera);
        if (cbuf.enabled && cbuf.size >= camera_size &&
            gpu_memory->IsFullyMappedRange(cbuf.address, camera_size)) {
            gpu_memory->ReadBlockUnsafe(cbuf.address, dlss_frame.camera.data(), camera_size);
            const auto& viewport = regs.viewport_transform[0];
            dlss_frame.viewport = {viewport.scale_x,     viewport.scale_y,
                                   viewport.scale_z,     viewport.translate_x,
                                   viewport.translate_y, viewport.translate_z};
            dlss_frame.has_camera = true;
            dlss_camera_captured = true;
        }
    }
    if (hashes[5] == hooks->final_pass_fragment) {
        const bool replaced = RunDlss();
        dlss_camera_captured = false;
        dlss_frame.has_camera = false;
        SetViewportJitter({});
        return replaced;
    }
    // Only the scene is jittered: draws into the scene depth buffer.
    const bool scene = regs.zeta_enable != 0 && dlss_scene_depth != 0 &&
                       regs.zeta.Address() == dlss_scene_depth;
    SetViewportJitter(scene ? dlss->Jitter() : std::array<float, 2>{});
    return false;
}

void RasterizerVulkan::DlssAfterConfigure() {
    if (!Tegra::Nx2Dlss() || !dlss->IsAvailable()) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    const auto& targets = texture_cache.CurrentRenderTargets();
    // The G-buffer pass (four or more colour targets with depth) names the scene depth buffer.
    if (regs.rt_control.count >= 4 && regs.zeta_enable != 0) {
        dlss_scene_depth = regs.zeta.Address();
        dlss_scene_depth_view = targets.depth_buffer_id;
    }
    dlss_last_color_view = targets.color_buffer_ids[0];
}

bool RasterizerVulkan::RunDlss() {
    if (!dlss_frame.has_camera || !dlss_scene_depth_view || !dlss_last_color_view) {
        return false;
    }
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(false);
    const VideoCommon::ImageViewId output_id =
        texture_cache.CurrentRenderTargets().color_buffer_ids[0];
    if (!output_id || output_id == dlss_last_color_view) {
        return false;
    }
    // The final pass reads the colour target of the draw before it, the finished frame.
    ImageView& color = texture_cache.GetImageView(dlss_last_color_view);
    ImageView& depth = texture_cache.GetImageView(dlss_scene_depth_view);
    ImageView& output = texture_cache.GetImageView(output_id);
    const auto vk_format = [this](VideoCore::Surface::PixelFormat format) {
        return MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, format).format;
    };
    dlss_frame.color_image = color.ImageHandle();
    dlss_frame.color_view = color.Handle(Shader::TextureType::Color2D);
    dlss_frame.color_format = vk_format(color.format);
    dlss_frame.color_width = color.size.width;
    dlss_frame.color_height = color.size.height;
    dlss_frame.depth_image = depth.ImageHandle();
    dlss_frame.depth_view = depth.DepthView();
    dlss_frame.depth_format = vk_format(depth.format);
    dlss_frame.depth_width = depth.size.width;
    dlss_frame.depth_height = depth.size.height;
    dlss_frame.output_image = output.ImageHandle();
    dlss_frame.output_width = output.size.width;
    dlss_frame.output_height = output.size.height;
    return dlss->Evaluate(dlss_frame);
}
#endif

void RasterizerVulkan::FlushWork() {
#ifdef __ANDROID__
    static constexpr u32 DRAWS_TO_DISPATCH = 512;
    static constexpr u32 CHECK_MASK = 3;
#else
    static constexpr u32 DRAWS_TO_DISPATCH = 4096;
    static constexpr u32 CHECK_MASK = 7;
#endif // __ANDROID__

    static_assert(DRAWS_TO_DISPATCH % (CHECK_MASK + 1) == 0);
    if ((++draw_counter & CHECK_MASK) != CHECK_MASK) {
        return;
    }
    if (draw_counter < DRAWS_TO_DISPATCH) {
        scheduler.DispatchWork();
        return;
    }
    scheduler.Flush();
    draw_counter = 0;
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

void RasterizerVulkan::UpdateDynamicStates() {
    auto& regs = maxwell3d->regs;
    auto& flags = maxwell3d->dirty.flags;
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const bool topology_changed = state_tracker.ChangePrimitiveTopology(topology);
    if (topology_changed) {
        flags[Dirty::DepthBiasEnable] = true;
        flags[Dirty::PrimitiveRestartEnable] = true;
    }

    UpdateViewportsState(regs);
    UpdateScissorsState(regs);
    UpdateDepthBias(regs);
    UpdateBlendConstants(regs);
    UpdateDepthBounds(regs);
    UpdateStencilFaces(regs);
    UpdateLineWidth(regs);

    if (device.IsExtExtendedDynamicStateSupported()) {
        UpdateCullMode(regs);
        UpdateDepthCompareOp(regs);
        UpdateFrontFace(regs);
        UpdateStencilOp(regs);
        if (state_tracker.TouchStateEnable()) {
            UpdateDepthBoundsTestEnable(regs);
            UpdateDepthTestEnable(regs);
            UpdateDepthWriteEnable(regs);
            UpdateStencilTestEnable(regs);
        }
        if (topology_changed) {
            scheduler.Record([topology_vk = MaxwellToVK::PrimitiveTopology(device, topology)](
                                 vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveTopologyEXT(topology_vk);
            });
        }
    }

    if (device.IsExtExtendedDynamicState2Supported()) {
        UpdatePrimitiveRestartEnable(regs);
        UpdateRasterizerDiscardEnable(regs);
        UpdateDepthBiasEnable(regs);
    }

    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        UpdateLogicOp(regs);
    }

    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        using namespace Tegra::Engines;
        // AMD Workaround: LogicOp incompatible with float render targets
        if (device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_OPEN_SOURCE ||
            device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_PROPRIETARY) {
            const auto has_float = std::any_of(
                regs.vertex_attrib_format.begin(), regs.vertex_attrib_format.end(),
                [](const auto& attrib) {
                    return attrib.type == Maxwell3D::Regs::VertexAttribute::Type::Float;
                }
            );
            if (regs.logic_op.enable) {
                regs.logic_op.enable = static_cast<u32>(!has_float);
            }
        }
        UpdateLogicOpEnable(regs);
        UpdateDepthClampEnable(regs);
        UpdateLineRasterizationMode(regs);
        UpdateLineStippleEnable(regs);
        UpdateConservativeRasterizationMode(regs);
        UpdateAlphaToCoverageEnable(regs);
        UpdateAlphaToOneEnable(regs);
    }

    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        UpdateBlending(regs);
    } else if (device.IsExtColorWriteEnableSupported()) {
        UpdateColorWriteEnable(regs);
    }

    if (device.IsExtVertexInputDynamicStateSupported()) {
        if (auto* gp = pipeline_cache.CurrentGraphicsPipeline(); gp && gp->HasDynamicVertexInput()) {
            UpdateVertexInput(regs);
        }
    }
}

void RasterizerVulkan::HandleTransformFeedback() {
    static std::once_flag warn_unsupported;

    const auto& regs = maxwell3d->regs;
    if (!device.IsExtTransformFeedbackSupported()) {
        if (regs.transform_feedback_enabled != 0) {
            std::call_once(warn_unsupported, [&] {
                LOG_WARNING(Render_Vulkan, "Transform feedback requested by guest but VK_EXT_transform_feedback is unavailable; queries disabled");
            });
        } else {
            std::call_once(warn_unsupported, [&] {
                LOG_INFO(Render_Vulkan, "VK_EXT_transform_feedback not available on device");
            });
        }
        return;
    }
    query_cache.CounterEnable(VideoCommon::QueryType::StreamingByteCount,
                              regs.transform_feedback_enabled);
    if (regs.transform_feedback_enabled != 0) {
        // Log extension usage for transform feedback
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
                "VK_EXT_transform_feedback", "HandleTransformFeedback");
        }
        UNIMPLEMENTED_IF(regs.IsShaderConfigEnabled(Maxwell::ShaderType::TessellationInit) ||
                         regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation));
    }
}

void RasterizerVulkan::UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchViewports()) {
        return;
    }

    maxwell3d->dirty.flags[Dirty::Scissors] = true;

    if (!regs.viewport_scale_offset_enabled) {
        float x = static_cast<float>(regs.surface_clip.x);
        float y = static_cast<float>(regs.surface_clip.y);
        float width = (std::max)(1.0f, static_cast<float>(regs.surface_clip.width));
        float height = (std::max)(1.0f, static_cast<float>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y += height;
            height = -height;
        }
        VkViewport viewport{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        scheduler.Record([this, viewport](vk::CommandBuffer cmdbuf) {
            const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkViewport, Maxwell::NumViewports> viewport_list{};
            viewport_list.fill(viewport);
            const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
            cmdbuf.SetViewport(0, viewports);
        });
        return;
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const float scale = is_rescaling ? Settings::values.resolution_info.up_factor : 1.0f;
    const auto jitter = viewport_jitter;
    const std::array viewport_list{
        GetViewportState(device, regs, 0, scale, jitter),  GetViewportState(device, regs, 1, scale, jitter),
        GetViewportState(device, regs, 2, scale, jitter),  GetViewportState(device, regs, 3, scale, jitter),
        GetViewportState(device, regs, 4, scale, jitter),  GetViewportState(device, regs, 5, scale, jitter),
        GetViewportState(device, regs, 6, scale, jitter),  GetViewportState(device, regs, 7, scale, jitter),
        GetViewportState(device, regs, 8, scale, jitter),  GetViewportState(device, regs, 9, scale, jitter),
        GetViewportState(device, regs, 10, scale, jitter), GetViewportState(device, regs, 11, scale, jitter),
        GetViewportState(device, regs, 12, scale, jitter), GetViewportState(device, regs, 13, scale, jitter),
        GetViewportState(device, regs, 14, scale, jitter), GetViewportState(device, regs, 15, scale, jitter),
    };
    scheduler.Record([this, viewport_list](vk::CommandBuffer cmdbuf) {
        const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
        cmdbuf.SetViewport(0, viewports);
    });
}

void RasterizerVulkan::UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchScissors()) {
        return;
    }
    if (!regs.viewport_scale_offset_enabled) {
        u32 x = regs.surface_clip.x;
        u32 y = regs.surface_clip.y;
        u32 width = (std::max)(1u, static_cast<u32>(regs.surface_clip.width));
        u32 height = (std::max)(1u, static_cast<u32>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y = regs.surface_clip.height - (y + height);
        }
        VkRect2D scissor{};
        scissor.offset.x = static_cast<int32_t>(x);
        scissor.offset.y = static_cast<int32_t>(y);
        scissor.extent.width  = width;
        scissor.extent.height = height;
        scheduler.Record([this, scissor](vk::CommandBuffer cmdbuf) {
            const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkRect2D, Maxwell::NumViewports> scissor_list{};
            scissor_list.fill(scissor);
            const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
            cmdbuf.SetScissor(0, scissors);
        });
        return;
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    const std::array scissor_list{
        GetScissorState(regs, 0, up_scale, down_shift),
        GetScissorState(regs, 1, up_scale, down_shift),
        GetScissorState(regs, 2, up_scale, down_shift),
        GetScissorState(regs, 3, up_scale, down_shift),
        GetScissorState(regs, 4, up_scale, down_shift),
        GetScissorState(regs, 5, up_scale, down_shift),
        GetScissorState(regs, 6, up_scale, down_shift),
        GetScissorState(regs, 7, up_scale, down_shift),
        GetScissorState(regs, 8, up_scale, down_shift),
        GetScissorState(regs, 9, up_scale, down_shift),
        GetScissorState(regs, 10, up_scale, down_shift),
        GetScissorState(regs, 11, up_scale, down_shift),
        GetScissorState(regs, 12, up_scale, down_shift),
        GetScissorState(regs, 13, up_scale, down_shift),
        GetScissorState(regs, 14, up_scale, down_shift),
        GetScissorState(regs, 15, up_scale, down_shift),
    };
    scheduler.Record([this, scissor_list](vk::CommandBuffer cmdbuf) {
        const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
        cmdbuf.SetScissor(0, scissors);
    });
}

void RasterizerVulkan::UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBias()) {
        return;
    }
    float units = regs.depth_bias / 2.0f;
    const bool is_d24 = regs.zeta.format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
                        regs.zeta.format == Tegra::DepthFormat::X8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::S8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::V8Z24_UNORM;

    if (is_d24 && !device.SupportsD24DepthBuffer()) {
        static constexpr const size_t length = sizeof(NEEDS_D24) / sizeof(NEEDS_D24[0]);

        static constexpr const u64* start = NEEDS_D24;
        static constexpr const u64* end = NEEDS_D24 + length;

        const u64* it = std::find(start, end, program_id);

        if (it != end) {
            // the base formulas can be obtained from here:
            //   https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
            const double rescale_factor =
                static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
            units = static_cast<float>(static_cast<double>(units) * rescale_factor);
        }
    }

    scheduler.Record([constant = units, clamp = regs.depth_bias_clamp,
                      factor = regs.slope_scale_depth_bias, this](vk::CommandBuffer cmdbuf) {
        if (device.IsExtDepthBiasControlSupported()) {
            static VkDepthBiasRepresentationInfoEXT bias_info{
                .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                .pNext = nullptr,
                .depthBiasRepresentation =
                    VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                .depthBiasExact = VK_FALSE,
            };

            cmdbuf.SetDepthBias(constant, clamp, factor, &bias_info);
        } else {
            cmdbuf.SetDepthBias(constant, clamp, factor);
        }
    });
}

void RasterizerVulkan::UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlendConstants()) {
        return;
    }
    const std::array blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                    regs.blend_color.a};
    scheduler.Record(
        [blend_color](vk::CommandBuffer cmdbuf) { cmdbuf.SetBlendConstants(blend_color.data()); });
}

void RasterizerVulkan::UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBounds()) {
        return;
    }
    scheduler.Record([min = regs.depth_bounds[0], max = regs.depth_bounds[1]](
                         vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBounds(min, max); });
}

void RasterizerVulkan::UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilProperties()) {
        return;
    }
    bool update_references = state_tracker.TouchStencilReference();
    bool update_write_mask = state_tracker.TouchStencilWriteMask();
    bool update_compare_masks = state_tracker.TouchStencilCompare();
    if (state_tracker.TouchStencilSide(regs.stencil_two_side_enable != 0)) {
        update_references = true;
        update_write_mask = true;
        update_compare_masks = true;
    }
    if (update_references) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_back_ref);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_front_ref);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_ref = regs.stencil_front_ref, back_ref = regs.stencil_back_ref,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_ref != back_ref;
                // Front face
                cmdbuf.SetStencilReference(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_ref);
                if (set_back) {
                    cmdbuf.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, back_ref);
                }
            });
        }();
    }
    if (update_write_mask) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_back_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_front_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_write_mask = regs.stencil_front_mask,
                              back_write_mask = regs.stencil_back_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_write_mask != back_write_mask;
                // Front face
                cmdbuf.SetStencilWriteMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_write_mask);
                if (set_back) {
                    cmdbuf.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, back_write_mask);
                }
            });
        }();
    }
    if (update_compare_masks) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_back_func_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_front_func_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_test_mask = regs.stencil_front_func_mask,
                              back_test_mask = regs.stencil_back_func_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_test_mask != back_test_mask;
                // Front face
                cmdbuf.SetStencilCompareMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                      : VK_STENCIL_FACE_FRONT_AND_BACK,
                                             front_test_mask);
                if (set_back) {
                    cmdbuf.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, back_test_mask);
                }
            });
        }();
    }
    state_tracker.ClearStencilReset();
}

void RasterizerVulkan::UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineWidth()) {
        return;
    }
    const float width =
        regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
    scheduler.Record([width](vk::CommandBuffer cmdbuf) { cmdbuf.SetLineWidth(width); });
}

void RasterizerVulkan::UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchCullMode()) {
        return;
    }
    scheduler.Record([enabled = regs.gl_cull_test_enabled,
                      cull_face = regs.gl_cull_face](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetCullModeEXT(enabled ? MaxwellToVK::CullFace(cull_face) : VK_CULL_MODE_NONE);
    });
}

void RasterizerVulkan::UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBoundsTestEnable()) {
        return;
    }
    bool enabled = regs.depth_bounds_enable;
    if (enabled && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
        enabled = false;
    }
    scheduler.Record([enable = enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthBoundsTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_test_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthWriteEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_write_enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthWriteEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchPrimitiveRestartEnable()) {
        return;
    }

    bool enable = regs.primitive_restart.enabled != 0;
    if (device.IsMoltenVK()) {
        enable = true;
    } else if (enable) {
        const auto topology = MaxwellToVK::PrimitiveTopology(device, maxwell3d->draw_manager.draw_state.topology);
        enable = IsPrimitiveRestartSupported(device, topology);
    }

    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetPrimitiveRestartEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchRasterizerDiscardEnable()) {
        return;
    }
    scheduler.Record([disable = regs.rasterize_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetRasterizerDiscardEnableEXT(disable == 0);
    });
}

void RasterizerVulkan::UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchConservativeRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3ConservativeRasterizationMode()) {
        return;
    }

    scheduler.Record([enable = regs.conservative_raster_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetConservativeRasterizationModeEXT(
            enable ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                   : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    });
}

void RasterizerVulkan::UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineStippleEnable()) {
        return;
    }

    if (!device.SupportsDynamicState3LineStippleEnable()) {
        return;
    }

    scheduler.Record([enable = regs.line_stipple_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineStippleEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!device.IsExtLineRasterizationSupported()) {
        return;
    }
    if (!state_tracker.TouchLineRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3LineRasterizationMode()) {
        static std::once_flag warn_missing_rect;
        std::call_once(warn_missing_rect, [] {
            LOG_WARNING(Render_Vulkan,
                        "Driver lacks rectangular line rasterization support; skipping dynamic "
                        "line state updates");
        });
        return;
    }

    const bool wants_smooth = regs.line_anti_alias_enable != 0;
    VkLineRasterizationModeEXT mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT;
    if (wants_smooth) {
        if (device.SupportsSmoothLines()) {
            mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT;
        } else {
            static std::once_flag warn_missing_smooth;
            std::call_once(warn_missing_smooth, [] {
                LOG_WARNING(Render_Vulkan,
                            "Line anti-aliasing requested but smoothLines feature unavailable; "
                            "using rectangular rasterization");
            });
        }
    }
    scheduler.Record([mode](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineRasterizationModeEXT(mode);
    });
}

void RasterizerVulkan::UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBiasEnable()) {
        return;
    }
    constexpr size_t POINT = 0;
    constexpr size_t LINE = 1;
    constexpr size_t POLYGON = 2;
    static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        POINT,   // Points
        LINE,    // Lines
        LINE,    // LineLoop
        LINE,    // LineStrip
        POLYGON, // Triangles
        POLYGON, // TriangleStrip
        POLYGON, // TriangleFan
        POLYGON, // Quads
        POLYGON, // QuadStrip
        POLYGON, // Polygon
        LINE,    // LinesAdjacency
        LINE,    // LineStripAdjacency
        POLYGON, // TrianglesAdjacency
        POLYGON, // TriangleStripAdjacency
        POLYGON, // Patches
    };
    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    const u32 topology_index = u32(maxwell3d->draw_manager.draw_state.topology);
    const u32 enable = enabled_lut[POLYGON_OFFSET_ENABLE_LUT[topology_index]];
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBiasEnableEXT(enable != 0); });
}

void RasterizerVulkan::UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOpEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3LogicOpEnable()) {
        return;
    }
    scheduler.Record([enable = regs.logic_op.enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLogicOpEnableEXT(enable != 0);
    });
}

void RasterizerVulkan::UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthClampEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3DepthClampEnable()) {
        return;
    }
    bool is_enabled = !(regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::Passthrough ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumZ);
    scheduler.Record(
        [is_enabled](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthClampEnableEXT(is_enabled); });
}

void RasterizerVulkan::UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToCoverageEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToCoverageEnable()) {
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToCoverage() &&
                        regs.anti_alias_alpha_control.alpha_to_coverage != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToCoverageEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToOneEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToOneEnable()) {
        static std::once_flag warn_alpha_to_one;
        std::call_once(warn_alpha_to_one, [] {
            LOG_WARNING(Render_Vulkan,
                        "Alpha-to-one is not supported on this device; forcing it disabled");
        });
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToOne() &&
                        regs.anti_alias_alpha_control.alpha_to_one != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToOneEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthCompareOp()) {
        return;
    }
    scheduler.Record([func = regs.depth_test_func](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthCompareOpEXT(MaxwellToVK::ComparisonOp(func));
    });
}

void RasterizerVulkan::UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchFrontFace()) {
        return;
    }

    VkFrontFace front_face = MaxwellToVK::FrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        front_face = front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                           : VK_FRONT_FACE_CLOCKWISE;
    }
    scheduler.Record(
        [front_face](vk::CommandBuffer cmdbuf) { cmdbuf.SetFrontFaceEXT(front_face); });
}

void RasterizerVulkan::UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilOp()) {
        return;
    }
    const Maxwell::StencilOp::Op fail = regs.stencil_front_op.fail;
    const Maxwell::StencilOp::Op zfail = regs.stencil_front_op.zfail;
    const Maxwell::StencilOp::Op zpass = regs.stencil_front_op.zpass;
    const Maxwell::ComparisonOp compare = regs.stencil_front_op.func;
    if (regs.stencil_two_side_enable) {
        // Separate stencil op per face
        const Maxwell::StencilOp::Op back_fail = regs.stencil_back_op.fail;
        const Maxwell::StencilOp::Op back_zfail = regs.stencil_back_op.zfail;
        const Maxwell::StencilOp::Op back_zpass = regs.stencil_back_op.zpass;
        const Maxwell::ComparisonOp back_compare = regs.stencil_back_op.func;
        scheduler.Record([fail, zfail, zpass, compare, back_fail, back_zfail, back_zpass,
                          back_compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, MaxwellToVK::StencilOp(back_fail),
                                   MaxwellToVK::StencilOp(back_zpass),
                                   MaxwellToVK::StencilOp(back_zfail),
                                   MaxwellToVK::ComparisonOp(back_compare));
        });
    } else {
        // Front face defines the stencil op of both faces
        scheduler.Record([fail, zfail, zpass, compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_AND_BACK, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
        });
    }
}

void RasterizerVulkan::UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOp()) {
        return;
    }
    const auto op_value = static_cast<u32>(regs.logic_op.op);
    auto op = op_value >= 0x1500 && op_value < 0x1510 ? static_cast<VkLogicOp>(op_value - 0x1500)
                                                      : VK_LOGIC_OP_NO_OP;
    scheduler.Record([op](vk::CommandBuffer cmdbuf) { cmdbuf.SetLogicOpEXT(op); });
}

void RasterizerVulkan::UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlending()) {
        return;
    }

    if (state_tracker.TouchColorMask()) {
        std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> setup_masks{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            auto& current = setup_masks[index];
            if (mask.R) {
                current |= VK_COLOR_COMPONENT_R_BIT;
            }
            if (mask.G) {
                current |= VK_COLOR_COMPONENT_G_BIT;
            }
            if (mask.B) {
                current |= VK_COLOR_COMPONENT_B_BIT;
            }
            if (mask.A) {
                current |= VK_COLOR_COMPONENT_A_BIT;
            }
        }
        scheduler.Record([setup_masks](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorWriteMaskEXT(0, setup_masks);
        });
    }

    if (state_tracker.TouchBlendEnable()) {
        std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            bool is_integer = false;
            if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format);
                is_integer = IsPixelFormatInteger(format);
            }
            setup_enables[index] =
                (!is_integer && regs.blend.enable[index] != 0) ? VK_TRUE : VK_FALSE;
        }
        scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEnableEXT(0, setup_enables);
        });
    }

    if (state_tracker.TouchBlendEquations()) {
        std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> setup_blends{};

        const auto blend_setup = [&](auto& host_blend, const auto& guest_blend) {
            host_blend.srcColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_source);
            host_blend.dstColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_dest);
            host_blend.colorBlendOp = MaxwellToVK::BlendEquation(guest_blend.color_op);
            host_blend.srcAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_source);
            host_blend.dstAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_dest);
            host_blend.alphaBlendOp = MaxwellToVK::BlendEquation(guest_blend.alpha_op);
        };

        // Single blend equation for all targets
        if (!regs.blend_per_target_enabled) {
            // Temporary workaround for games that use iterated blending
            if (regs.iterated_blend.enable && Settings::values.use_squashed_iterated_blend) {
                setup_blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].colorBlendOp = VK_BLEND_OP_ADD;
                setup_blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                setup_blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                setup_blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
            } else {
                blend_setup(setup_blends[0], regs.blend);
            }

            // Copy first blend state to all other targets
            for (size_t index = 1; index < Maxwell::NumRenderTargets; index++) {
                setup_blends[index] = setup_blends[0];
            }
        } else {
            // Per-target blending
            for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
                blend_setup(setup_blends[index], regs.blend_per_target[index]);
            }
        }

        scheduler.Record([setup_blends](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEquationEXT(0, setup_blends);
        });
    }
}

void RasterizerVulkan::UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchColorMask()) {
        return;
    }
    std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        setup_enables[index] = (mask.R || mask.G || mask.B || mask.A) ? VK_TRUE : VK_FALSE;
    }
    scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetColorWriteEnableEXT(setup_enables);
    });
}

void RasterizerVulkan::UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.stencil_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetStencilTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs) {
    auto& dirty{maxwell3d->dirty.flags};
    const bool vertex_input_dirty = dirty[Dirty::VertexInput];
    const bool vertex_buffers_dirty = dirty[VideoCommon::Dirty::VertexBuffers];
    if (!vertex_input_dirty && !vertex_buffers_dirty) {
        return;
    }
    dirty[Dirty::VertexInput] = false;

    boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> bindings;
    boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> attributes;

    const u32 max_attributes =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexAttributes,
                                          device.GetMaxVertexInputAttributes()));
    const u32 max_bindings =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexArrays,
                                          device.GetMaxVertexInputBindings()));


    for (u32 index = 0; index < max_attributes; ++index) {
        const Maxwell::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        const u32 binding{attribute.buffer};
        if (attribute.constant || binding >= max_bindings) {
            continue;
        }
        attributes.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .location = index,
            .binding = binding,
            .format = MaxwellToVK::VertexFormat(device, attribute.type, attribute.size),
            .offset = attribute.offset,
        });
    }

    for (u32 binding = 0; binding < max_bindings; ++binding) {
        const auto& input_binding{regs.vertex_streams[binding]};
        const bool is_instanced{regs.vertex_stream_instances.IsInstancingEnabled(binding)};
        bindings.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .binding = binding,
            .stride = input_binding.stride,
            .inputRate = is_instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = is_instanced ? input_binding.frequency : 1,
        });
    }

    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        dirty[Dirty::VertexAttribute0 + index] = false;
    }
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        dirty[Dirty::VertexBinding0 + index] = false;
    }

    scheduler.Record([bindings, attributes](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetVertexInputEXT(bindings, attributes);
    });
}

void RasterizerVulkan::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    query_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerVulkan::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    query_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
}

void RasterizerVulkan::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
    query_cache.EraseChannel(channel_id);
}

} // namespace Vulkan
