// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

#include <cstdio>

#include "common/stb.h"

#include "video_core/frame_dump.h"

namespace VideoCommon {
namespace {
using VideoCore::Surface::PixelFormat;

float HalfToFloat(u16 h) {
    const u32 sign = (h >> 15) & 1, exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa ? 0.0f : 65504.0f;
    } else {
        value = std::ldexp(static_cast<float>(mantissa | 0x400), static_cast<int>(exponent) - 25);
    }
    return sign ? -value : value;
}

// Unsigned small floats of B10G11R11: 5-bit exponent, 6- or 5-bit mantissa.
float SmallFloat(u32 bits, u32 mantissa_bits) {
    const u32 exponent = bits >> mantissa_bits, mantissa = bits & ((1U << mantissa_bits) - 1);
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -14 - static_cast<int>(mantissa_bits));
    }
    if (exponent == 31) {
        return 65000.0f;
    }
    return std::ldexp(static_cast<float>(mantissa | (1U << mantissa_bits)),
                      static_cast<int>(exponent) - 15 - static_cast<int>(mantissa_bits));
}

u8 ToByte(float value) {
    return static_cast<u8>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// HDR color to display: Reinhard, then gamma 2.2.
u8 ToneMap(float value) {
    value = std::max(value, 0.0f);
    return ToByte(std::pow(value / (1.0f + value), 1.0f / 2.2f));
}

template <typename T>
T Load(const u8* p) {
    T value;
    std::memcpy(&value, p, sizeof(T));
    return value;
}
} // namespace

bool FrameDumpSupportsFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8B8G8R8_UNORM:
    case PixelFormat::A8B8G8R8_SRGB:
    case PixelFormat::B8G8R8A8_UNORM:
    case PixelFormat::B8G8R8A8_SRGB:
    case PixelFormat::A2B10G10R10_UNORM:
    case PixelFormat::R16G16B16A16_FLOAT:
    case PixelFormat::R16G16B16A16_UNORM:
    case PixelFormat::R32G32B32A32_FLOAT:
    case PixelFormat::B10G11R11_FLOAT:
    case PixelFormat::R16G16_FLOAT:
    case PixelFormat::R16G16_UNORM:
    case PixelFormat::R32G32_FLOAT:
    case PixelFormat::R32_FLOAT:
    case PixelFormat::R16_FLOAT:
    case PixelFormat::R16_UNORM:
    case PixelFormat::R8_UNORM:
    case PixelFormat::R8G8_UNORM:
        return true;
    default:
        return false;
    }
}

void WriteFrameDumpImage(const std::filesystem::path& path, PixelFormat format, u32 width,
                         u32 height, u32 row_length, std::span<const u8> data) {
    const u32 bpp = VideoCore::Surface::BytesPerBlock(format);
    if (data.size() < static_cast<size_t>(row_length) * (height - 1) * bpp + static_cast<size_t>(width) * bpp) {
        return;
    }
    std::vector<u8> rgb(static_cast<size_t>(width) * height * 3);
    // One-channel images: normalize to the image's own range.
    float lo = 0.0f, hi = 1.0f;
    const bool single = format == PixelFormat::R32_FLOAT || format == PixelFormat::R16_FLOAT;
    const auto scalar = [&](const u8* p) {
        return format == PixelFormat::R32_FLOAT ? Load<float>(p) : HalfToFloat(Load<u16>(p));
    };
    if (single) {
        lo = 1e30f;
        hi = -1e30f;
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const float v = scalar(data.data() + (static_cast<size_t>(y) * row_length + x) * bpp);
                if (std::isfinite(v)) {
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
            }
        }
        if (!(hi > lo)) {
            hi = lo + 1.0f;
        }
    }
    for (u32 y = 0; y < height; ++y) {
        for (u32 x = 0; x < width; ++x) {
            const u8* p = data.data() + (static_cast<size_t>(y) * row_length + x) * bpp;
            u8* out = rgb.data() + (static_cast<size_t>(y) * width + x) * 3;
            switch (format) {
            case PixelFormat::A8B8G8R8_UNORM:
            case PixelFormat::A8B8G8R8_SRGB:
                out[0] = p[0], out[1] = p[1], out[2] = p[2];
                break;
            case PixelFormat::B8G8R8A8_UNORM:
            case PixelFormat::B8G8R8A8_SRGB:
                out[0] = p[2], out[1] = p[1], out[2] = p[0];
                break;
            case PixelFormat::A2B10G10R10_UNORM: {
                const u32 v = Load<u32>(p);
                out[0] = static_cast<u8>((v & 0x3ff) >> 2);
                out[1] = static_cast<u8>(((v >> 10) & 0x3ff) >> 2);
                out[2] = static_cast<u8>(((v >> 20) & 0x3ff) >> 2);
                break;
            }
            case PixelFormat::R16G16B16A16_FLOAT:
                for (int c = 0; c < 3; ++c) out[c] = ToneMap(HalfToFloat(Load<u16>(p + c * 2)));
                break;
            case PixelFormat::R16G16B16A16_UNORM:
                for (int c = 0; c < 3; ++c) out[c] = static_cast<u8>(Load<u16>(p + c * 2) >> 8);
                break;
            case PixelFormat::R32G32B32A32_FLOAT:
                for (int c = 0; c < 3; ++c) out[c] = ToneMap(Load<float>(p + c * 4));
                break;
            case PixelFormat::B10G11R11_FLOAT: {
                const u32 v = Load<u32>(p);
                out[0] = ToneMap(SmallFloat(v & 0x7ff, 6));
                out[1] = ToneMap(SmallFloat((v >> 11) & 0x7ff, 6));
                out[2] = ToneMap(SmallFloat((v >> 22) & 0x3ff, 5));
                break;
            }
            case PixelFormat::R16G16_FLOAT:
                out[0] = ToByte(0.5f + 16.0f * HalfToFloat(Load<u16>(p)));
                out[1] = ToByte(0.5f + 16.0f * HalfToFloat(Load<u16>(p + 2)));
                out[2] = 128;
                break;
            case PixelFormat::R32G32_FLOAT:
                out[0] = ToByte(0.5f + 16.0f * Load<float>(p));
                out[1] = ToByte(0.5f + 16.0f * Load<float>(p + 4));
                out[2] = 128;
                break;
            case PixelFormat::R16G16_UNORM:
                out[0] = static_cast<u8>(Load<u16>(p) >> 8);
                out[1] = static_cast<u8>(Load<u16>(p + 2) >> 8);
                out[2] = 0;
                break;
            case PixelFormat::R32_FLOAT:
            case PixelFormat::R16_FLOAT:
                out[0] = out[1] = out[2] = ToByte((scalar(p) - lo) / (hi - lo));
                break;
            case PixelFormat::R16_UNORM:
                out[0] = out[1] = out[2] = static_cast<u8>(Load<u16>(p) >> 8);
                break;
            case PixelFormat::R8_UNORM:
                out[0] = out[1] = out[2] = p[0];
                break;
            case PixelFormat::R8G8_UNORM:
                out[0] = p[0], out[1] = p[1], out[2] = 0;
                break;
            default:
                return;
            }
        }
    }
    // Eden builds stb without stdio: write through a callback.
    std::FILE* file = std::fopen(path.string().c_str(), "wb");
    if (!file) {
        return;
    }
    stbi_write_png_to_func(
        [](void* context, void* bytes, int size) {
            std::fwrite(bytes, 1, static_cast<size_t>(size), static_cast<std::FILE*>(context));
        },
        file, static_cast<int>(width), static_cast<int>(height), 3, rgb.data(),
        static_cast<int>(width) * 3);
    std::fclose(file);
}

} // namespace VideoCommon
