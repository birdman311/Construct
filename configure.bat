@echo off
rem Construct: set up the build folder (run once, then build.bat). Builds without NVIDIA DLSS,
rem whose SDK license does not allow it inside GPL software.
cd /d "%~dp0"
set "VULKAN_SDK=C:\VulkanSDK\1.4.341.1"
set "PATH=C:\Qt\6.9.3\msvc2022_64\bin;C:\VulkanSDK\1.4.341.1\Bin;C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
call "C:\New folder\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_QT=ON -DYUZU_TESTS=OFF -DENABLE_WERROR=OFF -DYUZU_USE_BUNDLED_QT=OFF -DCMAKE_PREFIX_PATH=C:\Qt\6.9.3\msvc2022_64 -DENABLE_DLSS=OFF -DENABLE_RESHADE=ON -DENABLE_UPDATE_CHECKER=OFF
