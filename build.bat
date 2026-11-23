@echo off
rem Construct: build construct.exe into build\bin (no DLSS). Run configure.bat first. 4 jobs.
cd /d "%~dp0"
set "VULKAN_SDK=C:\VulkanSDK\1.4.341.1"
set "PATH=C:\Qt\6.9.3\msvc2022_64\bin;C:\VulkanSDK\1.4.341.1\Bin;C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
call "C:\New folder\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --target yuzu -j 4
