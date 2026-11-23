# Construct
Construct is a Nintendo Switch and Switch 2 emulator for Windows and soon Linux, built on Eden and tuned for Hyrule Warriors: Age of Imprisonment. It has Switch 2 fixes, a custom ancient-style theme with a few easter eggs if your willing to find them. I hope you all enjoy i have spent ages working on this

Made by birdman311.

## Download
Get the ready-to-run Windows zip (and the early Linux AppImage) from the **Releases** page on the right.

You need your own keys, firmware and games, dumped from your own console. Construct includes none of them.

## Building from source (Windows)
Needs Visual Studio 2022 (C++ workload), Qt 6.9 (MSVC 64-bit), the Vulkan SDK and CMake + Ninja.
Edit the paths at the top of `configure.bat` and `build.bat` to match your PC, then run
`configure.bat` once and `build.bat` to build `build\bin\construct.exe`.

## License and credits
Construct is free software under the GNU GPL v3 (see `LICENSE.txt`). birdman311's own new parts
(the theme, glyph art and scrolling glyph ribbons) are also available under the MIT license.
See `CREDITS.txt` for everyone whose work Construct is built on, including Eden and yuzu.

Construct is not affiliated with or endorsed by Nintendo.
