<p align="center"><img src="dist/construct.png" width="160" alt="Construct icon"></p>

Construct is a Nintendo Switch and Switch 2 emulator for Windows and soon Linux, built on Eden and tuned for Hyrule Warriors: Age of Imprisonment. It has Switch 2 fixes, a custom ancient-style theme with a few easter eggs if your willing to find them. I hope you all enjoy i have spent ages working on this

DOWNLOAD
Download Construct-windows-x64.zip** from [Releases](../../releases) for the finished exe.

You need your own keys, firmware and games, dumped from your own console. Construct includes none of them.

Features
Sharper picture: the emulator lets you increase the quality of Hyrule warriors age of imprisonment compared to the standard gameplay on the switch 2 you can increase the game quility from 2x to even 3x compared to the console quality. the real switch 2 draws at 1600x900 and stretches it up, which leaves characters grainy. At 2X, Construct draws the whole scene and every character at 3200x1800 (double the pixels each way) and shrinks it down to the screen, so hair, armour, cloaks and faces come out clean. 3X goes to 4800x2700 for very powerful PCs.

Automatic resolution: Construct automatically changes the image quility based on your monitor: 1080p gets 1X, 1440p and 4K get 2X, bigger screens get 3X. Change it any time in Configure > Graphics > Resolution (1X Standard, 2X Double, 3X Triple).

You can set the number of emulated CPU cores anywhere from 4 to 6. This helps increase performance. For example, when I tested Pokopia it went from 48.7 FPS to 58.2 FPS on its title screen.

another feature is it Works with Switch 1 games too, just like the Eden emulator it's built on.

I also have added a custom ancient-style theme: dark stone, gold frames and teal glow. as well as An original glyph script that scrolls across the menu and status bars like a conveyor belt. The glyphs spell real messages, so try to find all the easter eggs i have hidden.

another feature most emulators forget is displaying the real game size i know its a small feature but its still a feature

you also can drop your Custom icons for games in their folder. Icon.png into the game's folder. And it will automatically display the chosen image 

Optimization features

Construct has been optimized a lot, with fixes made specifically for Hyrule Warriors: Age of Imprisonment and other Switch 2 games.

 Faster loading after cutscenes: shaders are prepared at startup, which cut the wait after cutscenes from about 55 seconds to about 10.

 Less memory: I fixed memory leaks and oversized buffers, so Construct runs better even on PCs with less RAM, like mine with only 8 GB of DDR4.

Stability: Switch 2 games make a huge number of calls to Switch 2 features. I fixed the ones that caused screen flashing, broken particle effects, bosses that couldn't be beaten, black screens and crashes, so Hyrule Warriors: Age of Imprisonment is fully playable. Big fights can still dip in FPS on slower PCs, and I'm still working on that.

Linux
 An early Linux version, is on the way once i finish testing it.

 Recommended settings
 RAM: 16 GB or more, plus extra virtual memory (page file) for Switch 2 games.
 Hyrule Warriors: Age of Imprisonment: Docked mode, GPU accuracy "Fast", resolution 2X. Keep the memory layout on the default setting.
 First runs are slow: expect some stutter the first time a game runs, while its shaders build.

Building from source (Windows)
Needs Visual Studio 2022 (C++ workload), Qt 6.9 (MSVC 64-bit), the Vulkan SDK and CMake + Ninja.
Edit the paths at the top of `configure.bat` and `build.bat` to match your PC, then run
`configure.bat` once and `build.bat` to build `build\bin\construct.exe`.

License and credits
Construct is free software under the GNU GPL v3 (see `LICENSE.txt`). MY own new parts
(the theme, glyph art and scrolling glyph ribbons) are also available under the MIT license.
See `CREDITS.txt` for everyone whose work Construct is built on, including Eden and yuzu.

Construct is not affiliated with or endorsed by Nintendo.

Disclaimer: I do not support piracy and am not trying to encourage or promote the use of piracy in any way.
