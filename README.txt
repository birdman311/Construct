Construct - Nintendo Switch / Switch 2 emulator for Windows (x64)
==================================================================
Made by birdman311.

Construct is built from Eden (a yuzu-based emulator) with extra Switch 2 work, tuned mainly for
Hyrule Warriors: Age of Imprisonment.

Getting started
1. Start construct.exe.
2. Construct needs your own prod.keys and firmware, dumped from your own console. They are not
   included. Put prod.keys in %APPDATA%\eden\keys and install the firmware from the Tools menu.
3. Add your game folder (double-click the game list) and start a game.

Requirements
- Windows 10/11 64-bit, a GPU with Vulkan 1.3 (NVIDIA recommended; tested on an RTX 4060).
- RAM: 16 GB or more strongly recommended. even with 16 gb ram i would implore you to add extra virtual memory try adding atleast 8 extra from storage however anything above 24gb should be perfectly fine.

- The Microsoft Visual C++ runtime files are included. If construct.exe still says a
  "MSVCP140.dll" or "VCRUNTIME140.dll" is missing, install the "Visual C++ Redistributable x64"
  from Microsoft.

Notes
- Settings, saves, keys and the shader cache live in %APPDATA%\eden (shared with Eden).
- The first time a game runs it compiles shaders while you play, so expect stutter at first.
- Logs for bug reports: %APPDATA%\eden\log (launch_<title id>.log and eden_log.txt).
- Hyrule Warriors: Age of Imprisonment: use Docked mode, GPU accuracy "Fast".
- CPU cores (Emulation > Configure > CPU, or per game in its Properties): 4 to 6 emulated cores
  6 helps games that run threads on the extra cores like (Pokopia: for example went from FPS 48.7 FPS -> 58.2 FPS on its title screen). Takes effect when a game starts.
- Keep a game's memory layout on the default (global) setting. Switch 2 games need 12 GB; with
  a smaller layout Hyrule Warriors quits back to the game list at boot.
- For freeze reports, start construct.exe with the environment variable EDEN_STALL_WATCHDOG=1;
  if the picture stops for 4+ seconds it writes where every game thread is into the launch log. For debugging on your own

License and credits
Construct is free software under the GNU General Public License v3 (see LICENSE.txt and the
LICENSES folder). The source code is in Construct-source.zip next to this package.
birdman311's own new parts (the Construct theme, glyph script art and scrolling glyph ribbons)
are also available under the MIT license. See CREDITS.txt for everyone whose work is in Construct.
