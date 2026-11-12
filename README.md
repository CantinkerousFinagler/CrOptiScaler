# CrOptiScaler

An [OptiScaler](https://github.com/optiscaler/OptiScaler) fork for games played in VR with
[R.E.A.L. VR](https://www.patreon.com/realvr). It adds **foveated DLSS**: full quality in the middle of each eye,
a cheap pass for the rest, for a higher frame rate.

Experimental, tested on one machine. Not affiliated with R.E.A.L. VR, NVIDIA or the OptiScaler team.

## You need

- R.E.A.L. VR working in the game, with its **DL(S)SS option on**.
- An **NVIDIA RTX** card.
- **DLSS or Ray Reconstruction on** in the game.

## Install

1. Unzip next to the game's executable: `winmm.dll`, `OptiScaler.ini`, `nvngx.dll_dlssnr.dll` and the `OptiScaler`
   folder. Do not replace R.E.A.L. VR's `dxgi.dll`.
2. Start the game and press **Insert** for the menu.

Optional:
- **Neural Rendering** also needs NVIDIA's `nvngx_dlssnr.dll`, which is not in the zip. Put it next to the game's
  executable. Which version you need is in [INSTALL-DLSSNR.md](INSTALL-DLSSNR.md) under "The runtime"; its
  install steps are for the general fork, follow the ones here.

- **OXR Headset Menu.** If our menu does not show in the headset in your game, copy the `xr_observer` folder
  from the zip's `OXR Headset Menu` folder into `<game>\OptiScaler\`. Its switch is
  **Show this menu in the headset**. Skip it where the menu already shows, or you get two. Not for Vulkan games.

Not for multiplayer games with anti-cheat.

## What it does, and how to turn each thing on

All of these are in the menu (**Insert**).

### Foveated DLSS: the crop

With the crop on, DLSS only runs at full quality in a box in the middle of each eye, and the rest of the eye gets
a much cheaper pass. The middle looks the same as before, the outer part is a little softer, and the frame rate
goes up, as long as **Cheap periphery** is checked and set to preset E while the crop itself is set to something
heavier like K or M (these are the defaults).

- **Turn it on:** tick **Crop on** under **Foveated SR**. Nothing else needs setting.
- **Width / Height:** how big the full-quality box is. Smaller is faster, but you see the softer part sooner.
- **Edge blend:** how wide the soft edge between the box and the rest is.
- **Left / right, Up / down:** move the box if the sharp part does not sit where you look.
- **Crop preset:** the DLSS preset used inside the box.
- **Cheap periphery** (on by default): the cheap pass outside the box. Leave it on; it is where the gain comes
  from. **Periphery preset** is the DLSS preset it uses.

### Ray Reconstruction games

If the game uses Ray Reconstruction (DLSS's denoiser for ray tracing), the crop runs Ray Reconstruction inside
the box and plain DLSS outside it, as long as the Foveated DLSS crop is turned on. That is a significant saving,
and the price is that ray-traced noise is not cleaned outside the box.

- **SR periphery in RR mode** (on by default): untick it to get Ray Reconstruction on the whole picture again, at
  a higher cost.
- **Change DLSS settings with the crop off.** If you press Apply on a DLSS setting in this menu while the crop
  is on, Ray Reconstruction can be gone from the whole picture once you switch the crop off. Turn Ray
  Reconstruction off and on again in the game's own settings to get it back.

### Neural Rendering (NR)

NVIDIA's AI model that changes the lighting, detail and color of the picture. It is heavy, so here it runs per eye
and, by default, only in the middle of the eye like the crop.

- **Turn it on:** tick **Enable Neural Rendering** under **DLSS Neural Rendering**. It needs NVIDIA's
  `nvngx_dlssnr.dll` (see Install).
- **Foveated NR** (on by default): NR only in the middle of the eye. With the crop on, it follows the crop's box.
- You can also enable NR without the Foveated DLSS crop and use it like you normally would, including fiddling
  with all the NR settings and multiple passes.

### Crimson Desert: Per-eye lighting

In VR the game's lighting shows bands beside characters and black spots on shadowed bodies. This fixes most of it.

- **Turn it on:** tick **Per-eye lighting** under **Crimson Desert**. It is off at every launch.

### Gamepad

By default this mod leaves the gamepad alone, so the game and R.E.A.L. VR's own overlay keep working as usual.
Use the mouse in our menu.

- **Gamepad in this menu (after a restart)**, top of the left column: lets the gamepad move through our menu. In
  some games R.E.A.L. VR's overlay then stops opening from the gamepad.

### Keys

| Key | Does |
|---|---|
| Insert | opens the menu |
| Page Up | FPS overlay |
| Page Down | switches the FPS overlay's view |
| End | OptiScaler's frame generation (does not work in VR) |

Under **Keybinds** you can change these, and give a key to **Foveal Crop** and **Per-eye Lighting**. Other mods
can use the same keys: if one reacts to Insert, change the key in that mod or here.

### Is the crop saving anything?

Compare the same DLSS preset with the crop on and with it off. A light preset without the crop against a heavy
preset with it will look like no gain, because you are getting the heavy preset's picture for about the light
one's cost. And if your headset already holds its refresh rate, the saving shows as headroom, not as more frames.

## Per game

**Crimson Desert**
- Per-eye lighting is off at every launch. Turn it on in the menu.
- **Turn it off before changing graphics settings**, then on again. Changing settings with it on can crash.
- Some spots on quick head turns and some haloing on characters can remain.
- A game patch can switch part of the fix off. The menu says what is running.
- To keep the fix out altogether: `PerEyeLighting=false` in `OptiScaler.ini`.

**Red Dead Redemption 2**
- NR's area does not follow the crop. Set NR's **Fovea width** and **Fovea height** to the crop's size.
- Our menu shows in the headset at the main menu only, not in the game world.

**Grand Theft Auto V Enhanced**
- Can crash at start now and then. I was always able to just launch again and it worked fine.

**Hogwarts Legacy**
- The Ray Reconstruction crop shimmers. Leave the crop off there. Not fixed after the latest DLSS 5 update.

**Star Wars Outlaws, Avatar: Frontiers of Pandora**
- No known issues.

## Limitations

- With the crop on, Ray Reconstruction runs only inside the crop, so ray-traced noise can show outside it.
- The first time you turn the crop on with Ray Reconstruction, one "upscaler failed" message can appear. Ignore it.
- With the gamepad setting on, R.E.A.L. VR's overlay may not open from the gamepad.
- With R.E.A.L. VR's DL(S)SS off the crop still runs, but that is barely tested.
- AMD and Intel cards: untested.
- The `OptiScaler` folder leaves out `libxell.dll` (Intel's low-latency library, used by XeSS frame
  generation). Do not add it: with it there, Grand Theft Auto V Enhanced and Red Dead Redemption 2 do not
  start.

## Reporting a problem

`OptiScaler.log` is written next to the DLL and replaced at every launch. Copy it right after the bad run and
open an issue with it, the game, your GPU and headset.

## Build

Visual Studio 2022, C++ desktop workload. `git clone --recurse-submodules`, then build `OptiScaler.sln`
(Release, x64).

## Credits and license

A modified version of the [OptiScaler Neural Rendering fork](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass)
by wilsjo2 (v0.7.7, September 2026), which builds on [OptiScaler](https://github.com/optiscaler/OptiScaler) and
[Dagherbou's NR work](https://github.com/Dagherbou/OptiScaler_DLSSNR), with color processing from
[RenoDX](https://github.com/clshortfuse/renodx). License unchanged: GPL-3.0. [Full credits](docs/CREDITS.md) ·
[License](LICENSE)

Developed with my mAIn man [Claude](https://claude.com/claude-code) (Anthropic).

If this is useful to you, you can support the work on [Ko-fi](https://ko-fi.com/cantinkerousfinagler).
