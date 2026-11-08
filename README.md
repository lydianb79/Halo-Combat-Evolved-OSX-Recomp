# Halo Recomp — native macOS (Metal) static recompilation of Halo: Combat Evolved (Xbox)

This turns the original Xbox executable on **your own disc image** into C,
compiles it natively for Apple Silicon or Intel Macs, and runs it inside a
small reimplementation of the Xbox kernel, with a Metal window as the display.
No game code or data is included: everything game-specific is generated
locally from your ISO.

> **Status: playable.** The campaign runs from the intro through *The Pillar of
> Autumn* and onto *Halo* at a steady 30 fps (the Xbox's own frame rate), with
> GPU rendering in Metal at up to 4× resolution, sound and room-specific reverb,
> keyboard/mouse and controller input, and working saves. See *Current status*.

## Quick start (macOS)

Requirements: macOS 12+, Xcode or Command Line Tools (`xcode-select --install`), Python 3.
Full Xcode is needed to precompile the Metal shader; with only the Command
Line Tools the shader is compiled at launch instead.

```sh
cd "HaloRecomp"
make extract     # copies default.xbe out of ../Halo - Combat Evolved (USA).iso
make recomp      # x86 -> C translation (~20 s), writes generated/
make app         # builds build/Halo Recomp.app (first build compiles ~30 MB of C)
make run         # launches it
```

Bring-up and debugging without a window:

```sh
make debug       # headless runner with kernel-call tracing
```

The app reads the disc straight from the ISO (no 3.5 GB extraction). Saves and
the cache partition go to `xbox_hdd/`. Environment overrides: `HALO_ISO`,
`HALO_XBE`, `HALO_SAVES`, `HALO_TRACE_KERNEL=1`, `HALO_TRACE_CALLS=1`.

## Controls

Every Xbox control can be remapped on the **Input** page of **Settings → Settings…** (⌘,): two keyboard keys or mouse buttons per control, plus one controller button or stick direction. There are also options to invert look and set the stick dead zone. Settings are saved and kept across launches; **Restore Defaults** brings back the mapping below. A connected Xbox, PlayStation or MFi controller works without setup.

| Default keys | Xbox control |
|---|---|
| W A S D | move (left stick) |
| Arrow keys | D-pad, and look (right stick); I J K L also look |
| Space / Return | A |
| Esc / Backspace | B |
| R / Q | X / Y |
| F / E | black / white |
| Left Shift | left trigger |
| Left click / Left Control | right trigger |
| C / V | left / right stick click |
| P / Tab | START / BACK |

## Graphics

**Settings → Settings…** (⌘,) opens one window with **Input**, **Graphics** and **Audio** pages; the Settings menu also has **Full Screen** (⌃⌘F). Everything applies immediately and is saved. Graphics page:

- **Full screen** (⌃⌘F); Esc leaves full screen (can be turned off; while it is on, Esc is not sent to the game in full screen, Backspace still is B); start in full screen; hide the pointer in full screen.
- **Window size** 1×–4×.
- **Render resolution** 1×–4× (sharper 3D: the game's render targets are drawn at up to 2560×1920 and the picture is scaled to the window).
- **Aspect ratio:** 4:3 (original), 16:9 (stretched), or stretch to the window.
- **Scaling:** fit to window, integer scale (pixel-perfect), or fill the window (crop).
- **Screen filter** (how the picture is enlarged): nearest, bilinear, sharp bilinear (default), bicubic, Lanczos.
- **Texture filtering** (inside the game's 3D scene): as the game sets it (default), fastest (nearest texel, no mipmaps), nearest, bilinear, trilinear, anisotropic 2×/4×/8×/16×. Smoother modes cost speed only with the CPU renderer. "Fastest" skips mipmaps and looks blocky and sparkly. Headless: `HALO_TEX_FILTER=0..8`.

Recommended: screen filter **Bilinear**, texture filtering **Trilinear** (or as the game sets it), render resolution **2×** or **3×**.
- **Scanlines**, **brightness** and **gamma**.
- **Frame rate and speed overlay.**
- **Renderer:** Metal (default: pixel work on the Mac's GPU) or CPU (the reference software renderer); applies at the next start. `HALO_RENDERER=cpu` forces the CPU renderer.
- **CPU render threads** for the CPU renderer (automatic = one per core; applies at the next start).

## Audio

The game's sound plays through the Mac's current output device (AVAudioEngine). The Audio page
has **volume**, **mute**, **mute while another app is in front** (on by default) and a
**reverb** level (0–400 %, default 100 %).

The Xbox mixes sound in hardware: DirectSound programs up to 256 voices in the MCPX audio
processor (APU), and two DSPs add effects and encode the output. `runtime/apu.c` emulates the
voice processor and mixes every voice straight to 48 kHz stereo: one-shot and looping
buffers, streams (music, speech, video soundtracks), PCM and Xbox ADPCM, pitch and the
per-speaker volumes, with 5.1 mix bins folded down to stereo. The DSPs are not emulated;
DirectSound's commands to them are acknowledged so it never waits.

**Reverb** is approximated: each voice's I3DL2 send (mix bin 10) feeds a Freeverb network
(8 combs and 4 allpasses per side, plus a pre-delay). The room follows the game: Halo's
I3DL2 listener settings for each area (decay time, high-frequency damping, room and reverb
levels, delays, diffusion) are read from DirectSound's listener object and mapped onto the
reverb, with smooth transitions between areas.

Environment: `HALO_REVERB=0..400` (level in percent, overrides the slider),
`HALO_REVERB_FIXED=1` (one fixed room), `HALO_TRACE_ROOM=1` (log the room settings).
Headless: `HALO_DUMP_WAV=out.wav` writes everything that was mixed; `HALO_TRACE_APU=1` logs
the voice methods; `HALO_APU_BINS=1` logs mix-bin use and reverb levels.

## How it works

```
 Halo (USA).iso ──xiso.py──▶ default.xbe ──recomp.py──▶ generated/*.c ─┐
                                                                      ├─ clang ─▶ Halo Recomp.app
 runtime/ (CPU model, Xbox kernel, file system, HLE) ─────────────────┤
 platform/macos/ (AppKit window, MTKView, Metal shaders, gamepad) ────┘
```

| Part | What it does |
|---|---|
| `tools/xiso.py` | Lists and extracts files from Xbox disc images (XDVDFS). |
| `tools/xbe.py` | Parses the XBE: sections, entry point, kernel imports, XDK library versions. |
| `recomp/disasm.py` | Decodes x86 with `llvm-objdump` (ships with Xcode as `objdump`). |
| `recomp/recomp.py` | Finds functions (entry point, calls, function-pointer tables, immediates), recovers switch jump tables, and emits one C function per guest function. |
| `recomp/x86c.py` | Translates instructions: integer ALU with lazy flags, x87 FPU, SSE, string ops. |
| `runtime/cpu.h` | Guest CPU state. Guest memory is a 4 GiB host reservation, so guest pointers are plain offsets. |
| `runtime/runtime.c` | XBE loader, indirect-call dispatch, thread blocks (KPCR/TLS), global guest lock and preemption points. |
| `runtime/kernel.c` | About 110 xboxkrnl exports: memory, events, mutants, timers/DPCs, threads, files, time, SHA-1/HMAC/RC4, and others. |
| `runtime/vfs.c` | `D:\` served from the ISO; `C:`/`E:`/`T:`/`U:`/`Z:` and the raw config area map to `xbox_hdd/`. |
| `runtime/nv2a.c` | GPU (NV2A) device model behind the MMIO window: pushbuffer FIFO, vblank, flips, reports. |
| `runtime/nv2a_raster.c` | Kelvin 3D state: vertex fetch, vertex programs (SIMD CPU interpreter, and translation to Metal), register combiners, textures; hands draws to a backend through `gpu_backend.h`. |
| `runtime/apu.c` | MCPX audio processor: voice processor, streams, ADPCM, mixing to 48 kHz stereo, reverb. |
| `runtime/xinput.c` | Gamepad input bound to the game's XInput entry points. |
| `runtime/mmx.h` | MMX helpers for recompiled code (used by Bink). |
| `runtime/hle.c` | Swaps statically linked XDK functions (Direct3D, DirectSound, input) for host code, bound by address through `extracted/hle_map.txt`. |
| `platform/macos/metal_backend.m`, `Shaders.metal` | Metal renderer: one specialized fragment shader per combiner setup (compiled in the background), GPU vertex programs, depth/stencil, polygon offset, fog, visibility queries, render scale. |
| `platform/macos/` (other) | Native app: window and presentation, settings window, GameController and keyboard/mouse input, AVAudioEngine output, and a game thread with a 64 MB stack. |

### Design choices
* **Static, ahead-of-time translation.** Every reachable guest function becomes a C function `f_XXXXXXXX(CPU*)`. Calls are direct C calls; indirect calls go through a sorted address table.
* **Lazy flags.** Flag-setting instructions record their operands and result, and conditions are computed only when a branch needs them. The compiler folds the common `cmp`/`jcc` pairs into a single branch.
* **One guest lock with preemption.** Guest threads are real host threads, but only one runs guest code at a time, as on the Xbox's single-core CPU. Blocking kernel calls release the lock, and loop back-edges carry a cheap safepoint, so busy-waits can't starve other threads.
* **Library code is recompiled too.** XDK code (D3D8, DSOUND, XNET, Bink, XPP) is recompiled like the rest, but calls into those sections go through dispatch, so any function can be replaced by a host implementation without regenerating code.

## Current status (measured on the US disc, XBE built 2001-10-12)

**Playable.** Boot → intro → menu → campaign: *The Pillar of Autumn* plays through to
*Halo*. On an Apple Silicon Mac it holds a steady 30 fps on the ship and on the planet,
in fights and corridors; the GPU is about 3–9 % busy at 1× resolution.

* 13,917 functions recovered (36 C files); **99.88 %** of instructions translate. The rest
  is almost entirely junk decoded from data, which never executes.
* **Graphics:** the NV2A command stream is decoded and drawn with Metal: vertex programs run
  on the GPU, register combiners become specialized shaders, plus stencil, polygon offset
  (decals), fog, visibility tests (lens flares) and render resolutions up to 4×. A CPU
  reference renderer is still available.
* **Sound:** voices, streams and music, with room-specific reverb.
* **Saves:** profiles and checkpoints go to `xbox_hdd/`; profiles can be deleted and their
  names reused.
* **Known:** the speed overlay shows 50 % during the intro movie (the movie and its audio
  play at the correct speed). Points and lines are not drawn (Halo doesn't use them).

### What the runtime models
* **Memory:** 128 MB of guest RAM, mapped at 0, 0x80000000 and 0xF0000000; contiguous and
  physical allocations stay in the low 64 MB as on a retail Xbox. Kernel pages below 0x20000,
  the XBE image mapped at virtual 0x10000. Halo's fixed-address `XPhysicalAlloc` requests
  (0x61000 and 0x3A6000) are satisfied exactly.
* **GPU:** register file, pushbuffer FIFO with jump/call, DMA objects via RAMHT, surfaces and
  clears, semaphores and reports, flip indices, software-method interrupts, vblank interrupts
  that follow the display's refresh.
* **Audio:** the APU front end, voice processor and audio-memory fences; GP DSP mailbox
  commands are acknowledged.
* **Kernel:** about 120 exports, including user APCs for overlapped I/O, interrupts and DPCs,
  timers, file deletion, and the MCPX status port.
* **Threads:** one guest lock with fair hand-off at loop back-edges.

## Diagnostics

The app writes `build/halo.log` (the previous run goes to `build/halo.prev.log`), with
`[pace]` and `[gpu]` lines about frame timing and GPU load. Useful switches:
`HALO_RENDERER=cpu`, `HALO_GPU_VS=0` (vertex programs on the CPU), `HALO_NO_FOG=1`,
`HALO_HSR=1` (Apple hidden-surface removal; causes artifacts), `HALO_GPU_THREAD=1`
(experimental asynchronous GPU thread), `HALO_TEX_STATS=1`, `HALO_FEATURE_STATS=1`.

## Roadmap

1. **CPU headroom:** run the pushbuffer on its own thread (needs D3D's progress markers in memory).
2. **Audio:** full GP DSP emulation instead of the reverb approximation; volume envelopes and filters.
3. **Remaining visual glitches** as they turn up.
4. **Guest SEH** (`RtlRaiseException`/`RtlUnwind`).

## Bring-up loop

1. `make debug` and read the last lines.
2. `[trap] instruction not supported` → add it to `recomp/x86c.py`.
3. `[dispatch] no recompiled function at X` → it's appended to `missing_seeds.txt`; move it into `seeds.txt`, then `make recomp headless`.
4. `[kernel] unimplemented kernel function` → add it to `runtime/kernel.c`.

## Legal

Bring your own legally obtained disc image. `extracted/`, `generated/` and
`xbox_hdd/` hold material derived from your copy: keep them private and don't
redistribute them. This project contains no Microsoft or Bungie code or assets.
