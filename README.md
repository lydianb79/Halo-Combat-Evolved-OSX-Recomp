# Halo Recomp — native macOS (Metal) static recompilation of Halo: Combat Evolved (Xbox)

This turns the original Xbox executable on **your own disc image** into C,
compiles it natively for Apple Silicon or Intel Macs, and runs it inside a
small reimplementation of the Xbox kernel, with a Metal window as the display.
No game code or data is included: everything game-specific is generated
locally from your ISO.

> **Status: early bring-up. It does not play yet.** The recompiled game boots,
> initializes graphics, sound and network, loads the menu map from your disc and
> starts the intro video; nothing is drawn yet. See *Current status* and *Roadmap*.

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
- **Aspect ratio:** 4:3 (original), 16:9 (stretched), or stretch to the window.
- **Scaling:** fit to window, integer scale (pixel-perfect), or fill the window (crop).
- **Screen filter** (how the 640×480 picture is enlarged): nearest, bilinear, sharp bilinear (default), bicubic, Lanczos.
- **Texture filtering** (inside the game's 3D scene): as the game sets it (default), fastest (nearest texel, no mipmaps), nearest, bilinear, trilinear, anisotropic 2×/4×/8×/16×. Smoother modes cost speed only with the CPU renderer. Headless: `HALO_TEX_FILTER=0..8`.
- **Scanlines**, **brightness** and **gamma**.
- **Frame rate and speed overlay.**
- **Renderer:** Metal (default: pixel work on the Mac's GPU) or CPU (the reference software renderer); applies at the next start. `HALO_RENDERER=cpu` forces the CPU renderer.
- **CPU render threads** for the CPU renderer (automatic = one per core; applies at the next start).

## Audio

The game's sound plays through the Mac's current output device (AVAudioEngine). The Audio page
has **volume**, **mute** and **mute while another app is in front** (on by default).

The Xbox mixes sound in hardware: DirectSound programs up to 256 voices in the MCPX audio
processor (APU), and two DSPs add effects and encode the output. `runtime/apu.c` emulates the
voice processor and mixes every voice straight to 48 kHz stereo: one-shot and looping
buffers, streams (music, speech, video soundtracks), PCM and Xbox ADPCM, pitch and the
per-speaker volumes, with 5.1 mix bins folded down to stereo. The DSPs are not emulated, so
**reverb and other environmental effects are missing**; DirectSound's commands to them are
acknowledged so it never waits.

Headless: `HALO_DUMP_WAV=out.wav` writes everything that was mixed; `HALO_TRACE_APU=1` logs
the voice methods.

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
| `runtime/nv2a.c` | GPU (NV2A) and APU-stub device models behind the MMIO window; pushbuffer decoder. |
| `runtime/mmx.h` | MMX helpers for recompiled code (used by Bink). |
| `runtime/hle.c` | Swaps statically linked XDK functions (Direct3D, DirectSound, input) for host code, bound by address through `extracted/hle_map.txt`. |
| `platform/macos/` | Native app: Metal presentation of a 640×480 framebuffer (4:3 letterboxed), GameController input, and a game thread with a 64 MB stack. |

### Design choices
* **Static, ahead-of-time translation.** Every reachable guest function becomes a C function `f_XXXXXXXX(CPU*)`. Calls are direct C calls; indirect calls go through a sorted address table.
* **Lazy flags.** Flag-setting instructions record their operands and result, and conditions are computed only when a branch needs them. The compiler folds the common `cmp`/`jcc` pairs into a single branch.
* **One guest lock with preemption.** Guest threads are real host threads, but only one runs guest code at a time, as on the Xbox's single-core CPU. Blocking kernel calls release the lock, and loop back-edges carry a cheap safepoint, so busy-waits can't starve other threads.
* **Library code is recompiled too.** XDK code (D3D8, DSOUND, XNET, Bink, XPP) is recompiled like the rest, but calls into those sections go through dispatch, so any function can be replaced by a host implementation without regenerating code.

## Current status (measured on the US disc, XBE built 2001-10-12)

**Boots, plays the intro, reaches the main menu with its music, and starts the attract-mode
video when idle, with no traps. Rendering runs on the Mac's GPU (Metal) at full speed, with a
software reference renderer; input and sound work.**

* 13,554 functions and 780,157 instructions recovered; **99.88 %** translate. The rest is
  almost entirely junk decoded from data, which never executes.
* Measured with your ISO, in a headless Linux build:
  1. XAPI startup; `T:`/`U:` saves, `Z:` utility drive formatted, Halo cache files created
  2. Direct3D: real GPU command stream (pushbuffer) consumed, vblank interrupts at 60 Hz,
     software-method traps, buffer flips one per vblank
  3. DirectSound, network stack and the XPP input drivers initialize
  4. `d:\maps\ui.map` loaded from the ISO and copied into the `Z:` cache
  5. `d:\\bink\\intro.bik` plays (Bink, MMX); then the main menu (`z:\\lastprof.txt` lookup),
     and after idling `d:\\bink\\attract2.bik`
  6. Menu rendered by the GPU backend; music and sound effects through the APU model

### What the runtime models
* **Memory:** exactly the Xbox's 64 MB of physical RAM, mapped at 0, 0x80000000 and 0xF0000000:
  kernel pages below 0x61000, the XBE image's pages near the top (mapped at virtual 0x10000), GPU
  instance memory in the top 64 KB, everything else allocated top-down. Halo's fixed-address
  `XPhysicalAlloc` requests (0x61000 and 0x3A6000) are satisfied exactly. Kernel pool uses 16-byte
  size classes.
* **GPU (`runtime/nv2a.c`):** register file, pushbuffer FIFO with jump/call, DMA objects via
  RAMHT, surfaces and clears, semaphores, flip indices, NOP software-method interrupts,
  vblank interrupt, framebuffer scan-out to the window.
* **Kernel:** about 120 exports, including user APCs for overlapped I/O, interrupts and DPCs,
  timers, and the MCPX status port.
* **Threads:** one guest lock with fair hand-off at loop back-edges.

## Roadmap (in order)

1. **Rendering gaps:** stencil, fog, points and lines, bump and other texture modes.
2. **Audio effects:** the DSP (GP) effects such as reverb; volume envelopes and filters.
3. **Guest SEH** (`RtlRaiseException`/`RtlUnwind`).

## Bring-up loop

1. `make debug` and read the last lines.
2. `[trap] instruction not supported` → add it to `recomp/x86c.py`.
3. `[dispatch] no recompiled function at X` → it's appended to `missing_seeds.txt`; move it into `seeds.txt`, then `make recomp headless`.
4. `[kernel] unimplemented kernel function` → add it to `runtime/kernel.c`.

## Legal

Bring your own legally obtained disc image. `extracted/`, `generated/` and
`xbox_hdd/` hold material derived from your copy: keep them private and don't
redistribute them. This project contains no Microsoft or Bungie code or assets.
