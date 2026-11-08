/* nv2a.c — model of the Xbox GPU (NV2A), plus a minimal audio-processor (APU)
 * stub, as seen by the statically linked
 * Direct3D library.
 *
 * Halo's own D3D8 code runs unmodified (recompiled). It programs GPU registers
 * through the MMIO window at 0xFD000000 and streams commands through a
 * "pushbuffer" in RAM. This file:
 *   - answers register reads the way an idle GPU would (status, timers, flush bits)
 *   - consumes the pushbuffer whenever D3D moves the DMA PUT pointer and decodes
 *     the commands (NV097 "Kelvin" 3D methods) — currently clears, surfaces,
 *     semaphores and references; drawing is logged for the Metal backend
 *   - raises the vertical-blank interrupt at 60 Hz through D3D's own ISR
 *   - scans the displayed framebuffer out to the host window (gfx_present_frame)
 *
 * Registers not modelled behave like plain memory (reads return the last write).
 */
#include "nv2a.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "nv2a_raster.h"
#include "gpu_backend.h"
#include "runtime.h"

static void pace_flip(void);
static void nv2a_vblank(uint64_t now);
static int gpu_threaded(void);

#define REG(off) (*(uint32_t *)(g_mem + MMIO_BASE + (off)))

/* register offsets (relative to 0xFD000000) */
#define PMC_BOOT_0          0x000000
#define PMC_INTR_0          0x000100
#define PMC_INTR_EN_0       0x000140
#define PFIFO_INTR_0        0x002100
#define PFIFO_RAMHT         0x002210
#define PTIMER_TIME_0       0x009400
#define PTIMER_TIME_1       0x009410
#define PFB_CFG0            0x100200
#define PFB_CSTATUS         0x10020C
#define PFB_WBC             0x100410
#define PGRAPH_INTR         0x400100
#define PGRAPH_NSOURCE      0x400108
#define PGRAPH_INTR_EN      0x400140
#define PGRAPH_TRAPPED_ADDR 0x400704
#define PGRAPH_TRAPPED_DATA 0x400708
#define PGRAPH_INTR_ERROR   0x00100000u           /* software method / notify trap */
#define PMC_INTR_PGRAPH     (1u << 12)
#define PGRAPH_STATUS       0x400700
#define PGRAPH_SURFACE      0x400710          /* flip indices: WRITE_3D 22:20, READ_3D 26:24, MODULO_3D 30:28 */
#define PGRAPH_INCREMENT    0x40071C          /* bit1: advance READ_3D (written by D3D's vblank handler) */
#define PCRTC_INTR_0        0x600100
#define PCRTC_INTR_EN_0     0x600140
#define PCRTC_START         0x600800
#define PCRTC_RASTER        0x600808
#define PRAMIN              0x700000
#define USER_BASE           0x800000          /* channel 0 user area */
#define USER_DMA_PUT        (USER_BASE + 0x40)
#define USER_DMA_GET        (USER_BASE + 0x44)
#define USER_REF            (USER_BASE + 0x48)

#define PMC_INTR_PCRTC      (1u << 24)
#define GPU_IRQ_LEVEL       3

static uint32_t g_gpu_kinterrupt;
static uint64_t g_boot_ns, g_next_vblank_ns;
static uint32_t g_frames;
static int g_trace;

/* display state (from AvSetDisplayMode) */
static uint32_t g_disp_pitch = 640 * 4, g_disp_w = 640, g_disp_h = 480;

/* Kelvin (3D) state that the command decoder tracks */
static struct {
    uint32_t dma_color, dma_zeta, dma_semaphore, dma_notifies, dma_a, dma_b;
    uint32_t surface_format, surface_pitch, color_offset, zeta_offset;
    uint32_t clip_h, clip_v;
    uint32_t clear_color, clear_zstencil;
    uint32_t semaphore_offset;
    uint32_t dma_report, zpass_enable;
    uint64_t reports;
    uint32_t begin_end;
    uint64_t draws, methods;
} K;

static uint32_t g_method_seen[0x2000 / 4];
static uint32_t g_flips_requested;
static int g_stall_req;                 /* FLIP_STALL seen: pause the FIFO until the display has moved on */
static int g_stalled;
static uint32_t g_stall_since;          /* vblank count when the stall began */
static uint64_t g_stall_count, g_stall_frames, g_stall_timeouts, g_scanout_conflicts;

static uint32_t dma_base(uint32_t handle, uint32_t *limit);

void nv2a_register_interrupt(uint32_t level, uint32_t kinterrupt) {
    nv2a_raster_bind(dma_base);
    if (level == 5) apu_set_interrupt(kinterrupt);                 /* MCPX APU */
    if (level == GPU_IRQ_LEVEL) {
        g_gpu_kinterrupt = kinterrupt;
        runtime_log("[nv2a] GPU interrupt %s (KINTERRUPT %08x)", kinterrupt ? "connected" : "disconnected", kinterrupt);
    }
}

void nv2a_set_display(uint32_t fb, uint32_t pitch, uint32_t mode, uint32_t format) {
    (void)format;
    if (pitch) g_disp_pitch = pitch;
    g_disp_w = g_disp_pitch / 4;
    if (g_disp_w > 1920 || g_disp_w < 320) g_disp_w = 640;
    g_disp_h = g_disp_w == 720 ? 480 : g_disp_w * 3 / 4;
    if (fb) REG(PCRTC_START) = fb;
    runtime_log("[nv2a] display mode %08x: %ux%u pitch %u fb %08x", mode, g_disp_w, g_disp_h, g_disp_pitch, fb);
}

/* ------------------------------------------------------------- objects */
/* Look up a DMA object by handle: scan the hash table (RAMHT) in instance
 * memory, then read the object's base address. Returns 0 if not found. */
static uint32_t dma_base(uint32_t handle, uint32_t *limit) {
    if (!handle) return 0;
    uint32_t ramht = REG(PFIFO_RAMHT);
    uint32_t base = ((ramht >> 4) & 0x1F) << 12;
    uint32_t size = 1u << (((ramht >> 16) & 3) + 12);
    for (uint32_t o = 0; o < size; o += 8) {
        if (REG(PRAMIN + base + o) == handle) {
            uint32_t ctx = REG(PRAMIN + base + o + 4);
            uint32_t inst = (ctx & 0xFFFF) << 4;
            uint32_t w0 = REG(PRAMIN + inst), w1 = REG(PRAMIN + inst + 4), w2 = REG(PRAMIN + inst + 8);
            if (limit) *limit = w1;
            return (w2 & 0xFFFFF000u) | (w0 >> 20);
        }
    }
    return 0;
}

static uint32_t phys_to_guest(uint32_t p) { return GUEST_RAM_MIRROR1 | (p & (GUEST_RAM_SIZE - 1)); }   /* GPU sees physical RAM */

/* ------------------------------------------------------------- methods */
static uint32_t g_clear_rect_h, g_clear_rect_v;
static void update_surface(void);
static void do_clear(uint32_t flags) {
    update_surface();
    uint32_t rh = g_clear_rect_h, rv = g_clear_rect_v;
    if (!rh && !rv) {   /* no clear rect set: use the surface clip */
        uint32_t x0 = K.clip_h & 0xFFFF, w = K.clip_h >> 16, y0 = K.clip_v & 0xFFFF, h = K.clip_v >> 16;
        if (!w || !h) return;
        rh = x0 | (x0 + w - 1) << 16; rv = y0 | (y0 + h - 1) << 16;
    }
    nv2a_raster_clear(flags, K.clear_color, K.clear_zstencil, rh, rv);
    if (flags & 0xF0) gfx_clear(K.clear_color, flags);
}

static uint64_t g_method_count[0x2000 / 4];
static uint32_t g_method_last[0x2000 / 4];
static void dump_method_histogram(void) {
    runtime_log("[nv2a] method histogram (method: count last-value):");
    for (int i = 0; i < 0x2000 / 4; i++)
        if (g_method_count[i]) runtime_log("   %04x: %10llu  %08x", i * 4, (unsigned long long)g_method_count[i], g_method_last[i]);
    memset(g_method_count, 0, sizeof g_method_count);
}

static void update_surface(void) {
    uint32_t lim = 0, base = dma_base(K.dma_color, &lim);
    Nv2aSurface sf;
    memset(&sf, 0, sizeof sf);
    uint32_t pitch = K.surface_pitch & 0xFFFF;
    if (lim && pitch && K.color_offset <= lim) {
        sf.color_addr = phys_to_guest(base + K.color_offset);
        sf.pitch = pitch;
        sf.clip_x = K.clip_h & 0xFFFF; sf.clip_w = K.clip_h >> 16;
        sf.clip_y = K.clip_v & 0xFFFF; sf.clip_h = K.clip_v >> 16;
        /* never draw past the DMA object */
        while (sf.clip_h && K.color_offset + (uint64_t)pitch * (sf.clip_y + sf.clip_h) > (uint64_t)lim + 1) sf.clip_h--;
    }
    /* surface type (bits 11:8): 1 pitch (linear rows), 2 swizzled with log2 size in bits 23:16 / 31:24 */
    if (((K.surface_format >> 8) & 15) == 2 && sf.color_addr) {
        sf.swizzled = 1;
        sf.swz_log2w = (K.surface_format >> 16) & 15;
        sf.swz_log2h = (K.surface_format >> 24) & 15;
    }
    uint32_t zfmt = (K.surface_format >> 4) & 15, zpitch = K.surface_pitch >> 16;
    uint32_t zlim = 0, zbase = K.dma_zeta ? dma_base(K.dma_zeta, &zlim) : 0;
    if (zfmt && zpitch && zlim && sf.color_addr &&
        K.zeta_offset + (uint64_t)zpitch * (sf.clip_y + sf.clip_h) <= (uint64_t)zlim + 1) {
        sf.zeta_addr = phys_to_guest(zbase + K.zeta_offset);
        sf.zeta_pitch = zpitch;
        sf.zeta_fmt = zfmt;   /* 1 = Z16, 2 = Z24S8 */
    }
    nv2a_raster_set_surface(&sf);
}

/* The pushbuffer is consumed on its own thread (see gpu_thread_main), so the
 * game's CPU work and the GPU work (vertex programs, draw submission) run in
 * parallel. HALO_GPU_THREAD=0 consumes it on the writing thread instead. */
static __thread int t_gpu_thread;
static volatile int g_gpu_state;   /* diagnosis: 0 idle, 1 in run_fifo, 2 waiting for the guest lock, 3 in an ISR */

/* ------------------------------------------------------------- 2D objects */
/* D3D binds the 3D object (Kelvin, NV097) on subchannel 0, the image blit
 * (NV09F) on 3 and its surfaces object (NV062) on 4. Their methods share
 * numbers with Kelvin's, so they must not reach the 3D state. */
static struct { uint32_t dma_src, dma_dst, fmt, pitch, off_src, off_dst; } g_s2d;
static uint32_t g_blit_in, g_blit_out;

static uint32_t s2d_bpp(uint32_t fmt) {
    switch (fmt) {
    case 0x1: return 1;                                      /* Y8 */
    case 0x2: case 0x3: case 0x4: case 0x5: return 2;        /* X1R5G5B5, A1R5G5B5?, R5G6B5, Y16 */
    default: return 4;                                       /* X8R8G8B8, A8R8G8B8, Y32 */
    }
}

static void image_blit(uint32_t size) {
    uint32_t w = size & 0xFFFF, h = size >> 16;
    uint32_t sx = g_blit_in & 0xFFFF, sy = g_blit_in >> 16, dx = g_blit_out & 0xFFFF, dy = g_blit_out >> 16;
    uint32_t sp = g_s2d.pitch & 0xFFFF, dp = g_s2d.pitch >> 16, bpp = s2d_bpp(g_s2d.fmt);
    uint32_t src = phys_to_guest(dma_base(g_s2d.dma_src, NULL) + g_s2d.off_src);
    uint32_t dst = phys_to_guest(dma_base(g_s2d.dma_dst, NULL) + g_s2d.off_dst);
    if (!w || !h || !sp || !dp) return;
    static int logged;
    if (g_trace && logged++ < 40)
        runtime_log("[nv2a] blit %ux%u from %08x (%u,%u pitch %u) to %08x (%u,%u pitch %u), %u bytes/pixel",
                    w, h, src, sx, sy, sp, dst, dx, dy, dp, bpp);
    for (uint32_t y = 0; y < h; y++) {
        uint64_t s = (uint64_t)src + (sy + y) * sp + sx * bpp, d = (uint64_t)dst + (dy + y) * dp + dx * bpp;
        if (s + (uint64_t)w * bpp > 0x100000000ull || d + (uint64_t)w * bpp > 0x100000000ull) break;
        memmove(g_mem + d, g_mem + s, (size_t)w * bpp);
    }
    if (g_gpu_backend && g_gpu_backend->blit) {
        nv2a_raster_gpu_sync();
        g_gpu_backend->blit(src, sp, dst, dp, bpp, sx, sy, dx, dy, w, h);
    }
}

static void object_method(uint32_t subch, uint32_t m, uint32_t v) {
    if (subch == 4) {                                        /* NV062 surfaces 2D */
        switch (m) {
        case 0x0184: g_s2d.dma_src = v; return;
        case 0x0188: g_s2d.dma_dst = v; return;
        case 0x0300: g_s2d.fmt = v; return;
        case 0x0304: g_s2d.pitch = v; return;
        case 0x0308: g_s2d.off_src = v; return;
        case 0x030C: g_s2d.off_dst = v; return;
        }
    } else if (subch == 3) {                                 /* NV09F image blit */
        switch (m) {
        case 0x0300: g_blit_in = v; return;
        case 0x0304: g_blit_out = v; return;
        case 0x0308: image_blit(v); return;
        case 0x019C: case 0x02FC: return;                     /* surfaces object, operation (SRCCOPY) */
        }
    }
    static uint32_t seen[8][0x800];
    if (m < 0x2000 && !seen[subch & 7][m / 4]++ && g_trace)
        runtime_log("[nv2a] subchannel %u method %04x = %08x (not modelled)", subch, m, v);
}

static void kelvin_method(uint32_t subch, uint32_t m, uint32_t v) {
    K.methods++;
    if (subch != 0 && m != 0x0100 && m != 0x0000) { object_method(subch, m, v); return; }
    if (m >= 0x0260 && nv2a_raster_method(m, v)) {
        if (m < 0x2000) { g_method_count[m / 4]++; g_method_last[m / 4] = v; }
        return;
    }
    if (m < 0x2000) { g_method_count[m / 4]++; g_method_last[m / 4] = v; }
    if (m < 0x2000 && !g_method_seen[m / 4]++ && g_trace)
        runtime_log("[nv2a] first method sc%u %04x = %08x", subch, m, v);
    if (m == 0x0050) { REG(USER_REF) = v; return; }            /* SET_REFERENCE */
    if (m == 0x0100) {                                        /* NO_OPERATION */
        /* Xbox D3D uses NOPs with a parameter as software methods: the GPU
         * traps, records method + data, and interrupts the CPU. D3D's ISR
         * reads the trapped registers (flip bookkeeping, fences, callbacks). */
        if (v) {
            /* on the GPU thread, take the guest lock: the ISR is guest code, and
             * the vblank ISR (timer thread) touches the same registers */
            if (t_gpu_thread) { g_gpu_state = 2; guest_lock(); g_gpu_state = 3; }
            REG(PGRAPH_TRAPPED_ADDR) = (subch << 16) | (m & 0x1FFF);
            REG(PGRAPH_TRAPPED_DATA) = v;
            REG(PGRAPH_NSOURCE) = 1;                          /* NOTIFICATION */
            REG(PGRAPH_INTR) |= PGRAPH_INTR_ERROR;
            if (g_gpu_kinterrupt) {
                for (int tries = 0; tries < 4 && (REG(PGRAPH_INTR) & PGRAPH_INTR_ERROR); tries++)
                    kernel_raise_interrupt(g_gpu_kinterrupt);
            }
            if (REG(PGRAPH_INTR) & PGRAPH_INTR_ERROR) {
                if (g_trace) runtime_log("[nv2a] software method %08x not acknowledged", v);
                REG(PGRAPH_INTR) &= ~PGRAPH_INTR_ERROR;
            }
            if (t_gpu_thread) { guest_unlock(); g_gpu_state = 1; }
        }
        return;
    }
    /* buffer flipping: the GPU keeps read/write indices of the swap chain */
    if (m >= 0x0120 && m <= 0x012C) {
        if (m == 0x012C) {                                   /* frame complete: submit it (outside the lock) */
            g_flips_requested++;
            pace_flip();
            if (g_gpu_backend) { if (g_gpu_backend->flip) g_gpu_backend->flip(); else g_gpu_backend->flush(); }
        }
        /* the vblank ISR advances READ_3D in the same register: update under the guest lock */
        if (t_gpu_thread) guest_lock();
        uint32_t sf = REG(PGRAPH_SURFACE);
        uint32_t wr = (sf >> 20) & 7, rd = (sf >> 24) & 7, mod = (sf >> 28) & 7;
        switch (m) {
        case 0x0120: rd = v & 7; break;                                     /* FLIP_READ */
        case 0x0124: wr = v & 7; break;                                     /* FLIP_WRITE */
        case 0x0128: mod = v & 7; break;                                    /* FLIP_MODULO */
        case 0x012C: wr = mod ? (wr + 1) % mod : 0; break;                   /* FLIP_INCREMENT_WRITE */
        }
        REG(PGRAPH_SURFACE) = (sf & ~0x77700000u) | (wr << 20) | (rd << 24) | (mod << 28);
        if (t_gpu_thread) guest_unlock();
        return;
    }
    if (m == 0x0130) { g_stall_req = 1; return; }             /* FLIP_STALL: checked by run_fifo */
    if (m == 0x0000) return;                                  /* SET_OBJECT */
    switch (m) {
    case 0x0180: K.dma_notifies = v; break;
    case 0x0184: K.dma_a = v; nv2a_raster_set_dma(K.dma_a, K.dma_b); break;
    case 0x0188: K.dma_b = v; nv2a_raster_set_dma(K.dma_a, K.dma_b); break;
    case 0x0194: K.dma_color = v; update_surface(); break;
    case 0x0198: K.dma_zeta = v; update_surface(); break;
    case 0x01A4: K.dma_semaphore = v; break;
    case 0x01A8: K.dma_report = v; break;
    case 0x17C8:                                              /* CLEAR_REPORT_VALUE */
        if (g_gpu_backend && g_gpu_backend->report && !gpu_threaded()) g_gpu_backend->report(1, v);
        break;
    case 0x17CC:                                              /* SET_ZPASS_PIXEL_COUNT_ENABLE */
        K.zpass_enable = v;
        if (g_gpu_backend && g_gpu_backend->report && !gpu_threaded()) g_gpu_backend->report(0, v != 0);
        break;
    case 0x17D0: {                                            /* GET_REPORT: type << 24 | offset */
        /* 16-byte report: 64-bit timestamp, result, status (0 = done; D3D
         * pre-fills -1 and polls it). With a backend that counts pixels, it
         * fills in the result and status when the GPU has finished; otherwise
         * the report says "fully visible" at once. */
        uint32_t lim = 0, b = dma_base(K.dma_report, &lim), off = v & 0xFFFFFF;
        if (lim && off + 15 <= lim) {
            uint32_t dst = phys_to_guest(b + off);
            wr32(dst, 0x44556677); wr32(dst + 4, 0x00112233);
            if ((v >> 24) == 1 && g_gpu_backend && g_gpu_backend->report && !gpu_threaded()) g_gpu_backend->report(2, dst);
            else { wr32(dst + 8, (v >> 24) == 1 ? 0x1000 : 0); wr32(dst + 12, 0); }
        }
        if (K.reports++ < 4) runtime_log("[nv2a] report %08x -> dma %x base %08x (limit %08x)", v, K.dma_report, b, lim);
        break;
    }
    case 0x0200: K.clip_h = v; update_surface(); break;
    case 0x0204: K.clip_v = v; update_surface(); break;
    case 0x0208: K.surface_format = v; update_surface(); break;
    case 0x020C: K.surface_pitch = v; update_surface(); break;
    case 0x0210: K.color_offset = v; update_surface(); break;
    case 0x0214: K.zeta_offset = v; update_surface(); break;
    case 0x17FC:                                              /* SET_BEGIN_END */
        if (v) K.draws++;
        K.begin_end = v;
        break;
    case 0x1D6C: K.semaphore_offset = v; break;
    case 0x1D70: {                                            /* BACK_END_WRITE_SEMAPHORE_RELEASE */
        uint32_t lim = 0, b = dma_base(K.dma_semaphore, &lim);
        uint32_t dst = phys_to_guest(b + K.semaphore_offset);
        if (lim && K.semaphore_offset <= lim) wr32(dst, v);
        if (g_trace > 1 || (g_trace && K.methods < 200000))
            { static int n; if (n++ < 20) runtime_log("[nv2a] semaphore release %08x -> dma %x base %08x+%08x (limit %08x) = %08x",
                                                     v, K.dma_semaphore, b, K.semaphore_offset, lim, dst); }
        break;
    }
    /* clear values are also visible as PGRAPH registers; D3D uses the
     * z/stencil one as the parameter slot for its NOP software methods
     * (e.g. the surface address of a flip) */
    case 0x1D8C: K.clear_zstencil = v; REG(0x401A88) = v; break;
    case 0x1D90: K.clear_color = v; REG(0x401A8C) = v; break;
    case 0x1D94: do_clear(v); break;
    case 0x1D98: g_clear_rect_h = v; break;
    case 0x1D9C: g_clear_rect_v = v; break;
    default: break;
    }
}

/* FLIP_STALL: the GPU may not start drawing into the buffer that is still on
 * screen. It waits while READ_3D == WRITE_3D; D3D's vblank handler advances
 * READ_3D (PGRAPH_INCREMENT) once the flip has reached the display. */
static int flip_stall_blocked(void) {
    static int off = -1;
    if (off < 0) off = getenv("HALO_NO_FLIP_STALL") != NULL;
    if (off) return 0;
    uint32_t sf = REG(PGRAPH_SURFACE);
    if (((sf >> 24) & 7) != ((sf >> 20) & 7)) return 0;
    if (g_frames - g_stall_since > 6) { g_stall_timeouts++; return 0; }   /* never hang the GPU */
    return 1;
}

/* Consume the pushbuffer from GET to PUT (physical addresses). The decoder
 * state is kept across calls so a stall can resume in the middle of a burst. */
static int g_in_fifo;
/* diagnosis: the last pushbuffer control events, dumped when the decoder hits garbage */
static struct { char k; uint32_t a, b; } g_ev[64];
static volatile uint32_t g_evn;
static void ev(char k, uint32_t a, uint32_t b) { uint32_t i = __atomic_fetch_add(&g_evn, 1, __ATOMIC_RELAXED) & 63; g_ev[i].k = k; g_ev[i].a = a; g_ev[i].b = b; }
static uint32_t f_ret_addr, f_method, f_subch, f_count, f_nonincreasing;
static void run_fifo(void) {
    if (g_in_fifo) return;
    if (g_stalled) {
        if (flip_stall_blocked()) return;
        g_stalled = 0;
        g_stall_frames += g_frames - g_stall_since;
    }
    g_in_fifo = 1;
    /* acquire: the game wrote the pushbuffer before it stored PUT (release) */
    uint32_t get = REG(USER_DMA_GET), put = __atomic_load_n(&REG(USER_DMA_PUT), __ATOMIC_ACQUIRE);
    ev('B', get, put);
    int guard = 0;
    while (get != put && guard++ < 4000000) {
        if ((guard & 63) == 0 && g_vblank_waiting && !t_gpu_thread) {
            /* let the ticker deliver a due vblank; other kicks see g_in_fifo and return */
            REG(USER_DMA_GET) = get;
            guest_yield_lock();
            put = __atomic_load_n(&REG(USER_DMA_PUT), __ATOMIC_ACQUIRE);
        }
        uint32_t w = rd32(phys_to_guest(get));
        get += 4;
        if (f_count) {
            kelvin_method(f_subch, f_method, w);
            if (!f_nonincreasing) f_method += 4;
            f_count--;
            if (g_stall_req) {
                g_stall_req = 0;
                g_stall_since = g_frames;
                if (flip_stall_blocked()) { g_stalled = 1; g_stall_count++; break; }
            }
            continue;
        }
        if ((w & 0xE0000003) == 0x20000000) { ev('j', get - 4, w); get = w & 0x1FFFFFFC; continue; }   /* old jump */
        if ((w & 3) == 1) { ev('J', get - 4, w); get = w & ~3u; continue; }                            /* jump */
        if ((w & 3) == 2) { ev('C', get - 4, w); f_ret_addr = get; get = w & ~3u; continue; }          /* call */
        if (w == 0x00020000) { ev('R', get - 4, f_ret_addr); get = f_ret_addr; continue; }              /* return */
        if ((w & 0xE0030003) == 0 || (w & 0xE0030003) == 0x40000000) {
            f_nonincreasing = (w & 0xE0030003) == 0x40000000;
            f_method = w & 0x1FFC;
            f_subch = (w >> 13) & 7;
            f_count = (w >> 18) & 0x7FF;
            continue;
        }
        runtime_log("[nv2a] unknown pushbuffer word %08x at %08x (put %08x) - stopping", w, get - 4, put);
        {
            uint32_t n = g_evn;
            for (uint32_t i = n > 64 ? n - 64 : 0; i < n; i++)
                runtime_log("[nv2a]   event %c %08x %08x", g_ev[i & 63].k, g_ev[i & 63].a, g_ev[i & 63].b);
        }
        f_count = 0;
        get = put;
        break;
    }
    __atomic_store_n(&REG(USER_DMA_GET), get, __ATOMIC_RELEASE);   /* the game polls GET for progress */
    g_in_fifo = 0;
}

#include <pthread.h>
static pthread_mutex_t g_fifo_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_fifo_cv = PTHREAD_COND_INITIALIZER;
static volatile int g_fifo_kick, g_flush_req, g_gpu_threaded = -1;

static void *gpu_thread_main(void *arg) {
    (void)arg;
    t_gpu_thread = 1;
    for (;;) {
        pthread_mutex_lock(&g_fifo_mx);
        if (!g_fifo_kick && !g_flush_req) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 1000000;                          /* 1 ms: a flip stall can clear without a kick */
            if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
            pthread_cond_timedwait(&g_fifo_cv, &g_fifo_mx, &ts);
        }
        g_fifo_kick = 0;
        pthread_mutex_unlock(&g_fifo_mx);
        g_gpu_state = 1;
        if (REG(USER_DMA_PUT) != REG(USER_DMA_GET)) run_fifo();
        g_gpu_state = 0;
        if (g_flush_req) {                                  /* vertical blank: submit what is queued */
            g_flush_req = 0;
            if (g_gpu_backend) g_gpu_backend->flush();
        }
    }
    return NULL;
}

static int gpu_threaded(void) {
    if (g_gpu_threaded < 0) {
        /* Off by default: Halo's D3D tracks how far the GPU has read the
         * pushbuffer in a way this model does not reproduce yet, and with an
         * asynchronous reader it overwrote commands before they were read.
         * HALO_GPU_THREAD=1 turns it on (experimental). */
        const char *e = getenv("HALO_GPU_THREAD");
        g_gpu_threaded = e && *e == '1';
        if (g_gpu_threaded) {
            pthread_t t;
            pthread_create(&t, NULL, gpu_thread_main, NULL);
            runtime_log("[nv2a] pushbuffer processed on its own thread");
        }
    }
    return g_gpu_threaded;
}

static void fifo_kick(void) {
    if (!gpu_threaded()) { run_fifo(); return; }
    pthread_mutex_lock(&g_fifo_mx);
    g_fifo_kick = 1;
    pthread_cond_signal(&g_fifo_cv);
    pthread_mutex_unlock(&g_fifo_mx);
}

/* ------------------------------------------------------------- MMIO */
/* Spin detector: a register read a million times is almost always a busy-wait
 * on hardware we don't model yet; say so once. */
static void note_hot_read(uint32_t a, uint32_t v) {
    static uint32_t addr[256], cnt[256];
    uint32_t h = (a >> 2) & 255;
    if (addr[h] != a) { addr[h] = a; cnt[h] = 0; }
    if (++cnt[h] == 1000000) {
        CPU *c = current_cpu();
        runtime_log("[mmio] hot register %08x (value %08x): guest is polling unmodelled hardware", a, v);
        if (c) runtime_log("       guest stack: %08x %08x %08x %08x %08x %08x", rd32(c->esp), rd32(c->esp + 4),
                           rd32(c->esp + 8), rd32(c->esp + 12), rd32(c->esp + 16), rd32(c->esp + 20));
    }
}

static uint32_t g_rd_off[512], g_rd_cnt[512];
static void mmio_stats(uint32_t off) {
    static int on = -1;
    if (on < 0) on = getenv("HALO_MMIO_STATS") != NULL;
    if (!on) return;
    uint32_t h = (off >> 2) & 511;
    for (int i = 0; i < 8; i++, h = (h + 1) & 511)
        if (g_rd_off[h] == off || !g_rd_cnt[h]) { g_rd_off[h] = off; g_rd_cnt[h]++; break; }
    static uint64_t next;
    uint64_t now = host_time_ns();
    if (!next) next = now + 20000000000ull;
    if (now > next) {
        next = now + 20000000000ull;
        for (int i = 0; i < 512; i++) if (g_rd_cnt[i] > 100) runtime_log("[mmio] read %06x x%u", g_rd_off[i], g_rd_cnt[i]);
    }
}

uint32_t mmio_read(uint32_t a, int size) {
    uint32_t off = a - MMIO_BASE, v;
    mmio_stats(off);
    if (apu_owns(a)) { v = 0; memcpy(&v, g_mem + a, (size_t)size); return apu_read(a, size, v); }
    switch (off & ~3u) {
    case PMC_BOOT_0: v = 0x02A000A1; break;
    case PMC_INTR_0:
        v = ((REG(PCRTC_INTR_0) & REG(PCRTC_INTR_EN_0) & 1) ? PMC_INTR_PCRTC : 0) |
            ((REG(PGRAPH_INTR) & REG(PGRAPH_INTR_EN)) ? PMC_INTR_PGRAPH : 0);
        break;
    case PTIMER_TIME_0: v = (uint32_t)(host_time_ns() - g_boot_ns) & ~0x1Fu; break;
    case PTIMER_TIME_1: v = (uint32_t)((host_time_ns() - g_boot_ns) >> 32) & 0x1FFFFFFF; break;
    case PFB_CSTATUS: v = 0x04000000; break;                       /* 64 MB */
    case 0x002400: v = 0x10; break;                                /* PFIFO_RUNOUT_STATUS: empty */
    case 0x003214: v = 0x10; break;                                /* CACHE1_STATUS: empty */
    case 0x003220: v = REG(0x003220) & ~0x1000u; break;           /* CACHE1_DMA_PUSH: not suspended/busy */
    case PFB_WBC: v = REG(PFB_WBC) & ~0x00010000u; break;          /* flush completes immediately */
    case PGRAPH_STATUS: v = 0; break;                              /* idle */
    case PGRAPH_INTR: v = REG(PGRAPH_INTR); break;
    case PCRTC_RASTER: v = (uint32_t)(((host_time_ns() - g_boot_ns) / 31777) % 525); break;  /* scanline */
    case USER_DMA_GET: if (!gpu_threaded()) run_fifo(); v = __atomic_load_n(&REG(USER_DMA_GET), __ATOMIC_ACQUIRE); break;
    /* --- MCPX APU (audio), 0xFE800000: minimal model, commands are accepted and dropped --- */
    case 0x1820010: v = 0x40; break;                               /* GP command FIFO: free slots */
    default: v = 0; memcpy(&v, g_mem + a, (size_t)size); note_hot_read(a, v); return v;
    }
    if (size < 4) v >>= (a & 3) * 8;
    return size == 1 ? (v & 0xFF) : size == 2 ? (v & 0xFFFF) : v;
}

void mmio_write(uint32_t a, uint32_t v, int size) {
    uint32_t off = a - MMIO_BASE;
    if (apu_owns(a)) {
        if (!apu_write_pre(a, v, size)) memcpy(g_mem + a, &v, (size_t)size);
        apu_write(a, v, size);
        return;
    }
    if (g_trace > 1 && ((off >= 0x2000 && off < 0x4000) || (off >= USER_BASE && off < USER_BASE + 0x100)))
        runtime_log("[nv2a] write %06x = %08x", off, v);
    if (size == 4) switch (off) {
    case PCRTC_INTR_0: REG(PCRTC_INTR_0) &= ~v; return;           /* write-1-to-clear */
    case PMC_INTR_0: return;
    case PFIFO_INTR_0: REG(PFIFO_INTR_0) &= ~v; return;           /* write-1-to-clear */
    case PGRAPH_INTR: REG(PGRAPH_INTR) &= ~v; return;
    case PGRAPH_INCREMENT:
        if (g_trace > 1) runtime_log("[nv2a] PGRAPH_INCREMENT %x", v);
        if (v & 2) {                                              /* READ_3D */
            uint32_t sf = REG(PGRAPH_SURFACE), rd = (sf >> 24) & 7, mod = (sf >> 28) & 7;
            rd = mod ? (rd + 1) % mod : 0;
            REG(PGRAPH_SURFACE) = (sf & ~0x07000000u) | (rd << 24);
        }
        return;
    case USER_DMA_PUT: ev('P', v, REG(USER_DMA_GET)); __atomic_store_n(&REG(USER_DMA_PUT), v, __ATOMIC_RELEASE); fifo_kick(); return;
    case USER_DMA_GET: ev('G', v, REG(USER_DMA_GET)); break;
    case PCRTC_START:
        if (g_trace) { static int n; if (n++ < 10) runtime_log("[nv2a] display start -> %08x", v); }
        REG(PCRTC_START) = v;
        return;
    default: break;
    }
    memcpy(g_mem + a, &v, (size_t)size);
}

/* ------------------------------------------------------------- vblank */
/* Frame pacing diagnosis, logged every 5 s as [pace]: flip-to-flip times,
 * late or dropped vblanks, and the longest wait of the ticker for the guest
 * lock (it delivers the vblank interrupt). */
extern uint64_t g_pace_gil_max;
extern unsigned g_pace_gil_over8;
static uint64_t g_pace_last_flip, g_pace_flip_max, g_pace_vbl_late_max;
static unsigned g_pace_hist[5], g_pace_vbl_dropped, g_pace_vbl_late4, g_pace_vbl_host, g_pace_vbl_timer;
static void pace_flip(void) {
    uint64_t now = host_time_ns();
    if (g_pace_last_flip) {
        uint64_t d = now - g_pace_last_flip;
        if (d > g_pace_flip_max) g_pace_flip_max = d;
        unsigned ms = (unsigned)(d / 1000000ull);
        g_pace_hist[ms < 28 ? 0 : ms < 42 ? 1 : ms < 58 ? 2 : ms < 75 ? 3 : 4]++;
    }
    g_pace_last_flip = now;
}
static void pace_report(void) {
    runtime_log("[pace] flips <28ms %u, 28-42 %u, 42-58 %u, 58-75 %u, >75 %u (max %.1f ms) | vblanks from display %u, timer %u; late>4ms %u, max %.1f ms, dropped %u | ticker lock wait max %.1f ms, >8ms %u",
                g_pace_hist[0], g_pace_hist[1], g_pace_hist[2], g_pace_hist[3], g_pace_hist[4], g_pace_flip_max / 1e6,
                g_pace_vbl_host, g_pace_vbl_timer, g_pace_vbl_late4, g_pace_vbl_late_max / 1e6, g_pace_vbl_dropped, g_pace_gil_max / 1e6, g_pace_gil_over8);
    g_pace_vbl_host = g_pace_vbl_timer = 0;
    memset(g_pace_hist, 0, sizeof g_pace_hist);
    g_pace_flip_max = g_pace_vbl_late_max = 0; g_pace_vbl_dropped = g_pace_vbl_late4 = 0;
    g_pace_gil_max = 0; g_pace_gil_over8 = 0;
}

void nv2a_tick(uint64_t now) {
    if (!g_boot_ns) {
        g_boot_ns = now;
        g_next_vblank_ns = now + 16666667ull;
        g_trace = getenv("HALO_TRACE_GPU") ? atoi(getenv("HALO_TRACE_GPU")) : 0;
        if (getenv("HALO_TRACE_GPU") && !g_trace) g_trace = 1;
        return;
    }
    nv2a_vblank(now);           /* first: it must not wait behind the pushbuffer */
    if (REG(USER_DMA_PUT) != REG(USER_DMA_GET)) fifo_kick();
    apu_tick(now);
    static int gpu_state = -1;
    if (gpu_state < 0) gpu_state = getenv("HALO_GPU_STATE") != NULL;
    if (gpu_state) {
        static uint64_t next;
        if (now > next) {
            next = now + 5000000000ull;
            runtime_log("[gpu] state %d, get %08x put %08x, stalled %d, surface %08x, method %04x count %u",
                        g_gpu_state, REG(USER_DMA_GET), REG(USER_DMA_PUT), g_stalled, REG(PGRAPH_SURFACE), f_method, f_count);
        }
    }
}

uint64_t nv2a_next_vblank_ns(void) { return g_next_vblank_ns ? g_next_vblank_ns : UINT64_MAX; }

/* Emulated vblanks follow the host display's refresh when it reports one
 * (main.m calls nv2a_host_vblank from the view's draw callback), so each
 * game frame stays on screen for the same number of refreshes. With a free
 * running 60 Hz timer the two clocks drift against each other and motion
 * judders periodically. On 120 Hz displays every other refresh counts. With
 * no host refreshes (headless, window hidden) the timer takes over. */
static uint32_t g_host_vbl_seq, g_host_vbl_seen;
static uint64_t g_host_vbl_last_ns, g_last_vbl_ns;
void nv2a_host_vblank(void) {
    __atomic_add_fetch(&g_host_vbl_seq, 1, __ATOMIC_RELEASE);
}
static void nv2a_vblank(uint64_t now) {
    if (!g_boot_ns) return;
    uint32_t hs = __atomic_load_n(&g_host_vbl_seq, __ATOMIC_ACQUIRE);
    int host_tick = hs != g_host_vbl_seen;
    if (host_tick) { g_host_vbl_seen = hs; g_host_vbl_last_ns = now; }
    int host_mode = g_host_vbl_last_ns && now - g_host_vbl_last_ns < 100000000ull;
    int fire;
    if (host_mode) {
        uint64_t since = now - g_last_vbl_ns;
        fire = (host_tick && since >= 12000000ull) || since >= 18000000ull;   /* fallback if a refresh is late */
        if (fire) { if (host_tick) g_pace_vbl_host++; else g_pace_vbl_timer++; }
        g_next_vblank_ns = (fire ? now : g_last_vbl_ns) + 16666667ull;           /* hint for the ticker */
    } else {
        fire = now >= g_next_vblank_ns;
        if (fire) {
            uint64_t late = now - g_next_vblank_ns;
            if (late > g_pace_vbl_late_max) g_pace_vbl_late_max = late;
            if (late > 4000000ull) g_pace_vbl_late4++;
            g_pace_vbl_timer++;
            g_next_vblank_ns += 16666667ull;
            if (now > g_next_vblank_ns) { g_pace_vbl_dropped += (unsigned)((now - g_next_vblank_ns) / 16666667ull) + 1; g_next_vblank_ns = now + 16666667ull; }
        }
    }
    if (!fire) return;
    if (host_mode && g_last_vbl_ns) {
        uint64_t gap = now - g_last_vbl_ns;
        if (gap > 20000000ull) {
            uint64_t late = gap - 16666667ull;
            if (late > g_pace_vbl_late_max) g_pace_vbl_late_max = late;
            if (late > 4000000ull) g_pace_vbl_late4++;
        }
    }
    g_last_vbl_ns = now;
    g_frames++;
    if (g_frames % 300 == 0) pace_report();

    /* scan out the visible framebuffer */
    if (g_gpu_backend) {
        if (gpu_threaded()) { g_flush_req = 1; fifo_kick(); }   /* the backend belongs to the GPU thread */
        else g_gpu_backend->flush();
    }
    uint32_t fb = phys_to_guest(REG(PCRTC_START));
    if (fb) gfx_present_frame((const uint32_t *)(g_mem + fb), g_disp_w, g_disp_h, g_disp_pitch);
    {   /* HALO_DUMP_FRAMES=n1,n2,...: write the displayed frame at those vblanks to frame_<n>.ppm (debugging) */
        static const char *spec = (const char *)1;
        if (spec == (const char *)1) spec = getenv("HALO_DUMP_FRAMES");
        if (spec && fb) {
            const char *p = spec;
            while (*p) {
                unsigned long n = strtoul(p, (char **)&p, 10);
                if (n == g_frames) {
                    char name[64];
                    snprintf(name, sizeof name, "frame_%lu.ppm", n);
                    FILE *f = fopen(name, "wb");
                    if (f) {
                        fprintf(f, "P6\n%u %u\n255\n", g_disp_w, g_disp_h);
                        for (uint32_t y = 0; y < g_disp_h; y++)
                            for (uint32_t x = 0; x < g_disp_w; x++) {
                                uint32_t c = *(uint32_t *)(g_mem + fb + y * g_disp_pitch + x * 4);
                                unsigned char rgb[3] = {(unsigned char)(c >> 16), (unsigned char)(c >> 8), (unsigned char)c};
                                fwrite(rgb, 1, 3, f);
                            }
                        fclose(f);
                        runtime_log("[nv2a] wrote %s", name);
                    }
                }
                if (*p == ',') p++; else break;
            }
        }
    }
    {   /* tearing check: was the GPU drawing into the buffer on screen during this frame? */
        static uint64_t last_draws;
        uint32_t lim = 0, cs = phys_to_guest(dma_base(K.dma_color, &lim) + K.color_offset);
        if (fb && cs == fb && K.draws != last_draws) g_scanout_conflicts++;
        last_draws = K.draws;
    }

    /* vertical blank interrupt */
    REG(PCRTC_INTR_0) |= 1;
    static uint32_t g_irqs, g_noirq;
    if ((REG(PCRTC_INTR_EN_0) & 1) && (REG(PMC_INTR_EN_0) & 1) && g_gpu_kinterrupt) {
        kernel_raise_interrupt(g_gpu_kinterrupt);
        g_irqs++;
    } else g_noirq++;
    if (g_trace && g_frames % 120 == 0)
        runtime_log("[nv2a] vblank irq delivered %u, masked %u (pcrtc_en %x pmc_en %x kint %08x) surface %08x",
                    g_irqs, g_noirq, REG(PCRTC_INTR_EN_0), REG(PMC_INTR_EN_0), g_gpu_kinterrupt, REG(PGRAPH_SURFACE));

    if (g_trace >= 3 && g_frames % 600 == 0) dump_method_histogram();
    if (g_trace && g_frames % 300 == 0) {
        nv2a_raster_debug_dump();
        if (fb) {   /* coverage of the visible frame: share of pixels that are not black */
            uint64_t lit = 0;
            for (uint32_t y = 0; y < g_disp_h; y++)
                for (uint32_t x = 0; x < g_disp_w; x++)
                    if (*(uint32_t *)(g_mem + fb + y * g_disp_pitch + x * 4) & 0x00F0F0F0u) lit++;
            runtime_log("[nv2a] visible frame: %.1f%% of pixels lit", 100.0 * lit / ((double)g_disp_w * g_disp_h));
        }
        runtime_log("[nv2a] flip stalls %llu (%llu vblanks waited, %llu timeouts), frames drawn into the visible buffer %llu",
                    (unsigned long long)g_stall_count, (unsigned long long)g_stall_frames,
                    (unsigned long long)g_stall_timeouts, (unsigned long long)g_scanout_conflicts);
    }
    if (g_trace && g_frames % 300 == 0)
        { uint64_t tris, px; nv2a_raster_stats(&tris, &px);
          runtime_log("[nv2a] %u vblanks, %llu methods, %llu draws, %llu triangles, %llu pixels, %u flips, fb %08x", g_frames,
                    (unsigned long long)K.methods, (unsigned long long)K.draws, (unsigned long long)tris,
                    (unsigned long long)px, g_flips_requested, fb); }
}

uint32_t nv2a_stat_flips(void) { return __atomic_load_n(&g_flips_requested, __ATOMIC_RELAXED); }
uint32_t nv2a_stat_vblanks(void) { return __atomic_load_n(&g_frames, __ATOMIC_RELAXED); }
