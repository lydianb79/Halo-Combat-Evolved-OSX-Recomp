/* gpdsp.c — the MCPX APU's Global Processor (GP), a Motorola DSP56300 that
 * runs DirectSound's effects image (I3DL2 reverb and the other effects).
 *
 * The core itself is the dsp56300 JIT emulator by Matt Borgerson (MIT
 * licence, https://github.com/mborgerson/dsp56300); this file is the Xbox
 * side around it:
 *
 *   - GP window in the APU MMIO (0xFE830000): X memory 0x0000 (4K words),
 *     mix buffer 0x5000 (1K words), Y memory 0x6000 (2K words),
 *     P memory 0xA000 (4K words), reset register 0xFFFC (bit 0 GPRST,
 *     bit 1 GPDSPRST).
 *   - Releasing both reset bits runs the boot ROM's job: the first 0x800
 *     words of GP scratch memory are copied into P memory.
 *   - The mix buffer (32 bins x 32 samples) appears twice in X space: at
 *     X:$0C00-$0FFF (so X RAM proper is 3K words) and at X:$1400-$17FF.
 *     Effects read and write bins through either address.
 *   - Each 32-sample frame the voice processor's 32 mix bins are written to
 *     the mix buffer (24-bit), the "start of frame" bit is raised in the
 *     interrupt register X:$FFFFC5, and the DSP runs until its program writes
 *     bit 0 of X:$FFFFC4 (done for this frame). The stereo result is then in
 *     bins 0 and 1 (X:$1400 / X:$1420).
 *   - DMA controller (X:$FFFFD4 next block, D5 start block, D6 control,
 *     D7 configuration): a linked list of 7-word descriptors in DSP memory
 *     moves data between DSP memory and system memory: GP scratch (linear or
 *     circular, through the scratch scatter-gather list) or the GP FIFOs
 *     (circular buffers through the FIFO scatter-gather list).
 *
 * These hardware facts follow xemu's MCPX APU model.
 *
 * Environment: HALO_DSP=0 turns the GP off (built-in reverb instead);
 * HALO_TRACE_DSP=1 logs resets, DMA transfers and cycles per frame.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#include "gpdsp.h"
#include "runtime.h"
#include "cpu.h"

#ifndef HALO_DSP

int gpdsp_compiled(void) { return 0; }
int gpdsp_enabled(void) { return 0; }
int gpdsp_running(void) { return 0; }
void gpdsp_mmio_write(int ep, uint32_t off, uint32_t v) { (void)ep; (void)off; (void)v; }
int gpdsp_mmio_read(int ep, uint32_t off, uint32_t *v) { (void)ep; (void)off; (void)v; return 0; }
void gpdsp_submit(float bins[GP_BINS][GP_FRAME], void (*sink)(const float *, int)) { (void)bins; (void)sink; }

#else

#include <dsp56300.h>

#define APU_REG(off) (*(uint32_t *)(g_mem + 0xFE800000u + (off)))
static uint64_t g_run_ns, g_cb_ns, g_cb_n;
static uint64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

#define W_X     0x0000
#define W_MIX   0x5000
#define W_Y     0x6000
#define W_P     0xA000
#define W_RST   0xFFFC
#define XRAM_N  0xC00                  /* X RAM below the mix buffer alias */
#define MIX_N   0x400
#define MIX_X0  0xC00                  /* the mix buffer's first DSP address (alias) */
#define YRAM_N  0x800
#define PRAM_N  0x1000
#define MIX_X   0x1400                 /* the mix buffer's DSP address */

#define INT_START_FRAME (1u << 1)
#define INT_DMA_EOL     (1u << 7)

#define DMA_RUNNING (1u << 4)
#define DMA_STOPPED (1u << 5)
#define DMA_FROZEN  (1u << 3)
#define NODE_EOL    (1u << 14)

#define EP_OUT_BYTES 1024              /* one EP frame: 256 stereo 16-bit samples */

/* One DSP: the GP (effects) or the EP (encoder / final stereo mix). */
typedef struct Dsp {
    const char *name;
    int is_gp;
    uint32_t saddr, faddr, ofbase, ifbase;     /* APU registers: scratch SG list, FIFO SG list, FIFO regs */
    uint32_t x[XRAM_N], mix[MIX_N], y[YRAM_N], p[PRAM_N];
    Dsp56300Jit *jit;
    uint32_t rst;
    int running;
    uint32_t ints;                             /* X:$FFFFC5 */
    uint32_t periph[128];
    struct { uint32_t config, control, start, next, reads; int eol; } dma;
    uint64_t frames, cycles_sum, cycles_max, unfinished;
    uint8_t *buf; uint32_t buf_n;              /* DMA staging */
    int traced;                                /* DMA lines logged */
} Dsp;

static Dsp g_gp = {.name = "GP", .is_gp = 1, .saddr = 0x2040, .faddr = 0x2044, .ofbase = 0x3024, .ifbase = 0x3064};
static Dsp g_ep = {.name = "EP", .is_gp = 0, .saddr = 0x2048, .faddr = 0x204C, .ofbase = 0x4024, .ifbase = 0x4064};
static int g_enabled = -1, g_trace, g_use_ep;
static uint8_t g_ep_out[EP_OUT_BYTES];         /* what the EP sent to its output FIFO 0 this frame */
static uint32_t g_ep_out_n;
static uint32_t g_frame_div;

/* ---------------------------------------------------------------- system memory */
static uint8_t *ram(uint32_t phys_addr) { return g_mem + (GUEST_RAM_MIRROR1 | (phys_addr & (GUEST_RAM_SIZE - 1))); }

/* Copy len bytes between buf and the byte range [addr, addr+len) of a
 * scatter-gather space (a list of 8-byte entries, one per 4 KB page). */
static void sg_copy(uint32_t list, uint8_t *buf, uint32_t addr, uint32_t len, int to_mem) {
    if (!list) { if (!to_mem) memset(buf, 0, len); return; }
    while (len) {
        uint32_t page = *(uint32_t *)ram(list + (addr >> 12) * 8) & ~0xFFFu, in = addr & 0xFFF;
        uint32_t n = 4096 - in;
        if (n > len) n = len;
        if (page) {
            if (to_mem) memcpy(ram(page + in), buf, n);
            else memcpy(buf, ram(page + in), n);
        } else if (!to_mem) memset(buf, 0, n);
        buf += n; addr += n; len -= n;
    }
}

/* Circular copy inside [base, end) starting at *cur. */
static void sg_ring(uint32_t list, uint8_t *buf, uint32_t base, uint32_t end, uint32_t *cur, uint32_t len, int to_mem) {
    if (end <= base) { if (!to_mem) memset(buf, 0, len); return; }
    if (*cur >= end || *cur < base) *cur = base;
    while (len) {
        uint32_t n = end - *cur;
        if (n > len) n = len;
        sg_copy(list, buf, *cur, n, to_mem);
        buf += n; len -= n; *cur += n;
        if (*cur >= end) *cur = base;
    }
}

static void fifo_copy(Dsp *d, int index, int out, uint8_t *buf, uint32_t len) {
    if (!d->is_gp && out && index == 0) {                /* the EP's final output: keep a copy for the host */
        uint32_t n = len;
        if (g_ep_out_n + n > EP_OUT_BYTES) n = EP_OUT_BYTES - g_ep_out_n;
        memcpy(g_ep_out + g_ep_out_n, buf, n);
        g_ep_out_n += n;
    }
    uint32_t r = (out ? d->ofbase : d->ifbase) + 0x10u * (uint32_t)index;
    uint32_t base = APU_REG(r) & 0xFFFFFF, end = APU_REG(r + 4) & 0xFFFFFF, cur = APU_REG(r + 8) & 0xFFFFFF;
    sg_ring(APU_REG(d->faddr), buf, base, end, &cur, len, out);
    APU_REG(r + 8) = (APU_REG(r + 8) & ~0xFFFFFFu) | cur;
}

/* ---------------------------------------------------------------- DSP memory (for DMA) */
/* DMA addresses: X below $1800, Y at $1800-$1FFF, P at $2800-$37FF. */
static int dma_space(uint32_t a, Dsp56300MemSpace *sp, uint32_t *addr) {
    if (a < 0x1800) { *sp = DSP56300_MEM_SPACE_X; *addr = a; return 1; }
    if (a >= 0x1800 && a < 0x2000) { *sp = DSP56300_MEM_SPACE_Y; *addr = a - 0x1800; return 1; }
    if (a >= 0x2800 && a < 0x3800) { *sp = DSP56300_MEM_SPACE_P; *addr = a - 0x2800; return 1; }
    return 0;
}
static uint32_t *xy_word(Dsp *d, Dsp56300MemSpace sp, uint32_t a) {
    if (sp == DSP56300_MEM_SPACE_X) {
        if (a < XRAM_N) return &d->x[a];
        if (a >= MIX_X0 && a < MIX_X0 + MIX_N) return &d->mix[a - MIX_X0];
        if (a >= MIX_X && a < MIX_X + MIX_N) return &d->mix[a - MIX_X];
        return NULL;
    }
    if (sp == DSP56300_MEM_SPACE_Y) return a < YRAM_N ? &d->y[a] : NULL;
    return NULL;
}
static uint32_t mem_rd(Dsp *d, Dsp56300MemSpace sp, uint32_t a) {
    if (sp == DSP56300_MEM_SPACE_P) return a < PRAM_N ? d->p[a] : 0;
    uint32_t *w = xy_word(d, sp, a);
    return w ? *w : 0;
}
static void mem_wr(Dsp *d, Dsp56300MemSpace sp, uint32_t a, uint32_t v) {
    v &= 0xFFFFFF;
    if (sp == DSP56300_MEM_SPACE_P) {                 /* through the library: it tracks code changes */
        if (a < PRAM_N && d->p[a] != v) dsp56300_write_memory(d->jit, sp, a, v);
        return;
    }
    uint32_t *w = xy_word(d, sp, a);
    if (w) *w = v;
}

/* ---------------------------------------------------------------- DMA */
static void dma_run(Dsp *d) {
    if (!(d->dma.control & DMA_RUNNING) || (d->dma.control & DMA_FROZEN)) return;
    int guard = 0;
    while (!(d->dma.next & NODE_EOL) && guard++ < 4096) {
        Dsp56300MemSpace bsp; uint32_t ba;
        if (!dma_space(d->dma.next & 0x3FFF, &bsp, &ba)) { runtime_log("[dsp] %s DMA descriptor at bad address %06x", d->name, d->dma.next); break; }
        uint32_t next = mem_rd(d, bsp, ba), control = mem_rd(d, bsp, ba + 1), count = mem_rd(d, bsp, ba + 2);
        uint32_t dsp_off = mem_rd(d, bsp, ba + 3), soff = mem_rd(d, bsp, ba + 4), sbase = mem_rd(d, bsp, ba + 5);
        uint32_t ssize = mem_rd(d, bsp, ba + 6) + 1;
        d->dma.next = next;
        if (next & NODE_EOL) d->dma.eol = 1;

        int interleave = control & 1, to_mem = (control >> 1) & 1, writeback = (control >> 4) & 1;
        uint32_t buf_id = (control >> 5) & 0xF, format = (control >> 10) & 7;
        uint32_t item = format == 0 ? 1 : format == 1 ? 2 : 4;
        uint32_t channels = (count & 0xF) + 1, blocks = count >> 4;
        Dsp56300MemSpace sp; uint32_t ma;
        if (!dma_space(dsp_off, &sp, &ma)) { runtime_log("[dsp] %s DMA to bad DSP address %06x", d->name, dsp_off); continue; }
        uint32_t words = to_mem && interleave ? blocks * channels : count, bytes = words * item;
        if (bytes > d->buf_n) { d->buf_n = bytes + 4096; d->buf = realloc(d->buf, d->buf_n); }
        uint8_t *buf = d->buf;
        if (g_trace && (d->traced++ < 200 || (words != 6 && words != 16 && words != 32 && words != 640)))
            runtime_log("[dsp] %s DMA %s %s%s %u words (fmt %u%s) dsp %c:%04x scratch %06x+%06x/%06x", d->name,
                        to_mem ? "dsp->" : "->dsp", buf_id <= 3 ? "fifo" : buf_id == 0xE ? "ring" : buf_id == 0xF ? "scratch" : "?",
                        buf_id <= 3 ? (const char[]){(char)('0' + buf_id), 0} : "", words, format,
                        interleave ? " interleaved" : "", "XYP"[sp], ma, sbase, soff, ssize);
        if (to_mem) {
            for (uint32_t i = 0; i < words; i++) {
                uint32_t src = interleave ? ma + (i % channels) * blocks + i / channels : ma + i;
                uint32_t v = mem_rd(d, sp, src);
                if (item == 1) buf[i] = (uint8_t)(v >> 16);
                else if (item == 2) { uint16_t h = (uint16_t)(v >> 8); memcpy(buf + i * 2, &h, 2); }
                else memcpy(buf + i * 4, &v, 4);
            }
            if (buf_id <= 3) fifo_copy(d, (int)buf_id, 1, buf, bytes);
            else if (buf_id == 0xE) { uint32_t cur = sbase + (soff < ssize ? soff : 0); sg_ring(APU_REG(d->saddr), buf, sbase, sbase + ssize, &cur, bytes, 1); soff = cur - sbase; }
            else if (buf_id == 0xF) sg_copy(APU_REG(d->saddr), buf, sbase + soff, bytes, 1);
        } else {
            if (buf_id <= 1) fifo_copy(d, (int)buf_id, 0, buf, bytes);
            else if (buf_id == 0xE) { uint32_t cur = sbase + (soff < ssize ? soff : 0); sg_ring(APU_REG(d->saddr), buf, sbase, sbase + ssize, &cur, bytes, 0); soff = cur - sbase; }
            else if (buf_id == 0xF) sg_copy(APU_REG(d->saddr), buf, sbase + soff, bytes, 0);
            else memset(buf, 0, bytes);
            for (uint32_t i = 0; i < words; i++) {
                uint32_t v;
                if (item == 1) v = (uint32_t)buf[i] << 16;
                else if (item == 2) { uint16_t h; memcpy(&h, buf + i * 2, 2); v = (uint32_t)h << 8; }
                else memcpy(&v, buf + i * 4, 4);
                mem_wr(d, sp, ma + i, v);
            }
        }
        if (writeback) mem_wr(d, bsp, ba + 4, soff);
    }
}

static uint32_t dma_reg_read(Dsp *d, int r) {
    switch (r) {
    case 0: return d->dma.next;
    case 1: return d->dma.start;
    case 2:
        if (d->dma.control & DMA_RUNNING && ++d->dma.reads > 2) {   /* transfers finish at once; report it after a few polls */
            d->dma.control = (d->dma.control & ~DMA_RUNNING) | DMA_STOPPED;
            d->dma.reads = 0;
        }
        return d->dma.control;
    default: return d->dma.config;
    }
}
static void dma_reg_write(Dsp *d, int r, uint32_t v) {
    switch (r) {
    case 0: d->dma.next = v; break;
    case 1: d->dma.start = v; break;
    case 2:
        switch (v & 7) {
        case 1: d->dma.control = (d->dma.control | DMA_RUNNING) & ~DMA_STOPPED; d->dma.reads = 0; break;
        case 2: d->dma.control = (d->dma.control | DMA_STOPPED) & ~DMA_RUNNING; break;
        case 3: d->dma.control |= DMA_FROZEN; break;
        case 4: d->dma.control &= ~DMA_FROZEN; break;
        default: break;
        }
        dma_run(d);
        break;
    default: d->dma.config = v; break;
    }
}

/* ---------------------------------------------------------------- peripherals (X:$FFFF80-$FFFFFF) */
static uint32_t periph_read_(Dsp *d, uint32_t a);
static uint32_t periph_read(void *opaque, uint32_t a) {
    uint64_t t = g_trace ? now_ns() : 0;
    uint32_t v = periph_read_(opaque, a);
    if (g_trace) { g_cb_ns += now_ns() - t; g_cb_n++; }
    return v;
}
static uint32_t periph_read_(Dsp *d, uint32_t a) {
    switch (a) {
    case 0xFFFFB3: return 0;
    case 0xFFFFC5: return d->ints | (d->dma.eol ? INT_DMA_EOL : 0);
    case 0xFFFFD4: case 0xFFFFD5: case 0xFFFFD6: case 0xFFFFD7: return dma_reg_read(d, (int)(a - 0xFFFFD4));
    default: return d->periph[a & 0x7F];
    }
}
static void periph_write_(Dsp *d, uint32_t a, uint32_t v);
static void periph_write(void *opaque, uint32_t a, uint32_t v) {
    uint64_t t = g_trace ? now_ns() : 0;
    periph_write_(opaque, a, v);
    if (g_trace) { g_cb_ns += now_ns() - t; g_cb_n++; }
}
static void periph_write_(Dsp *d, uint32_t a, uint32_t v) {
    switch (a) {
    case 0xFFFFC4: if (v & 1) dsp56300_set_halt_requested(d->jit, 1); break;   /* frame done */
    case 0xFFFFC5: d->ints &= ~v; if (v & INT_DMA_EOL) d->dma.eol = 0; break;
    case 0xFFFFD4: case 0xFFFFD5: case 0xFFFFD6: case 0xFFFFD7: dma_reg_write(d, (int)(a - 0xFFFFD4), v); break;
    default: d->periph[a & 0x7F] = v; break;
    }
}

/* ---------------------------------------------------------------- setup */
static int dsp_create(Dsp *d) {
    Dsp56300MemoryRegion xr[4], yr[1], pr[1];
    xr[0] = (Dsp56300MemoryRegion){.start = 0, .end = XRAM_N, .kind = DSP56300_REGION_BUFFER, .data.buffer = {d->x, 0}};
    xr[1] = (Dsp56300MemoryRegion){.start = MIX_X0, .end = MIX_X0 + MIX_N, .kind = DSP56300_REGION_BUFFER, .data.buffer = {d->mix, 0}};
    xr[3] = (Dsp56300MemoryRegion){.start = MIX_X, .end = MIX_X + MIX_N, .kind = DSP56300_REGION_BUFFER, .data.buffer = {d->mix, 0}};
    xr[2] = (Dsp56300MemoryRegion){.start = 0xFFFF80, .end = 0x1000000, .kind = DSP56300_REGION_CALLBACK,
                                   .data.callback = {d, periph_read, periph_write}};
    yr[0] = (Dsp56300MemoryRegion){.start = 0, .end = YRAM_N, .kind = DSP56300_REGION_BUFFER, .data.buffer = {d->y, 0}};
    pr[0] = (Dsp56300MemoryRegion){.start = 0, .end = PRAM_N, .kind = DSP56300_REGION_BUFFER, .data.buffer = {d->p, 0}};
    Dsp56300MemoryRegion xs[4] = {xr[0], xr[1], xr[3], xr[2]};       /* in address order */
    Dsp56300CreateInfo info = {.memory_map = {xs, 4, yr, 1, pr, 1}};
    d->jit = dsp56300_create(&info);
    if (!d->jit) return 0;
    dsp56300_reset(d->jit);
    return 1;
}

int gpdsp_compiled(void) { return 1; }

int gpdsp_enabled(void) {
    if (g_enabled >= 0) return g_enabled;
    const char *e = getenv("HALO_DSP");
    g_enabled = !(e && *e == '0');
    const char *t = getenv("HALO_TRACE_DSP");
    g_trace = t && *t && *t != '0';
    if (!g_enabled) { runtime_log("[dsp] audio DSPs off (HALO_DSP=0): built-in reverb"); return 0; }
    const char *ep = getenv("HALO_DSP_EP");
    g_use_ep = ep && *ep == '1';
    if (!dsp_create(&g_gp) || (g_use_ep && !dsp_create(&g_ep))) { runtime_log("[dsp] could not create the DSPs: built-in reverb"); g_enabled = 0; return 0; }
    runtime_log("[dsp] GP DSP ready (dsp56300 JIT)%s", g_use_ep ? ", EP too (HALO_DSP_EP=1)" : "");
    return 1;
}

int gpdsp_running(void) { return g_gp.running; }

static void bootstrap(Dsp *d) {
    /* the boot ROM: scratch words 0..0x7FF -> P:0..0x7FF */
    static uint32_t img[0x800];
    sg_copy(APU_REG(d->saddr), (uint8_t *)img, 0, sizeof img, 0);
    for (uint32_t i = 0; i < 0x800; i++) dsp56300_write_memory(d->jit, DSP56300_MEM_SPACE_P, i, img[i] & 0xFFFFFF);
    dsp56300_invalidate_cache(d->jit);
}

static void dsp_mmio_write(Dsp *d, uint32_t off, uint32_t v) {
    if (off < W_X + XRAM_N * 4) { d->x[(off - W_X) / 4] = v & 0xFFFFFF; return; }
    if (off < W_X + 0x1000 * 4) { d->mix[(off - W_X) / 4 - XRAM_N] = v & 0xFFFFFF; return; }
    if (off >= W_MIX && off < W_MIX + MIX_N * 4) { d->mix[(off - W_MIX) / 4] = v & 0xFFFFFF; return; }
    if (off >= W_Y && off < W_Y + YRAM_N * 4) { d->y[(off - W_Y) / 4] = v & 0xFFFFFF; return; }
    if (off >= W_P && off < W_P + PRAM_N * 4) { dsp56300_write_memory(d->jit, DSP56300_MEM_SPACE_P, (off - W_P) / 4, v & 0xFFFFFF); return; }
    if (off == W_RST) {
        int was = (d->rst & 3) == 3, now = (v & 3) == 3;
        d->rst = v;
        if (!now) {
            if (d->running && g_trace) runtime_log("[dsp] %s held in reset (%x)", d->name, v);
            dsp56300_reset(d->jit);
            d->running = 0;
            memset(&d->dma, 0, sizeof d->dma);
            d->ints = 0;
        } else if (!was) {
            bootstrap(d);
            d->running = 1;
            if (!d->is_gp) g_frame_div = 0;
            runtime_log("[dsp] %s started: boot image loaded from scratch (P:%06x %06x %06x ...)", d->name, d->p[0], d->p[1], d->p[2]);
        }
    }
}

static int dsp_mmio_read(Dsp *d, uint32_t off, uint32_t *v) {
    if (off < W_X + XRAM_N * 4) { *v = d->x[(off - W_X) / 4]; return 1; }
    if (off < W_X + 0x1000 * 4) { *v = d->mix[(off - W_X) / 4 - XRAM_N]; return 1; }
    if (off >= W_MIX && off < W_MIX + MIX_N * 4) { *v = d->mix[(off - W_MIX) / 4]; return 1; }
    if (off >= W_Y && off < W_Y + YRAM_N * 4) { *v = d->y[(off - W_Y) / 4]; return 1; }
    if (off >= W_P && off < W_P + PRAM_N * 4) { *v = d->p[(off - W_P) / 4]; return 1; }
    return 0;
}

/* The DSPs run on their own thread (gpdsp_submit); the game's MMIO accesses
 * to their memories and reset registers wait for the frame in progress. */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

void gpdsp_mmio_write(int ep, uint32_t off, uint32_t v) {
    if (!gpdsp_enabled() || (ep && !g_use_ep)) return;
    pthread_mutex_lock(&g_lock);
    dsp_mmio_write(ep ? &g_ep : &g_gp, off, v);
    pthread_mutex_unlock(&g_lock);
}
int gpdsp_mmio_read(int ep, uint32_t off, uint32_t *v) {
    if (!gpdsp_enabled() || (ep && !g_use_ep)) return 0;
    pthread_mutex_lock(&g_lock);
    int r = dsp_mmio_read(ep ? &g_ep : &g_gp, off, v);
    pthread_mutex_unlock(&g_lock);
    return r;
}

/* ---------------------------------------------------------------- frames */
static uint32_t to24(float f) {
    if (f > 0.99999988f) f = 0.99999988f;
    if (f < -1.0f) f = -1.0f;
    return (uint32_t)(int32_t)(f * 8388608.0f) & 0xFFFFFF;
}
static float from24(uint32_t v) { return (float)((int32_t)(v << 8) >> 8) / 8388608.0f; }

/* Run one DSP frame: raise "start of frame" and run until the program says
 * it is done. Each DSP runs at about 160 MHz, some 107,000 cycles per
 * 32-sample frame; give up at limit cycles. */

static void dsp_run_frame(Dsp *d, uint32_t limit) {
    uint64_t t0 = now_ns();
    d->ints |= INT_START_FRAME;
    dsp56300_set_halt_requested(d->jit, 0);
    dsp56300_set_cycle_count(d->jit, 0);
    uint32_t cyc = 0;
    while (!dsp56300_halt_requested(d->jit)) {
        dsp56300_run(d->jit, (int32_t)limit);
        cyc = dsp56300_cycle_count(d->jit);
        if (cyc > limit) { d->unfinished++; break; }
    }
    d->frames++;
    g_run_ns += now_ns() - t0;
    d->cycles_sum += cyc;
    if (cyc > d->cycles_max) d->cycles_max = cyc;
    uint32_t every = d->is_gp ? 1500 : 188;
    if (g_trace && d->is_gp && d->frames % 300 == 0) {        /* the DirectSound <-> GP mailbox (scratch 0x800) */
        uint32_t mb[8];
        sg_copy(APU_REG(d->saddr), (uint8_t *)mb, 0x800, sizeof mb, 0);
        static uint32_t last[8];
        if (memcmp(mb, last, sizeof mb)) {
            runtime_log("[dsp] mailbox: %06x %06x %06x %06x status %06x %06x | %06x %06x", mb[0], mb[1], mb[2], mb[3], mb[4], mb[5], mb[6], mb[7]);
            memcpy(last, mb, sizeof mb);
        }
    }
    if (g_trace && d->frames % every == 0) {
        Dsp56300State st;
        dsp56300_get_state(d->jit, &st);
        runtime_log("[dsp] %s %llu frames: %.0f cycles/frame on average, max %llu, unfinished %llu; pc %06x; %.1f us/frame (%.1f%% of real time), callbacks %.1f us/frame (%.0f per frame)", d->name,
                    (unsigned long long)d->frames, (double)d->cycles_sum / every, (unsigned long long)d->cycles_max,
                    (unsigned long long)d->unfinished, st.pc, g_run_ns / 1000.0 / every, g_run_ns / 1e7 / (every * 32.0 / 48000.0),
                    g_cb_ns / 1000.0 / every, (double)g_cb_n / every);
        g_run_ns = g_cb_ns = g_cb_n = 0;
        d->cycles_sum = d->cycles_max = 0;
    }
}

static void dump_dsp(Dsp *d) {                  /* HALO_DSP_DUMP=dir: P/X/Y memory for offline disassembly */
    const char *dir = getenv("HALO_DSP_DUMP");
    if (!dir) return;
    char path[600];
    const char *sp[3] = {"p", "x", "y"};
    const uint32_t *mem[3] = {d->p, d->x, d->y};
    const uint32_t n[3] = {PRAM_N, XRAM_N, YRAM_N};
    for (int k = 0; k < 3; k++) {
        snprintf(path, sizeof path, "%s/%s_%s.bin", dir, d->name, sp[k]);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(mem[k], 4, n[k], f); fclose(f); }
    }
    Dsp56300State st;
    dsp56300_get_state(d->jit, &st);
    runtime_log("[dsp] dumped %s: pc %06x sr %06x sp %x la %06x lc %06x", d->name, st.pc, st.registers[57], st.registers[59], st.registers[62], st.registers[63]);
}

static int gpdsp_frame(float bins[GP_BINS][GP_FRAME], float *out) {
    Dsp *d = &g_gp;
    static int test = -1;
    if (test < 0) test = getenv("HALO_DSP_TEST") != NULL;
    if (test) {                                /* diagnosis: white noise into the reverb send */
        static uint32_t seed = 1;
        for (int i = 0; i < GP_FRAME; i++) { seed = seed * 1664525u + 1013904223u; bins[10][i] = (float)((int32_t)seed >> 8) / 33554432.0f; }
        static double acc; static unsigned k;
        for (int i = 0xC0; i < 0x140; i++) { float v = (float)((int32_t)(d->mix[i] << 8) >> 8) / 8388608.0f; acc += v * v; }
        if (++k % 1500 == 0) { runtime_log("[dsp] test: reverb output buffers X:CC0-D3F rms %.5f", sqrt(acc / (k * 128.0))); acc = 0; k = 0; }
    }
    if (d->frames == 3000) { dump_dsp(&g_gp); if (g_use_ep) dump_dsp(&g_ep); }
    if (g_trace && d->frames >= 4000 && d->frames < 4200) {      /* which X words the program keeps changing */
        static uint32_t prev[XRAM_N], cnt[XRAM_N];
        for (int i = 0; i < XRAM_N; i++) { if (d->frames > 4000 && d->x[i] != prev[i]) cnt[i]++; prev[i] = d->x[i]; }
        if (d->frames == 4199) {
            char line[2000]; int o = 0, run = -1;
            for (int i = 0; i <= XRAM_N && o < 1900; i++) {
                int on = i < XRAM_N && cnt[i] > 20;
                if (on && run < 0) run = i;
                if (!on && run >= 0) { o += snprintf(line + o, sizeof line - (size_t)o, " %03x-%03x", run, i - 1); run = -1; }
            }
            runtime_log("[dsp] GP X words changing every frame:%s", line);
        }
    }
    for (int b = 0; b < GP_BINS; b++)
        for (int i = 0; i < GP_FRAME; i++) d->mix[b * GP_FRAME + i] = to24(bins[b][i]);
    static uint32_t before[MIX_N];
    if (g_trace) memcpy(before, d->mix, sizeof before);
    dsp_run_frame(d, 220000);
    if (g_trace) {                             /* which bins the GP program changes, and how loud they are afterwards */
        static double chg[GP_BINS], lvl[GP_BINS]; static unsigned k;
        for (int b = 0; b < GP_BINS; b++)
            for (int i = 0; i < GP_FRAME; i++) {
                float a0 = from24(before[b * GP_FRAME + i]), a1 = from24(d->mix[b * GP_FRAME + i]);
                chg[b] += (a1 - a0) * (a1 - a0); lvl[b] += a1 * a1;
            }
        if (++k % 3000 == 0) {
            char line[1400]; int o = 0;
            for (int b = 0; b < GP_BINS; b++)
                if (chg[b] > 0 || lvl[b] > 0) o += snprintf(line + o, sizeof line - (size_t)o, " %d:%.4f/%.4f", b, sqrt(lvl[b] / (k * 32.0)), sqrt(chg[b] / (k * 32.0)));
            runtime_log("[dsp] bins after GP (rms level/change):%s", line);
            memset(chg, 0, sizeof chg); memset(lvl, 0, sizeof lvl); k = 0;
        }
    }

    if (!g_ep.running) {
        /* The GP leaves its 5.1 result in bins 0-5 (front L/R, centre, LFE,
         * rear L/R) for the EP, which encodes Dolby Digital and the analog
         * stereo mix. The EP is not run by default (its AC-3 encoder stalls in
         * the emulator); fold the six channels to stereo here instead. */
        const uint32_t *m = d->mix;
        for (int i = 0; i < GP_FRAME; i++) {
            float c = from24(m[2 * GP_FRAME + i]) * 0.7071f;
            out[i * 2] = from24(m[i]) + c + from24(m[4 * GP_FRAME + i]) * 0.7071f;
            out[i * 2 + 1] = from24(m[GP_FRAME + i]) + c + from24(m[5 * GP_FRAME + i]) * 0.7071f;
        }
        if (g_trace) {                         /* how much the GP changed the dry fold of bins 0-5, and bin 10 (reverb send) */
            static double dry, diff, send; static unsigned k;
            for (int i = 0; i < GP_FRAME; i++) {
                float l = bins[0][i] + 0.7071f * (bins[2][i] + bins[4][i]), r = bins[1][i] + 0.7071f * (bins[2][i] + bins[5][i]);
                dry += l * l + r * r;
                diff += (out[i * 2] - l) * (out[i * 2] - l) + (out[i * 2 + 1] - r) * (out[i * 2 + 1] - r);
                send += bins[10][i] * bins[10][i];
            }
            if (++k % 1500 == 0) {
                runtime_log("[dsp] rms: dry %.5f, GP output minus dry %.5f, reverb send %.5f", sqrt(dry / (k * 64.0)), sqrt(diff / (k * 64.0)), sqrt(send / (k * 32.0)));
                dry = diff = send = 0; k = 0;
            }
        }
        return GP_FRAME;
    }
    /* The EP works on 256-sample frames: run it at the start of every 8th GP
     * frame and hand out what it wrote to output FIFO 0 (16-bit stereo). */
    int produced = 0;
    if (g_frame_div % 8 == 0) {
        g_ep_out_n = 0;
        dsp_run_frame(&g_ep, 8 * 220000);
        int n = (int)(g_ep_out_n / 4);
        for (int i = 0; i < n; i++) {
            int16_t l, r;
            memcpy(&l, g_ep_out + i * 4, 2);
            memcpy(&r, g_ep_out + i * 4 + 2, 2);
            out[i * 2] = l / 32768.0f;
            out[i * 2 + 1] = r / 32768.0f;
        }
        produced = n;
        static int logged;
        if (g_trace && logged++ < 3) runtime_log("[dsp] EP frame wrote %u bytes to output FIFO 0", g_ep_out_n);
    }
    g_frame_div++;
    return produced;
}

/* ---------------------------------------------------------------- worker thread */
#define QN 1024                                /* frames queued for the DSP thread (~0.7 s; the APU clock catches up in bursts) */
static float g_q[QN][GP_BINS][GP_FRAME];
static unsigned g_qw, g_qr;
static pthread_mutex_t g_qlock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_qcv = PTHREAD_COND_INITIALIZER;
static void (*g_sink)(const float *, int);
static uint64_t g_dropped;

static void *dsp_thread(void *arg) {
    (void)arg;
    static float out[GP_OUT_MAX * 2];
    for (;;) {
        pthread_mutex_lock(&g_qlock);
        while (g_qr == g_qw) pthread_cond_wait(&g_qcv, &g_qlock);
        unsigned slot = g_qr % QN;
        pthread_mutex_unlock(&g_qlock);
        pthread_mutex_lock(&g_lock);
        int n = g_gp.running ? gpdsp_frame(g_q[slot], out) : 0;
        pthread_mutex_unlock(&g_lock);
        if (n && g_sink) g_sink(out, n);
        pthread_mutex_lock(&g_qlock);
        g_qr++;
        pthread_mutex_unlock(&g_qlock);
    }
    return NULL;
}

void gpdsp_submit(float bins[GP_BINS][GP_FRAME], void (*sink)(const float *stereo, int frames)) {
    static int started;
    if (!started) {
        pthread_t t;
        started = 1;
        pthread_create(&t, NULL, dsp_thread, NULL);
        pthread_detach(t);
    }
    pthread_mutex_lock(&g_qlock);
    g_sink = sink;
    if (g_qw - g_qr >= QN) {                   /* the DSP thread fell behind: drop this frame */
        if (g_dropped++ % 1000 == 0) runtime_log("[dsp] DSP thread behind: %llu frames dropped", (unsigned long long)g_dropped);
    } else {
        memcpy(g_q[g_qw % QN], bins, sizeof g_q[0]);
        g_qw++;
        pthread_cond_signal(&g_qcv);
    }
    pthread_mutex_unlock(&g_qlock);
}

#endif
