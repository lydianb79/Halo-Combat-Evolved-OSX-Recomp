/* gpu_probe.c — headless stand-in for a GPU backend (HALO_GPU_PROBE=1).
 * Records what the NV2A model hands to a host GPU, decodes every texture it
 * would upload (bounds check of nv2a_decode_texture_level), and reports. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../runtime/gpu_backend.h"
#include "../runtime/runtime.h"

#define MAXK 64
typedef struct { uint32_t addr, w, h, zaddr; uint64_t draws, clears; } Target;
static Target g_targets[MAXK];
static int g_ntargets;
static uint64_t g_draws, g_verts, g_clears, g_flushes, g_tex_binds, g_rt_tex, g_decoded_texels;
static uint32_t g_fmts[256];
static uint32_t g_blend_combos[64][3];
static int g_nblend;
static uint32_t g_tex_seen[1024];
static int g_ntex_seen;

static Target *target(const NvDrawState *st) {
    for (int i = 0; i < g_ntargets; i++) if (g_targets[i].addr == st->color_addr) return &g_targets[i];
    if (g_ntargets == MAXK) return &g_targets[MAXK - 1];
    Target *t = &g_targets[g_ntargets++];
    memset(t, 0, sizeof *t);
    t->addr = st->color_addr; t->w = st->width; t->h = st->height; t->zaddr = st->zeta_addr;
    return t;
}

static int is_target(uint32_t a) {
    for (int i = 0; i < g_ntargets; i++) if (g_targets[i].addr == a) return 1;
    return 0;
}

static void probe_draw(const NvDrawState *st, const OutVtx *v, int count) {
    g_draws++; g_verts += (uint64_t)count;
    if (st->width <= 128) {
        static int n;
        if (n++ < 16)
            runtime_log("[probe] small-target draw #%d flush %llu -> %08x %ux%u pitch %u clip %u,%u %ux%u zeta %08x fmt %u ztest %u zfunc %x zmask %u "
                        "blend %u %x %x cmask %08x tex0 %08x fmt %02x %ux%u stages %u, %d verts z0 %.1f w0 %.3f x0 %.1f y0 %.1f",
                        n, (unsigned long long)g_flushes, st->color_addr, st->width, st->height, st->pitch, st->clip_x, st->clip_y,
                        st->clip_w, st->clip_h, st->zeta_addr, st->zeta_fmt, st->depth_test, st->depth_func, st->depth_mask,
                        st->blend_enable, st->blend_src, st->blend_dst, st->color_mask, st->tex[0].addr, st->tex[0].fmt,
                        st->tex[0].w, st->tex[0].h, st->comb_control & 0xFF, count, v[0].z, v[0].w, v[0].x, v[0].y);
        static uint64_t lastf; static int frames;
        if (g_flushes != lastf) { lastf = g_flushes; frames++; if (frames % 100 == 0) runtime_log("[probe] small-target drawn in %d distinct flushes so far", frames); }
    }
    for (int i = 0; i < 4; i++)
        if (st->tex[i].enabled && st->tex[i].w == 64 && st->tex[i].h == 64 && st->tex[i].fmt == 6) {
            static int m;
            if (m++ < 8)
                runtime_log("[probe] sample 64x64 RT %08x stage %d -> target %08x blend %u %x %x ztest %u stages %u, %d verts, flush %llu, addr_u %u v %u",
                            st->tex[i].addr, i, st->color_addr, st->blend_enable, st->blend_src, st->blend_dst, st->depth_test,
                            st->comb_control & 0xFF, count, (unsigned long long)g_flushes, st->tex[i].addr_u, st->tex[i].addr_v);
        }
    target(st)->draws++;
    for (int i = 0; i < 4; i++) {
        const NvTexture *t = &st->tex[i];
        if (!t->enabled) continue;
        g_tex_binds++;
        if ((uint64_t)t->addr + t->bytes > 0x84000000ull || t->bytes > (16u << 20)) {
            static int k; if (k++ < 10) runtime_log("[probe] texture outside RAM: stage %d addr %08x fmt %02x %ux%u pitch %u linear %d levels %d cube %d bytes %u",
                                                   i, t->addr, t->fmt, t->w, t->h, t->pitch, t->linear, t->levels, t->cube, t->bytes);
        }
        g_fmts[t->fmt & 255]++;
        if (is_target(t->addr)) {
            g_rt_tex++;
            static int n; if (n++ < 12) runtime_log("[probe] texture = render target %08x fmt %02x %ux%u linear %d", t->addr, t->fmt, t->w, t->h, t->linear);
            continue;
        }
        for (int k = 0; k < g_ntargets; k++) {
            uint32_t a = g_targets[k].addr, end = a + g_targets[k].w * 4 * g_targets[k].h;
            if ((t->addr > a && t->addr < end) || (t->addr < a && t->addr + t->bytes > a)) {
                static int m; if (m++ < 20) runtime_log("[probe] texture %08x (fmt %02x %ux%u linear %d, %u bytes) overlaps render target %08x", t->addr, t->fmt, t->w, t->h, t->linear, t->bytes, a);
            }
        }
        int seen = 0;
        for (int k = 0; k < g_ntex_seen; k++) if (g_tex_seen[k] == t->addr) seen = 1;
        if (seen || g_ntex_seen >= 1024) continue;
        g_tex_seen[g_ntex_seen++] = t->addr;
        if (getenv("HALO_DUMP_TEX") && t->w >= 1024 && !t->cube) {   /* debugging: write every mip level as PPM */
            static uint32_t lv[2048 * 2048];
            for (int l = 0; l < t->levels; l++) {
                uint32_t w, h;
                nv2a_decode_texture_level(t, 0, l, lv, &w, &h);
                char name[96];
                snprintf(name, sizeof name, "tex_%08x_fmt%02x_l%d_%ux%u.ppm", t->addr, t->fmt, l, w, h);
                FILE *f = fopen(name, "wb");
                if (!f) continue;
                fprintf(f, "P6\n%u %u\n255\n", w, h * 2);
                for (uint32_t y = 0; y < h; y++)          /* top: colour */
                    for (uint32_t x = 0; x < w; x++) { uint32_t c = lv[y * w + x]; unsigned char p[3] = {c >> 16, c >> 8, c}; fwrite(p, 1, 3, f); }
                for (uint32_t y = 0; y < h; y++)          /* bottom: alpha */
                    for (uint32_t x = 0; x < w; x++) { unsigned char a = lv[y * w + x] >> 24, p[3] = {a, a, a}; fwrite(p, 1, 3, f); }
                fclose(f);
                runtime_log("[probe] wrote %s (levels %d, bytes %u)", name, t->levels, t->bytes);
            }
        }
        static uint32_t buf[4096 * 4096];
        for (int l = 0; l < t->levels; l++) {
            uint32_t w, h;
            for (int f = 0; f < (t->cube ? 6 : 1); f++) nv2a_decode_texture_level(t, f, l, buf, &w, &h);
            g_decoded_texels += (uint64_t)w * h;
        }
    }
    int found = 0;
    for (int k = 0; k < g_nblend; k++)
        if (g_blend_combos[k][0] == (st->blend_enable ? st->blend_src : 0xFFFF) && g_blend_combos[k][1] == st->blend_dst &&
            g_blend_combos[k][2] == st->blend_eq) found = 1;
    if (!found && g_nblend < 64) {
        g_blend_combos[g_nblend][0] = st->blend_enable ? st->blend_src : 0xFFFF;
        g_blend_combos[g_nblend][1] = st->blend_dst; g_blend_combos[g_nblend][2] = st->blend_eq; g_nblend++;
    }
    (void)v;
}

static void probe_clear(const NvDrawState *st, uint32_t flags, uint32_t color, uint32_t zs,
                        uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
    g_clears++;
    target(st)->clears++;
}

static void probe_flush(void) {
    g_flushes++;
    if (g_flushes % 600 == 0) {
        runtime_log("[probe] %llu draws, %llu verts, %llu clears, %llu flushes, %llu texture binds (%llu from render targets), "
                    "%d distinct guest textures, %llu texels decoded",
                    (unsigned long long)g_draws, (unsigned long long)g_verts, (unsigned long long)g_clears,
                    (unsigned long long)g_flushes, (unsigned long long)g_tex_binds, (unsigned long long)g_rt_tex,
                    g_ntex_seen, (unsigned long long)g_decoded_texels);
        for (int i = 0; i < g_ntargets; i++)
            runtime_log("[probe]   target %08x %ux%u zeta %08x: %llu draws, %llu clears", g_targets[i].addr, g_targets[i].w,
                        g_targets[i].h, g_targets[i].zaddr, (unsigned long long)g_targets[i].draws,
                        (unsigned long long)g_targets[i].clears);
        char line[512] = "[probe]   formats:";
        for (int f = 0; f < 256; f++) if (g_fmts[f]) snprintf(line + strlen(line), sizeof line - strlen(line), " %02x:%u", f, g_fmts[f]);
        runtime_log("%s", line);
        for (int k = 0; k < g_nblend; k++)
            runtime_log("[probe]   blend src %x dst %x eq %x", g_blend_combos[k][0], g_blend_combos[k][1], g_blend_combos[k][2]);
    }
}

/* HALO_GPU_PROBE_VS=1: offer draw_vs but never accept, so the collected
 * vertices go through the CPU fallback (tests that path headless). */
static int probe_draw_vs(const NvDrawState *st, const NvVsDraw *d) {
    (void)st;
    static uint64_t n, idx, verts;
    n++; idx += (uint64_t)d->nindices; verts += (uint64_t)d->nverts;
    if (n % 20000 == 0) runtime_log("[probe] draw_vs offered %llu times: %llu indices, %llu unique vertices", (unsigned long long)n,
                                    (unsigned long long)idx, (unsigned long long)verts);
    if (n == 1 && d->source) { const char *src = d->source(); runtime_log("[probe] first vertex program source: %zu bytes", src ? strlen(src) : 0); }
    return 0;
}
static GpuBackend g_probe = {probe_draw, probe_clear, probe_flush};
void gpu_probe_install(void) {
    if (getenv("HALO_GPU_PROBE_VS")) g_probe.draw_vs = probe_draw_vs;
    g_gpu_backend = &g_probe; runtime_log("[probe] GPU hand-off probe installed"); }
