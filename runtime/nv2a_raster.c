/* nv2a_raster.c — 3D pipeline of the NV2A model ("Kelvin" class 0x97).
 *
 * Reference software implementation, structured like the hardware:
 *   vertex attributes (immediate or arrays) -> vertex program (NV2A VSH
 *   microcode interpreter) or fixed-function transform -> primitive assembly
 *   -> culling -> triangle rasterizer -> texture stages -> register combiners
 *   (pixel "shader") -> alpha test -> depth test -> blending -> colour surface.
 * The resulting framebuffer is what the GPU scans out (shown via Metal).
 * The same decoded state is what a Metal backend will consume.
 */
#include "nv2a_raster.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_backend.h"
#include "nv2a.h"
#include "runtime.h"

GpuBackend *g_gpu_backend;
int g_gpu_cpu_shadow;

/* ================================================================ state */
#define NATTR 16
#define VP_SLOTS 136
#define VP_CONSTS 192

static struct {
    V4 attr[NATTR];                                   /* immediate attribute values */
    uint32_t array_offset[NATTR], array_format[NATTR];
    uint32_t prog[VP_SLOTS][4];
    uint32_t prog_load, prog_sub, prog_start, exec_mode;
    V4 c[VP_CONSTS];
    uint32_t const_load, const_sub;
    uint32_t prim;
    /* texture stages */
    uint32_t tex_offset[4], tex_format[4], tex_address[4], tex_control0[4], tex_control1[4];
    uint32_t tex_filter[4], tex_rect[4], tex_palette[4];
    uint32_t shader_stage_prog;                       /* 5 bits per stage */
    int stage_prog_set;
    /* register combiners */
    uint32_t color_icw[8], color_ocw[8], alpha_icw[8], alpha_ocw[8], factor0[8], factor1[8];
    uint32_t comb_control, final0, final1, fog_color;
    uint32_t fog_mode, fog_gen, fog_enable;
    uint32_t poly_offset_fill;
    float poly_scale, poly_bias;
    float fog_params[3];
    uint32_t specfog[2];                              /* final combiner constants C0/C1 (0x1E20/0x1E24) */
    /* raster ops */
    uint32_t blend_enable, blend_src, blend_dst, blend_eq, blend_color;
    uint32_t alpha_test_enable, alpha_ref, alpha_func;
    uint32_t depth_test, depth_func, depth_mask;
    uint32_t cull_enable, cull_face, front_face;
    uint32_t color_mask;
    /* stencil (Z24S8 low byte) */
    uint32_t stencil_enable, stencil_mask, stencil_func, stencil_ref, stencil_func_mask;
    uint32_t stencil_op_fail, stencil_op_zfail, stencil_op_zpass;
    float composite[16];
    int composite_set;
    uint64_t tris, pixels, verts;
} S = { .color_mask = 0x01010101, .depth_mask = 1, .depth_func = 0x203, .cull_face = 0x405, .front_face = 0x901,
        .stencil_mask = 0xFF, .stencil_func = 0x207, .stencil_func_mask = 0xFF,
        .stencil_op_fail = 0x1E00, .stencil_op_zfail = 0x1E00, .stencil_op_zpass = 0x1E00 };

static uint64_t g_prim_by_mode[4], g_prim_novert, g_prim_nosurf, g_tri_offscreen, g_tri_degenerate, g_tri_culled;

#define MAXV 65536
static OutVtx g_vbuf[MAXV];
static int g_nv;

static Nv2aSurface g_surf;
static uint32_t (*g_dma_lookup)(uint32_t handle, uint32_t *limit);
static uint32_t g_dma_a, g_dma_b;

void nv2a_raster_bind(uint32_t (*dma_lookup)(uint32_t, uint32_t *)) { g_dma_lookup = dma_lookup; }
static int g_npend;
static void resolve_pending(void);
void nv2a_raster_set_surface(const Nv2aSurface *s) { if (g_npend) resolve_pending(); g_surf = *s; }
void nv2a_raster_set_dma(uint32_t a, uint32_t b) { if (g_npend) resolve_pending(); g_dma_a = a; g_dma_b = b; }
void nv2a_raster_stats(uint64_t *tris, uint64_t *pixels) { *tris = S.tris; *pixels = S.pixels; }

static float u2f(uint32_t v) { float f; memcpy(&f, &v, 4); return f; }
static inline float clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
static inline V4 argb_to_v4(uint32_t c) {
    return (V4){(float)(c >> 16 & 255) / 255.0f, (float)(c >> 8 & 255) / 255.0f, (float)(c & 255) / 255.0f, (float)(c >> 24) / 255.0f};
}

/* ============================================================ memory */
static uint32_t dma_base_cached(uint32_t handle) {
    static uint32_t h[4] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}, b[4];
    static int next;
    for (int k = 0; k < 4; k++) if (h[k] == handle) return b[k];
    uint32_t lim = 0, base = g_dma_lookup ? g_dma_lookup(handle, &lim) : 0;
    h[next] = handle; b[next] = base; next = (next + 1) & 3;
    return base;
}
static inline const uint8_t *phys_ptr(uint32_t p) { return g_mem + GUEST_RAM_MIRROR1 + (p & (GUEST_RAM_SIZE - 1)); }

/* ============================================================ textures */
typedef struct {
    const uint8_t *base;
    const uint8_t *palette;
    uint32_t fmt, w, h, pitch, linear, addr_u, addr_v;
    int enabled;
    int cube;               /* cube map: 6 faces, each a mip chain, face_bytes apart */
    uint32_t face_bytes;
} Sampler;
static Sampler g_samp[4];
#define MAX_LEVELS 13
static Sampler g_lvl[4][MAX_LEVELS];   /* per mipmap level: base, w, h */
static int g_levels[4];
static uint32_t tex_bytes(uint32_t f) {   /* bytes per texel, swizzled formats */
    switch (f) {
    case 0x00: case 0x01: case 0x0B: case 0x19: return 1;
    case 0x02: case 0x03: case 0x04: case 0x05: case 0x1A: return 2;
    default: return 4;
    }
}

static inline uint32_t swz(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    uint32_t r = 0, bit = 0;
    for (uint32_t m = 1; m < w || m < h; m <<= 1) {
        if (m < w) { r |= (x & m) ? (1u << bit) : 0; bit++; }
        if (m < h) { r |= (y & m) ? (1u << bit) : 0; bit++; }
    }
    return r;
}

static inline uint32_t c565(uint16_t v) {
    uint32_t r = (v >> 11) & 31, g = (v >> 5) & 63, b = v & 31;
    return 0xFF000000u | ((r << 3 | r >> 2) << 16) | ((g << 2 | g >> 4) << 8) | (b << 3 | b >> 2);
}
static inline uint32_t c1555(uint16_t s, int force_alpha) {
    uint32_t r = s >> 10 & 31, g = s >> 5 & 31, b = s & 31, a = (force_alpha || (s & 0x8000)) ? 255 : 0;
    return a << 24 | (r << 3 | r >> 2) << 16 | (g << 3 | g >> 2) << 8 | (b << 3 | b >> 2);
}
static inline uint32_t c4444(uint16_t s) {
    return (uint32_t)(s >> 12) * 17 << 24 | (uint32_t)(s >> 8 & 15) * 17 << 16 | (uint32_t)(s >> 4 & 15) * 17 << 8 | (uint32_t)(s & 15) * 17;
}
static inline uint32_t abgr_to_argb(uint32_t v) { return (v & 0xFF00FF00u) | (v >> 16 & 255) | (v & 255) << 16; }

static uint32_t dxt_texel(const uint8_t *base, uint32_t w, uint32_t x, uint32_t y, int kind) {
    uint32_t bw = (w + 3) / 4, bsz = kind == 1 ? 8 : 16;
    const uint8_t *blk = base + ((y / 4) * bw + (x / 4)) * bsz;
    const uint8_t *cb = kind == 1 ? blk : blk + 8;
    uint16_t c0 = (uint16_t)(cb[0] | cb[1] << 8), c1 = (uint16_t)(cb[2] | cb[3] << 8);
    uint32_t bits = (uint32_t)cb[4] | cb[5] << 8 | cb[6] << 16 | (uint32_t)cb[7] << 24;
    uint32_t idx = (bits >> (2 * ((y & 3) * 4 + (x & 3)))) & 3;
    uint32_t p0 = c565(c0), p1 = c565(c1), col;
#define MIX(a, b, na, nb, d) ((((((a) >> 16 & 255) * na + ((b) >> 16 & 255) * nb) / d) << 16) | \
                              (((((a) >> 8 & 255) * na + ((b) >> 8 & 255) * nb) / d) << 8) |   \
                              ((((a) & 255) * na + ((b) & 255) * nb) / d))
    if (kind != 1 || c0 > c1)
        col = idx == 0 ? p0 : idx == 1 ? p1 : idx == 2 ? (0xFF000000u | MIX(p0, p1, 2, 1, 3)) : (0xFF000000u | MIX(p0, p1, 1, 2, 3));
    else
        col = idx == 0 ? p0 : idx == 1 ? p1 : idx == 2 ? (0xFF000000u | MIX(p0, p1, 1, 1, 2)) : 0x00000000u;
#undef MIX
    uint32_t a = col >> 24;
    if (kind == 3) {
        uint32_t n = (y & 3) * 4 + (x & 3);
        a = (blk[n / 2] >> ((n & 1) * 4)) & 15;
        a = a << 4 | a;
    } else if (kind == 5) {
        uint32_t a0 = blk[0], a1 = blk[1];
        uint64_t ab = 0;
        for (int i = 0; i < 6; i++) ab |= (uint64_t)blk[2 + i] << (8 * i);
        uint32_t ai = (uint32_t)(ab >> (3 * ((y & 3) * 4 + (x & 3)))) & 7;
        if (ai == 0) a = a0; else if (ai == 1) a = a1;
        else if (a0 > a1) a = ((8 - ai) * a0 + (ai - 1) * a1) / 7;
        else a = ai == 6 ? 0 : ai == 7 ? 255 : ((6 - ai) * a0 + (ai - 1) * a1) / 5;
    }
    return (col & 0x00FFFFFFu) | (a << 24);
}

/* NV097 colour formats: swizzled (power-of-two, normalized coords) and
 * linear "LU" (any size, texel coords, explicit pitch). */
static int fmt_is_linear(uint32_t f) {
    switch (f) {
    case 0x10: case 0x11: case 0x12: case 0x13: case 0x1B: case 0x1C: case 0x1D: case 0x1E: case 0x1F: case 0x20:
    case 0x3F: case 0x40: case 0x41: return 1;
    default: return 0;
    }
}

static uint32_t fetch(const Sampler *sp, uint32_t x, uint32_t y) {
    const uint8_t *b = sp->base;
    uint32_t v;
    uint16_t s;
    uint32_t lo = sp->linear ? y * sp->pitch : 0;     /* byte offset of row (linear) */
    uint32_t si = sp->linear ? 0 : swz(x, y, sp->w, sp->h);
    switch (sp->fmt) {
    /* swizzled */
    case 0x00: return 0xFF000000u | b[si] * 0x010101u;                                   /* L8 */
    case 0x01: return (uint32_t)b[si] << 24 | b[si] * 0x010101u;                         /* AY8 */
    case 0x02: memcpy(&s, b + si * 2, 2); return c1555(s, 0);                            /* A1R5G5B5 */
    case 0x03: memcpy(&s, b + si * 2, 2); return c1555(s, 1);                            /* X1R5G5B5 */
    case 0x04: memcpy(&s, b + si * 2, 2); return c4444(s);                               /* A4R4G4B4 */
    case 0x05: memcpy(&s, b + si * 2, 2); return c565(s);                                /* R5G6B5 */
    case 0x06: memcpy(&v, b + si * 4, 4); return v;                                      /* A8R8G8B8 */
    case 0x07: memcpy(&v, b + si * 4, 4); return v | 0xFF000000u;                        /* X8R8G8B8 */
    case 0x0B: {                                                                          /* I8 palette */
        uint8_t i = b[si];
        if (!sp->palette) return 0xFFFF00FFu;
        memcpy(&v, sp->palette + i * 4, 4); return v;
    }
    case 0x0C: return dxt_texel(b, sp->w, x, y, 1);
    case 0x0E: return dxt_texel(b, sp->w, x, y, 3);
    case 0x0F: return dxt_texel(b, sp->w, x, y, 5);
    case 0x19: return (uint32_t)b[si] << 24 | 0xFFFFFF;                                  /* A8 */
    case 0x1A: memcpy(&s, b + si * 2, 2); return (uint32_t)(s >> 8) << 24 | (s & 255) * 0x010101u;  /* A8Y8 */
    case 0x3A: memcpy(&v, b + si * 4, 4); return abgr_to_argb(v);                        /* A8B8G8R8 */
    case 0x3B: memcpy(&v, b + si * 4, 4); return v >> 8 | v << 24;                       /* B8G8R8A8 */
    case 0x3C: memcpy(&v, b + si * 4, 4); return v >> 8 | (v & 255) << 24;               /* R8G8B8A8 (approx.) */
    /* linear */
    case 0x10: memcpy(&s, b + lo + x * 2, 2); return c1555(s, 0);
    case 0x11: memcpy(&s, b + lo + x * 2, 2); return c565(s);
    case 0x12: memcpy(&v, b + lo + x * 4, 4); return v;
    case 0x13: return 0xFF000000u | b[lo + x] * 0x010101u;                               /* Y8 */
    case 0x1B: return (uint32_t)b[lo + x] << 24 | b[lo + x] * 0x010101u;                 /* AY8 */
    case 0x1C: memcpy(&s, b + lo + x * 2, 2); return c1555(s, 1);
    case 0x1D: memcpy(&s, b + lo + x * 2, 2); return c4444(s);
    case 0x1E: memcpy(&v, b + lo + x * 4, 4); return v | 0xFF000000u;
    case 0x1F: return (uint32_t)b[lo + x] << 24 | 0xFFFFFF;                              /* A8 */
    case 0x20: memcpy(&s, b + lo + x * 2, 2); return (uint32_t)(s >> 8) << 24 | (s & 255) * 0x010101u;
    case 0x3F: memcpy(&v, b + lo + x * 4, 4); return abgr_to_argb(v);                    /* A8B8G8R8 */
    default: {
        static uint32_t warned[256];
        if (!warned[sp->fmt & 255]++) runtime_log("[raster] texture format %02x not supported yet (sampled as white)", sp->fmt);
        return 0xFFFFFFFFu;
    }
    }
}

static inline int32_t wrap_coord(int32_t c, uint32_t n, uint32_t mode) {
    switch (mode) {
    case 2: {                                                   /* MIRROR */
        int32_t p = (int32_t)(2 * n), m = c % p; if (m < 0) m += p;
        return m < (int32_t)n ? m : p - 1 - m;
    }
    case 3: case 4: case 5:                                     /* CLAMP variants */
        return c < 0 ? 0 : c >= (int32_t)n ? (int32_t)n - 1 : c;
    default: {                                                  /* WRAP */
        int32_t m = c % (int32_t)n; return m < 0 ? m + (int32_t)n : m;
    }
    }
}

static V4 sample(const Sampler *sp, float s, float t) {
    int32_t x, y;
    if (sp->linear) { x = (int32_t)floorf(s); y = (int32_t)floorf(t); }
    else { x = (int32_t)floorf(s * (float)sp->w); y = (int32_t)floorf(t * (float)sp->h); }
    x = wrap_coord(x, sp->w, sp->linear ? 3 : sp->addr_u);
    y = wrap_coord(y, sp->h, sp->linear ? 3 : sp->addr_v);
    return argb_to_v4(fetch(sp, (uint32_t)x, (uint32_t)y));
}

static void setup_samplers(void) {
    for (int i = 0; i < 4; i++) {
        Sampler *sp = &g_samp[i];
        uint32_t mode = (S.shader_stage_prog >> (5 * i)) & 31;
        sp->enabled = (S.tex_control0[i] >> 30) & 1;
        if (!S.stage_prog_set && i == 0) mode = 1;               /* program never set: plain 2D */
        uint32_t f = S.tex_format[i];
        /* 1 2D, 2 3D (sampled as 2D), 3 cube map; other modes (bump, dot products...) not yet */
        sp->cube = mode == 3 && (f & 4);
        if (mode != 1 && mode != 2 && !sp->cube) sp->enabled = 0;
        if (!sp->enabled) continue;
        sp->fmt = (f >> 8) & 0xFF;
        sp->linear = fmt_is_linear(sp->fmt);
        if (sp->linear) { sp->w = S.tex_rect[i] >> 16; sp->h = S.tex_rect[i] & 0xFFFF; sp->pitch = S.tex_control1[i] >> 16; }
        else { sp->w = 1u << ((f >> 20) & 15); sp->h = 1u << ((f >> 24) & 15); sp->pitch = 0; }
        if (!sp->w || !sp->h) { sp->enabled = 0; continue; }
        uint32_t dma = (f & 3) == 2 ? g_dma_b : g_dma_a;
        sp->base = phys_ptr(dma_base_cached(dma) + S.tex_offset[i]);
        uint32_t pal = S.tex_palette[i];
        sp->palette = pal ? phys_ptr(dma_base_cached((pal & 3) == 2 ? g_dma_b : g_dma_a) + (pal & ~0x3Fu)) : NULL;
        sp->addr_u = S.tex_address[i] & 15;
        sp->addr_v = (S.tex_address[i] >> 8) & 15;
        /* mipmap chain: swizzled levels follow each other, halving down to 1x1 */
        uint32_t levels = sp->linear ? 1 : (f >> 16) & 15;
        if (levels < 1) levels = 1;
        if (levels > MAX_LEVELS) levels = MAX_LEVELS;
        uint32_t bpt = tex_bytes(sp->fmt), off = 0, w = sp->w, h = sp->h;
        g_levels[i] = levels;
        for (uint32_t l = 0; l < levels; l++) {
            Sampler *L = &g_lvl[i][l];
            *L = *sp;
            L->base = sp->base + off;
            L->w = w; L->h = h;
            if (sp->fmt == 0x0C || sp->fmt == 0x0E || sp->fmt == 0x0F)
                off += ((w + 3) / 4) * ((h + 3) / 4) * (sp->fmt == 0x0C ? 8u : 16u);
            else off += w * h * bpt;
            if (w > 1) w >>= 1;
            if (h > 1) h >>= 1;
        }
        sp->face_bytes = sp->cube ? (off + 127) & ~127u : off;   /* cube faces are 128-byte aligned */
    }
}

/* -------------------------------------------------------- texture filtering
 * Modes (Graphics → Texture Filtering, or HALO_TEX_FILTER):
 *   0 game default (the filters the game sets per texture stage)
 *   1 fastest: nearest texel of the full-size image, no mipmaps
 *   2 nearest, 3 bilinear, 4 trilinear, 5-8 anisotropic 2x/4x/8x/16x
 * Footprints come from the screen-space derivatives of the texture
 * coordinates (finite differences in the pixel loop). */
static int g_tex_mode = -1;
static int g_need_deriv;
void nv2a_raster_set_texture_filter(int mode) { g_tex_mode = mode < 0 ? 0 : mode > 8 ? 8 : mode; }

static inline V4 texel_at(const Sampler *L, int32_t x, int32_t y) {
    x = wrap_coord(x, L->w, L->linear ? 3 : L->addr_u);
    y = wrap_coord(y, L->h, L->linear ? 3 : L->addr_v);
    return argb_to_v4(fetch(L, (uint32_t)x, (uint32_t)y));
}

/* u, v in texels of level L */
static V4 filter_level(const Sampler *L, float u, float v, int linear) {
    if (!linear) return texel_at(L, (int32_t)floorf(u), (int32_t)floorf(v));
    u -= 0.5f; v -= 0.5f;
    float fx0 = floorf(u), fy0 = floorf(v), fx = u - fx0, fy = v - fy0;
    int32_t x0 = (int32_t)fx0, y0 = (int32_t)fy0;
    V4 a = texel_at(L, x0, y0), b = texel_at(L, x0 + 1, y0), c = texel_at(L, x0, y0 + 1), d = texel_at(L, x0 + 1, y0 + 1);
    float wa = (1 - fx) * (1 - fy), wb = fx * (1 - fy), wc = (1 - fx) * fy, wd = fx * fy;
    return (V4){a.x * wa + b.x * wb + c.x * wc + d.x * wd, a.y * wa + b.y * wb + c.y * wc + d.y * wd,
                a.z * wa + b.z * wb + c.z * wc + d.z * wd, a.w * wa + b.w * wb + c.w * wc + d.w * wd};
}

/* s, t: normalized (swizzled) or texel (linear) coordinates of level 0 */
static V4 filter_lod(int stage, float s, float t, float lod, int linear, int mip) {
    int n = g_levels[stage];
    const Sampler *sp = &g_samp[stage];
    #define AT(l) (sp->linear ? filter_level(&g_lvl[stage][l], s, t, linear) \
                              : filter_level(&g_lvl[stage][l], s * (float)g_lvl[stage][l].w, t * (float)g_lvl[stage][l].h, linear))
    if (mip == 0 || n == 1 || lod <= 0) return AT(0);
    if (mip == 1) {                                                  /* nearest level */
        int l = (int)(lod + 0.5f);
        return AT(l < n ? l : n - 1);
    }
    int l0 = (int)lod;                                               /* linear between levels */
    if (l0 >= n - 1) return AT(n - 1);
    float f = lod - (float)l0;
    V4 a = AT(l0), b = AT(l0 + 1);
    #undef AT
    return (V4){a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.z + (b.z - a.z) * f, a.w + (b.w - a.w) * f};
}

/* Filters for a stage under the current texture-filtering mode:
 * magnification linear?, minification linear?, mip 0 none / 1 nearest / 2 linear, max anisotropy. */
static void tex_stat(uint32_t tf, uint32_t fmt, int levels, int linear, uint32_t w, uint32_t h);
static void resolve_filter(int stage, int *mag_lin, int *min_lin, int *mip, int *aniso) {
    if (g_tex_mode == 1) { *mag_lin = 0; *min_lin = 0; *mip = 0; *aniso = 1; return; }
    switch (g_tex_mode) {
    case 2: *mag_lin = 0; *min_lin = 0; *mip = 1; *aniso = 1; break;
    case 3: *mag_lin = 1; *min_lin = 1; *mip = 1; *aniso = 1; break;
    case 4: *mag_lin = 1; *min_lin = 1; *mip = 2; *aniso = 1; break;
    case 5: case 6: case 7: case 8: *mag_lin = 1; *min_lin = 1; *mip = 2; *aniso = 1 << (g_tex_mode - 4); break;
    default: {                                                       /* what the game asked for */
        uint32_t tf = S.tex_filter[stage], minf = (tf >> 16) & 15, magf = (tf >> 24) & 15;
        *mag_lin = magf == 2 || magf == 4;
        *min_lin = minf == 2 || minf == 4 || minf == 6 || minf == 7;
        *mip = (minf == 3 || minf == 4) ? 1 : (minf == 5 || minf == 6) ? 2 : 0;
        *aniso = 1 << ((S.tex_control0[stage] >> 4) & 3);
        break;
    }
    }
}

/* d = {ds/dx, dt/dx, ds/dy, dt/dy}, or NULL */
/* Cube map: pick the face from the major axis of (s, t, r) (D3D/Metal layout:
 * +X -X +Y -Y +Z -Z) and sample level 0 of that face. */
static V4 sample_cube(int stage, float s, float t, float r) {
    const Sampler *sp = &g_samp[stage];
    float as = fabsf(s), at = fabsf(t), ar = fabsf(r), ma, sc, tc;
    int face;
    if (as >= at && as >= ar) { ma = as; face = s >= 0 ? 0 : 1; sc = s >= 0 ? -r : r; tc = -t; }
    else if (at >= ar) { ma = at; face = t >= 0 ? 2 : 3; sc = s; tc = t >= 0 ? r : -r; }
    else { ma = ar; face = r >= 0 ? 4 : 5; sc = r >= 0 ? s : -s; tc = -t; }
    if (ma == 0) return (V4){0, 0, 0, 0};
    Sampler L = g_lvl[stage][0];
    L.base += (size_t)face * sp->face_bytes;
    L.addr_u = L.addr_v = 3;
    float u = (sc / ma + 1.0f) * 0.5f * (float)L.w, v = (tc / ma + 1.0f) * 0.5f * (float)L.h;
    int mag_lin, min_lin, mip, aniso;
    resolve_filter(stage, &mag_lin, &min_lin, &mip, &aniso);
    return filter_level(&L, u, v, mag_lin || min_lin);
}

static V4 sample_stage(int stage, float s, float t, const V4 *d) {
    const Sampler *sp = &g_samp[stage];
    if (g_tex_mode == 1 || !d) return sample(sp, s, t);
    float sx = sp->linear ? 1.0f : (float)sp->w, sy = sp->linear ? 1.0f : (float)sp->h;
    float ax = d->x * sx, ay = d->y * sy, bx = d->z * sx, by = d->w * sy;
    float lx = sqrtf(ax * ax + ay * ay), ly = sqrtf(bx * bx + by * by);
    float major = fmaxf(lx, ly), minor = fminf(lx, ly);
    int mag_lin, min_lin, mip, aniso;
    resolve_filter(stage, &mag_lin, &min_lin, &mip, &aniso);
    int n = 1;
    if (aniso > 1 && minor > 1e-6f) { n = (int)ceilf(major / minor); if (n > aniso) n = aniso; }
    float lod = log2f(fmaxf(major / (float)n, 1e-6f));
    if (lod <= 0) return filter_lod(stage, s, t, 0, mag_lin, 0);
    if (n == 1) return filter_lod(stage, s, t, lod, min_lin, mip);
    /* anisotropic: n probes along the long axis of the pixel footprint */
    float ms = lx >= ly ? d->x : d->z, mt = lx >= ly ? d->y : d->w;
    V4 acc = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        float o = ((float)i + 0.5f) / (float)n - 0.5f;
        V4 c = filter_lod(stage, s + ms * o, t + mt * o, lod, min_lin, mip);
        acc.x += c.x; acc.y += c.y; acc.z += c.z; acc.w += c.w;
    }
    float r = 1.0f / (float)n;
    return (V4){acc.x * r, acc.y * r, acc.z * r, acc.w * r};
}

/* ====================================================== vertex program */
enum { MAC_NOP, MAC_MOV, MAC_MUL, MAC_ADD, MAC_MAD, MAC_DP3, MAC_DPH, MAC_DP4, MAC_DST, MAC_MIN, MAC_MAX,
       MAC_SLT, MAC_SGE, MAC_ARL };
enum { ILU_NOP, ILU_MOV, ILU_RCP, ILU_RCC, ILU_RSQ, ILU_EXP, ILU_LOG, ILU_LIT };

static inline uint32_t fld(const uint32_t *ins, int w, int sh, int n) { return (ins[w] >> sh) & ((1u << n) - 1); }

static void write_mask(V4 *dst, V4 v, uint32_t mask) {   /* mask bits: 8=x 4=y 2=z 1=w */
    if (mask & 8) dst->x = v.x;
    if (mask & 4) dst->y = v.y;
    if (mask & 2) dst->z = v.z;
    if (mask & 1) dst->w = v.w;
}

/* The program is decoded once (after an upload or a new start slot) into
 * this form; running it per vertex then touches no bit fields. */
typedef float f4 __attribute__((vector_size(16)));
typedef int32_t i4 __attribute__((vector_size(16)));
typedef struct {
    i4 lm_mac, lm_ilu, lm_o;            /* write masks as lane masks (all ones = write) */
    uint8_t swk[3];                     /* swizzle kind per source: 0 xyzw, 1 one component broadcast, 2 general */
    uint8_t mac, ilu, mux[3], reg[3], neg[3], sw[3][4];
    uint8_t vidx, rel, mac_mask, rdst, ilu_mask, o_mask, orb, omux, final;
    uint16_t cidx, oaddr;
} VPIns;
static VPIns g_vp[VP_SLOTS];
static int g_vp_check = -1;
static uint64_t g_vp_hash;
static int g_vp_hash_valid;
static unsigned g_vp_check_bad, g_vp_check_n;
static uint64_t g_xf_ns, g_xf_n;      /* HALO_VSTATS: time spent transforming queued vertices */
static int g_xf_time = -1;
static int g_vp_len, g_vp_dirty = 1, g_vp_writes_consts;

static void decode_program(void) {
    if (g_vp_check < 0) g_vp_check = getenv("HALO_VP_REF") ? 2 : getenv("HALO_VP_CHECK") != NULL;   /* 2: old interpreter (timing) */
    g_vp_len = 0;
    g_vp_writes_consts = 0;
    for (int pc = (int)S.prog_start; pc < VP_SLOTS; pc++) {
        const uint32_t *ins = S.prog[pc];
        VPIns *d = &g_vp[g_vp_len++];
        d->ilu = (uint8_t)fld(ins, 1, 25, 3);
        d->mac = (uint8_t)fld(ins, 1, 21, 4);
        d->cidx = (uint16_t)fld(ins, 1, 13, 8);
        d->vidx = (uint8_t)fld(ins, 1, 9, 4);
        d->mux[0] = (uint8_t)fld(ins, 2, 26, 2); d->mux[1] = (uint8_t)fld(ins, 2, 11, 2); d->mux[2] = (uint8_t)fld(ins, 3, 28, 2);
        d->reg[0] = (uint8_t)fld(ins, 2, 28, 4); d->reg[1] = (uint8_t)fld(ins, 2, 13, 4);
        d->reg[2] = (uint8_t)((fld(ins, 2, 0, 2) << 2) | fld(ins, 3, 30, 2));
        d->neg[0] = (uint8_t)fld(ins, 1, 8, 1); d->neg[1] = (uint8_t)fld(ins, 2, 25, 1); d->neg[2] = (uint8_t)fld(ins, 2, 10, 1);
        const int shifts[3][5] = {{1, 6, 4, 2, 0}, {2, 23, 21, 19, 17}, {2, 8, 6, 4, 2}};
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 4; j++) d->sw[k][j] = (uint8_t)fld(ins, shifts[k][0], shifts[k][1 + j], 2);
        d->rel = (uint8_t)fld(ins, 3, 1, 1);
        d->mac_mask = (uint8_t)fld(ins, 3, 24, 4); d->rdst = (uint8_t)fld(ins, 3, 20, 4);
        d->ilu_mask = (uint8_t)fld(ins, 3, 16, 4); d->o_mask = (uint8_t)fld(ins, 3, 12, 4);
        d->orb = (uint8_t)fld(ins, 3, 11, 1); d->oaddr = (uint16_t)fld(ins, 3, 3, 8); d->omux = (uint8_t)fld(ins, 3, 2, 1);
        d->final = (uint8_t)fld(ins, 3, 0, 1);
        for (int k = 0; k < 3; k++) {
            const uint8_t *w = d->sw[k];
            d->swk[k] = (w[0] == 0 && w[1] == 1 && w[2] == 2 && w[3] == 3) ? 0 : (w[0] == w[1] && w[1] == w[2] && w[2] == w[3]) ? 1 : 2;
        }
        #define LANES(m) ((i4){((m) & 8) ? -1 : 0, ((m) & 4) ? -1 : 0, ((m) & 2) ? -1 : 0, ((m) & 1) ? -1 : 0})
        d->lm_mac = LANES(d->mac_mask); d->lm_ilu = LANES(d->ilu_mask); d->lm_o = LANES(d->o_mask);
        #undef LANES
        if (d->o_mask && !d->orb) g_vp_writes_consts = 1;
        if (d->final) break;
    }
    g_vp_dirty = 0;
    g_vp_hash_valid = 0;
}

static inline V4 vp_src(const VPIns *d, int k, const V4 *r, const V4 *o, const V4 *in, int ci) {
    const V4 *v;
    static const V4 zero = {0, 0, 0, 0};
    switch (d->mux[k]) {
    case 1: v = d->reg[k] >= 12 ? &o[0] : &r[d->reg[k]]; break;
    case 2: v = &in[d->vidx]; break;
    case 3: v = &S.c[ci]; break;
    default: v = &zero; break;
    }
    const float *a = &v->x;
    V4 x = {a[d->sw[k][0]], a[d->sw[k][1]], a[d->sw[k][2]], a[d->sw[k][3]]};
    if (d->neg[k]) { x.x = -x.x; x.y = -x.y; x.z = -x.z; x.w = -x.w; }
    return x;
}

/* Runs the vertex program on inputs v[16]; outputs o[16] (o0 = position). */
static uint64_t g_vp_runs, g_vp_ins;
static void run_vertex_program_ref(const V4 *in, V4 *o) {
    V4 r[12];
    memset(r, 0, sizeof r);
    /* r12 is an alias of oPos on the NV2A: D3D's epilogue reads it back */
    #define TREG(i) ((i) >= 12 ? &o[0] : &r[(i)])
    int a0 = 0;
    for (int n = 0; n < g_vp_len; n++) {
        const VPIns *d = &g_vp[n];
        int ci = (int)d->cidx + (d->rel ? a0 : 0);
        if (ci < 0) ci = 0;
        if (ci >= VP_CONSTS) ci = VP_CONSTS - 1;
        V4 A = d->mux[0] ? vp_src(d, 0, r, o, in, ci) : (V4){0, 0, 0, 0};
        V4 B = d->mux[1] ? vp_src(d, 1, r, o, in, ci) : (V4){0, 0, 0, 0};
        V4 C = d->mux[2] ? vp_src(d, 2, r, o, in, ci) : (V4){0, 0, 0, 0};
        V4 mr = {0, 0, 0, 0}, ir = {0, 0, 0, 0};
        switch (d->mac) {
        case MAC_MOV: mr = A; break;
        case MAC_MUL: mr = (V4){A.x * B.x, A.y * B.y, A.z * B.z, A.w * B.w}; break;
        case MAC_ADD: mr = (V4){A.x + C.x, A.y + C.y, A.z + C.z, A.w + C.w}; break;
        case MAC_MAD: mr = (V4){A.x * B.x + C.x, A.y * B.y + C.y, A.z * B.z + C.z, A.w * B.w + C.w}; break;
        case MAC_DP3: { float t = A.x * B.x + A.y * B.y + A.z * B.z; mr = (V4){t, t, t, t}; break; }
        case MAC_DPH: { float t = A.x * B.x + A.y * B.y + A.z * B.z + B.w; mr = (V4){t, t, t, t}; break; }
        case MAC_DP4: { float t = A.x * B.x + A.y * B.y + A.z * B.z + A.w * B.w; mr = (V4){t, t, t, t}; break; }
        case MAC_DST: mr = (V4){1.0f, A.y * B.y, A.z, B.w}; break;
        case MAC_MIN: mr = (V4){fminf(A.x, B.x), fminf(A.y, B.y), fminf(A.z, B.z), fminf(A.w, B.w)}; break;
        case MAC_MAX: mr = (V4){fmaxf(A.x, B.x), fmaxf(A.y, B.y), fmaxf(A.z, B.z), fmaxf(A.w, B.w)}; break;
        case MAC_SLT: mr = (V4){A.x < B.x, A.y < B.y, A.z < B.z, A.w < B.w}; break;
        case MAC_SGE: mr = (V4){A.x >= B.x, A.y >= B.y, A.z >= B.z, A.w >= B.w}; break;
        case MAC_ARL: a0 = (int)floorf(A.x); break;
        default: break;
        }
        float cw = C.w;
        switch (d->ilu) {
        case ILU_MOV: ir = C; break;
        case ILU_RCP: { float v = cw == 0 ? INFINITY : 1.0f / cw; ir = (V4){v, v, v, v}; break; }
        case ILU_RCC: { float v = 1.0f / cw; if (isinf(v) || fabsf(v) > 1.884467e19f) v = copysignf(1.884467e19f, v);
                        else if (fabsf(v) < 5.42101e-20f) v = copysignf(5.42101e-20f, v); ir = (V4){v, v, v, v}; break; }
        case ILU_RSQ: { float v = 1.0f / sqrtf(fabsf(cw)); ir = (V4){v, v, v, v}; break; }
        case ILU_EXP: ir = (V4){exp2f(floorf(cw)), cw - floorf(cw), exp2f(cw), 1.0f}; break;
        case ILU_LOG: { float v = fabsf(cw); int e = 0; float m = v ? frexpf(v, &e) * 2.0f : 0; e = v ? e - 1 : 0;
                        ir = (V4){(float)e, m, v ? log2f(v) : -INFINITY, 1.0f}; break; }
        case ILU_LIT: { float dd = fmaxf(C.x, 0), sp = fmaxf(C.y, 0), p = fminf(fmaxf(cw, -127.9961f), 127.9961f);
                        ir = (V4){1.0f, dd, dd > 0 ? powf(sp, p) : 0, 1.0f}; break; }
        default: break;
        }
        if (d->mac && d->mac != MAC_ARL && d->mac_mask) write_mask(TREG(d->rdst), mr, d->mac_mask);
        /* when an instruction pairs a MAC and an ILU op, the ILU result goes to r1 */
        if (d->ilu && d->ilu_mask) write_mask(d->mac ? &r[1] : TREG(d->rdst), ir, d->ilu_mask);
        if (d->o_mask) {
            V4 res = d->omux ? ir : mr;
            if (d->orb) write_mask(&o[d->oaddr & 15], res, d->o_mask);                    /* output register */
            else if (d->oaddr < VP_CONSTS) write_mask(&S.c[d->oaddr], res, d->o_mask);    /* constant write-back */
        }
    }
    #undef TREG
}

/* The same program run with 4-wide vector operations: plain and broadcast
 * swizzles load directly, masked writes are lane blends. HALO_VP_CHECK=1 runs
 * the scalar interpreter above next to it and logs differences. */
static inline f4 f4ld(const V4 *p) { f4 r; memcpy(&r, p, 16); return r; }
static inline void f4st(V4 *p, f4 v) { memcpy(p, &v, 16); }
static inline void f4wm(V4 *dst, f4 v, i4 m) {
    f4 old = f4ld(dst);
    f4st(dst, (f4)(((i4)v & m) | ((i4)old & ~m)));
}
static inline f4 vsrc(const VPIns *d, int k, const V4 *r, const V4 *o, const V4 *in, int ci) {
    const V4 *v;
    switch (d->mux[k]) {
    case 1: v = d->reg[k] >= 12 ? &o[0] : &r[d->reg[k]]; break;
    case 2: v = &in[d->vidx]; break;
    case 3: v = &S.c[ci]; break;
    default: { f4 z = {0, 0, 0, 0}; return d->neg[k] ? -z : z; }
    }
    f4 x;
    if (d->swk[k] == 0) x = f4ld(v);
    else {
        const float *a = &v->x;
        if (d->swk[k] == 1) { float f = a[d->sw[k][0]]; x = (f4){f, f, f, f}; }
        else x = (f4){a[d->sw[k][0]], a[d->sw[k][1]], a[d->sw[k][2]], a[d->sw[k][3]]};
    }
    return d->neg[k] ? -x : x;
}
static void run_vertex_program(const V4 *in, V4 *o) {
    V4 r[12];
    memset(r, 0, sizeof r);
    int a0 = 0;
    const f4 zero = {0, 0, 0, 0};
    for (int n = 0; n < g_vp_len; n++) {
        const VPIns *d = &g_vp[n];
        int ci = (int)d->cidx + (d->rel ? a0 : 0);
        if (ci < 0) ci = 0;
        if (ci >= VP_CONSTS) ci = VP_CONSTS - 1;
        f4 A = d->mux[0] ? vsrc(d, 0, r, o, in, ci) : zero;
        f4 B = d->mux[1] ? vsrc(d, 1, r, o, in, ci) : zero;
        f4 C = d->mux[2] ? vsrc(d, 2, r, o, in, ci) : zero;
        f4 mr = zero, ir = zero;
        switch (d->mac) {
        case MAC_MOV: mr = A; break;
        case MAC_MUL: mr = A * B; break;
        case MAC_ADD: mr = A + C; break;
        case MAC_MAD: mr = A * B + C; break;
        case MAC_DP3: { float t = A[0] * B[0] + A[1] * B[1] + A[2] * B[2]; mr = (f4){t, t, t, t}; break; }
        case MAC_DPH: { float t = A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + B[3]; mr = (f4){t, t, t, t}; break; }
        case MAC_DP4: { float t = A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + A[3] * B[3]; mr = (f4){t, t, t, t}; break; }
        case MAC_DST: mr = (f4){1.0f, A[1] * B[1], A[2], B[3]}; break;
        case MAC_MIN: mr = (f4){fminf(A[0], B[0]), fminf(A[1], B[1]), fminf(A[2], B[2]), fminf(A[3], B[3])}; break;
        case MAC_MAX: mr = (f4){fmaxf(A[0], B[0]), fmaxf(A[1], B[1]), fmaxf(A[2], B[2]), fmaxf(A[3], B[3])}; break;
        case MAC_SLT: mr = (f4){A[0] < B[0], A[1] < B[1], A[2] < B[2], A[3] < B[3]}; break;
        case MAC_SGE: mr = (f4){A[0] >= B[0], A[1] >= B[1], A[2] >= B[2], A[3] >= B[3]}; break;
        case MAC_ARL: a0 = (int)floorf(A[0]); break;
        default: break;
        }
        if (d->ilu) {
            float cw = C[3];
            switch (d->ilu) {
            case ILU_MOV: ir = C; break;
            case ILU_RCP: { float v = cw == 0 ? INFINITY : 1.0f / cw; ir = (f4){v, v, v, v}; break; }
            case ILU_RCC: { float v = 1.0f / cw; if (isinf(v) || fabsf(v) > 1.884467e19f) v = copysignf(1.884467e19f, v);
                            else if (fabsf(v) < 5.42101e-20f) v = copysignf(5.42101e-20f, v); ir = (f4){v, v, v, v}; break; }
            case ILU_RSQ: { float v = 1.0f / sqrtf(fabsf(cw)); ir = (f4){v, v, v, v}; break; }
            case ILU_EXP: ir = (f4){exp2f(floorf(cw)), cw - floorf(cw), exp2f(cw), 1.0f}; break;
            case ILU_LOG: { float v = fabsf(cw); int e = 0; float m = v ? frexpf(v, &e) * 2.0f : 0; e = v ? e - 1 : 0;
                            ir = (f4){(float)e, m, v ? log2f(v) : -INFINITY, 1.0f}; break; }
            case ILU_LIT: { float dd = fmaxf(C[0], 0), sp = fmaxf(C[1], 0), p = fminf(fmaxf(cw, -127.9961f), 127.9961f);
                            ir = (f4){1.0f, dd, dd > 0 ? powf(sp, p) : 0, 1.0f}; break; }
            default: break;
            }
        }
        if (d->mac && d->mac != MAC_ARL && d->mac_mask) f4wm(d->rdst >= 12 ? &o[0] : &r[d->rdst], mr, d->lm_mac);
        /* when an instruction pairs a MAC and an ILU op, the ILU result goes to r1 */
        if (d->ilu && d->ilu_mask) f4wm(d->mac ? &r[1] : (d->rdst >= 12 ? &o[0] : &r[d->rdst]), ir, d->lm_ilu);
        if (d->o_mask) {
            f4 res = d->omux ? ir : mr;
            if (d->orb) f4wm(&o[d->oaddr & 15], res, d->lm_o);
            else if (d->oaddr < VP_CONSTS) f4wm(&S.c[d->oaddr], res, d->lm_o);
        }
    }
}

static void vp_check(const V4 *in, const V4 *o_fast, const V4 *o_init) {
    V4 o2[16];
    memcpy(o2, o_init, sizeof o2);
    run_vertex_program_ref(in, o2);
    __atomic_add_fetch(&g_vp_check_n, 1, __ATOMIC_RELAXED);
    for (int i = 0; i < 16; i++) {
        const float *a = &o_fast[i].x, *b = &o2[i].x;
        for (int k = 0; k < 4; k++) {
            float da = a[k], db = b[k];
            if (da == db || (isnan(da) && isnan(db))) continue;
            float tol = 1e-4f * fmaxf(1.0f, fmaxf(fabsf(da), fabsf(db)));
            if (fabsf(da - db) <= tol) continue;
            if (__atomic_add_fetch(&g_vp_check_bad, 1, __ATOMIC_RELAXED) <= 20)
                runtime_log("[vpcheck] o%d.%c: fast %g, reference %g (program length %d)", i, "xyzw"[k], da, db, g_vp_len);
            return;
        }
    }
}


/* ===================================================== GPU vertex programs
 * vs_generate() translates the decoded program (g_vp) into straight-line
 * shader code. The body uses only per-component scalar operations and the
 * nv_* helpers, so the same text compiles as Metal (VS_PREAMBLE_MSL, the
 * backend) and as C++ (the HALO_VS_TEST harness checks it against the
 * interpreter on real game data). Inputs V0..V15, constants c[], outputs
 * O0..O15 (o0 = position). Programs that write constants are not translated. */
static uint64_t vs_hash(void) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (int n = 0; n < g_vp_len; n++) {
        const VPIns *d = &g_vp[n];
        const uint8_t *b = &d->swk[0];                 /* everything after the lane masks */
        size_t len = sizeof(VPIns) - (size_t)(b - (const uint8_t *)d);
        for (size_t i = 0; i < len; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    }
    return h ^ (uint64_t)g_vp_len;
}

typedef struct { char *p; size_t n, cap; } SBuf;
static void sb(SBuf *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void sb(SBuf *s, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(s->p + s->n, s->cap > s->n ? s->cap - s->n : 0, fmt, ap);
    va_end(ap);
    if (k > 0) s->n += (size_t)k;
}

static void vs_src(SBuf *s, const VPIns *d, int k, char *out, size_t cap) {
    char base[32];
    switch (d->mux[k]) {
    case 1: if (d->reg[k] >= 12) snprintf(base, sizeof base, "O0"); else snprintf(base, sizeof base, "R%u", d->reg[k]); break;
    case 2: snprintf(base, sizeof base, "V%u", d->vidx); break;
    case 3: if (d->rel) snprintf(base, sizeof base, "c[ci]"); else snprintf(base, sizeof base, "c[%u]", d->cidx < VP_CONSTS ? d->cidx : VP_CONSTS - 1); break;
    default: snprintf(out, cap, "float4(0.0f, 0.0f, 0.0f, 0.0f)"); return;
    }
    const char *neg = d->neg[k] ? "-" : "";
    snprintf(out, cap, "float4(%s%s.%c, %s%s.%c, %s%s.%c, %s%s.%c)", neg, base, "xyzw"[d->sw[k][0]], neg, base, "xyzw"[d->sw[k][1]],
             neg, base, "xyzw"[d->sw[k][2]], neg, base, "xyzw"[d->sw[k][3]]);
    (void)s;
}

static void vs_write(SBuf *s, const char *dst, const char *src, uint32_t mask) {
    for (int j = 0; j < 4; j++)
        if (mask & (8u >> j)) sb(s, "    %s.%c = %s.%c;\n", dst, "xyzw"[j], src, "xyzw"[j]);
}

/* the function body: expects float4 V0..V15, a constant array c, declares O0..O15 */
static int vs_generate(SBuf *s) {
    if (g_vp_dirty) decode_program();
    if (g_vp_writes_consts) return 0;
    sb(s, "  float4 R0 = float4(0.0f, 0.0f, 0.0f, 0.0f)");
    for (int i = 1; i < 12; i++) sb(s, ", R%d = R0", i);
    sb(s, ";\n  float4 O0 = float4(0.0f, 0.0f, 0.0f, 1.0f)");
    for (int i = 1; i < 16; i++) sb(s, i == 3 ? ", O3 = float4(1.0f, 1.0f, 1.0f, 1.0f)" : ", O%d = O0", i);
    sb(s, ";\n  int A0 = 0;\n  (void)A0;\n");
    for (int n = 0; n < g_vp_len; n++) {
        const VPIns *d = &g_vp[n];
        char a[160], b[160], c[160];
        sb(s, "  { // %d\n", n);
        if (d->rel && (d->mux[0] == 3 || d->mux[1] == 3 || d->mux[2] == 3))
            sb(s, "    int ci = nv_clampc(%u + A0);\n", d->cidx);
        vs_src(s, d, 0, a, sizeof a); vs_src(s, d, 1, b, sizeof b); vs_src(s, d, 2, c, sizeof c);
        sb(s, "    float4 A = %s;\n    float4 B = %s;\n    float4 C = %s;\n", a, b, c);
        sb(s, "    (void)A; (void)B; (void)C;\n");
        const char *mac;
        switch (d->mac) {
        case MAC_MOV: mac = "A"; break;
        case MAC_MUL: mac = "nv_mul(A, B)"; break;
        case MAC_ADD: mac = "nv_add(A, C)"; break;
        case MAC_MAD: mac = "nv_mad(A, B, C)"; break;
        case MAC_DP3: mac = "nv_dp3(A, B)"; break;
        case MAC_DPH: mac = "nv_dph(A, B)"; break;
        case MAC_DP4: mac = "nv_dp4(A, B)"; break;
        case MAC_DST: mac = "nv_dst(A, B)"; break;
        case MAC_MIN: mac = "nv_min(A, B)"; break;
        case MAC_MAX: mac = "nv_max(A, B)"; break;
        case MAC_SLT: mac = "nv_slt(A, B)"; break;
        case MAC_SGE: mac = "nv_sge(A, B)"; break;
        default: mac = "float4(0.0f, 0.0f, 0.0f, 0.0f)"; break;
        }
        const char *ilu;
        switch (d->ilu) {
        case ILU_MOV: ilu = "C"; break;
        case ILU_RCP: ilu = "nv_rcp(C.w)"; break;
        case ILU_RCC: ilu = "nv_rcc(C.w)"; break;
        case ILU_RSQ: ilu = "nv_rsq(C.w)"; break;
        case ILU_EXP: ilu = "nv_exp(C.w)"; break;
        case ILU_LOG: ilu = "nv_log(C.w)"; break;
        case ILU_LIT: ilu = "nv_lit(C)"; break;
        default: ilu = "float4(0.0f, 0.0f, 0.0f, 0.0f)"; break;
        }
        sb(s, "    float4 M = %s;\n    float4 I = %s;\n    (void)M; (void)I;\n", mac, ilu);
        if (d->mac == MAC_ARL) sb(s, "    A0 = nv_floori(A.x);\n");
        char dst[8];
        if (d->rdst >= 12) snprintf(dst, sizeof dst, "O0"); else snprintf(dst, sizeof dst, "R%u", d->rdst);
        if (d->mac && d->mac != MAC_ARL && d->mac_mask) vs_write(s, dst, "M", d->mac_mask);
        if (d->ilu && d->ilu_mask) vs_write(s, d->mac ? "R1" : dst, "I", d->ilu_mask);
        if (d->o_mask && d->orb) {
            char od[8];
            snprintf(od, sizeof od, "O%u", d->oaddr & 15);
            vs_write(s, od, d->omux ? "I" : "M", d->o_mask);
        }
        sb(s, "  }\n");
    }
    return 1;
}

/* helpers, identical in meaning in both languages (the interpreter's semantics) */
#define VS_HELPERS \
    "static inline float4 nv_mul(float4 a, float4 b) { return float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }\n" \
    "static inline float4 nv_add(float4 a, float4 c) { return float4(a.x + c.x, a.y + c.y, a.z + c.z, a.w + c.w); }\n" \
    "static inline float4 nv_mad(float4 a, float4 b, float4 c) { return float4(a.x * b.x + c.x, a.y * b.y + c.y, a.z * b.z + c.z, a.w * b.w + c.w); }\n" \
    "static inline float4 nv_s(float t) { return float4(t, t, t, t); }\n" \
    "static inline float4 nv_dp3(float4 a, float4 b) { return nv_s(a.x * b.x + a.y * b.y + a.z * b.z); }\n" \
    "static inline float4 nv_dph(float4 a, float4 b) { return nv_s(a.x * b.x + a.y * b.y + a.z * b.z + b.w); }\n" \
    "static inline float4 nv_dp4(float4 a, float4 b) { return nv_s(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w); }\n" \
    "static inline float4 nv_dst(float4 a, float4 b) { return float4(1.0f, a.y * b.y, a.z, b.w); }\n" \
    "static inline float4 nv_min(float4 a, float4 b) { return float4(NV_FMIN(a.x, b.x), NV_FMIN(a.y, b.y), NV_FMIN(a.z, b.z), NV_FMIN(a.w, b.w)); }\n" \
    "static inline float4 nv_max(float4 a, float4 b) { return float4(NV_FMAX(a.x, b.x), NV_FMAX(a.y, b.y), NV_FMAX(a.z, b.z), NV_FMAX(a.w, b.w)); }\n" \
    "static inline float4 nv_slt(float4 a, float4 b) { return float4(a.x < b.x ? 1.0f : 0.0f, a.y < b.y ? 1.0f : 0.0f, a.z < b.z ? 1.0f : 0.0f, a.w < b.w ? 1.0f : 0.0f); }\n" \
    "static inline float4 nv_sge(float4 a, float4 b) { return float4(a.x >= b.x ? 1.0f : 0.0f, a.y >= b.y ? 1.0f : 0.0f, a.z >= b.z ? 1.0f : 0.0f, a.w >= b.w ? 1.0f : 0.0f); }\n" \
    "static inline float4 nv_rcp(float w) { return nv_s(w == 0.0f ? NV_INF : 1.0f / w); }\n" \
    "static inline float4 nv_rcc(float w) { float v = 1.0f / w; if (NV_ISINF(v) || NV_FABS(v) > 1.884467e19f) v = NV_COPYSIGN(1.884467e19f, v);\n" \
    "  else if (NV_FABS(v) < 5.42101e-20f) v = NV_COPYSIGN(5.42101e-20f, v); return nv_s(v); }\n" \
    "static inline float4 nv_rsq(float w) { return nv_s(1.0f / NV_SQRT(NV_FABS(w))); }\n" \
    "static inline float4 nv_exp(float w) { return float4(NV_EXP2(NV_FLOOR(w)), w - NV_FLOOR(w), NV_EXP2(w), 1.0f); }\n" \
    "static inline float4 nv_log(float w) { float v = NV_FABS(w); if (v == 0.0f) return float4(0.0f, 0.0f, -NV_INF, 1.0f);\n" \
    "  int e = 0; float m = NV_FREXP(v, e) * 2.0f; return float4((float)(e - 1), m, NV_LOG2(v), 1.0f); }\n" \
    "static inline float4 nv_lit(float4 c) { float dd = NV_FMAX(c.x, 0.0f), sp = NV_FMAX(c.y, 0.0f), p = NV_FMIN(NV_FMAX(c.w, -127.9961f), 127.9961f);\n" \
    "  return float4(1.0f, dd, dd > 0.0f ? NV_POW(sp, p) : 0.0f, 1.0f); }\n" \
    "static inline int nv_floori(float x) { return (int)NV_FLOOR(x); }\n" \
    "static inline int nv_clampc(int i) { return i < 0 ? 0 : i > 191 ? 191 : i; }\n"

#define VS_PREAMBLE_MSL \
    "#include <metal_stdlib>\nusing namespace metal;\n" \
    "#define NV_FMIN fmin\n#define NV_FMAX fmax\n#define NV_INF INFINITY\n#define NV_ISINF isinf\n#define NV_FABS fabs\n" \
    "#define NV_COPYSIGN copysign\n#define NV_SQRT sqrt\n#define NV_EXP2 exp2\n#define NV_FLOOR floor\n#define NV_LOG2 log2\n#define NV_POW pow\n" \
    "#define NV_FREXP(v, e) frexp(v, e)\n" \
    "struct NVVaryings { float4 pos [[position, invariant]]; float4 d0, d1; float4 t0, t1, t2, t3; float fog; };\n" \
    "struct NVVsUniforms { float W, H, zmax, pad; };\n" \
    VS_HELPERS

/* the Metal vertex function around a body: inputs are 16 float4 per vertex */
static void vs_msl(SBuf *s, const char *body) {
    sb(s, "%s", VS_PREAMBLE_MSL);
    sb(s, "vertex NVVaryings nv_vs_prog(uint vid [[vertex_id]], const device float4 *vin [[buffer(0)]],\n"
          "                              constant float4 *c [[buffer(1)]], constant NVVsUniforms &u [[buffer(2)]]) {\n");
    for (int i = 0; i < 16; i++) sb(s, "  float4 V%d = vin[vid * 16 + %d];\n", i, i);
    sb(s, "%s", body);
    sb(s, "  float w = O0.w == 0.0f ? 1.0f : O0.w;\n"
          "  NVVaryings o;\n"
          "  o.pos = float4((O0.x / u.W * 2.0f - 1.0f) * w, (1.0f - O0.y / u.H * 2.0f) * w, O0.z / u.zmax * w, w);\n"
          "  o.d0 = O3; o.d1 = O4; o.t0 = O9; o.t1 = O10; o.t2 = O11; o.t3 = O12; o.fog = O5.x;\n"
          "  return o;\n}\n");
}

/* HALO_VS_TEST=dir: for each new program, write a C++ test that runs the
 * translated body on captured inputs and compares with the interpreter. */
static void vs_test_dump(const V4 *ins, int nin) {
    static const char *dir = (const char *)1;
    static uint64_t done[512];
    static int ndone;
    if (dir == (const char *)1) dir = getenv("HALO_VS_TEST");
    if (!dir || ndone >= 512 || nin <= 0) return;
    if (g_vp_dirty) decode_program();
    if (g_vp_writes_consts) return;
    uint64_t h = vs_hash();
    for (int i = 0; i < ndone; i++) if (done[i] == h) return;
    done[ndone++] = h;
    static char buf[1 << 20];
    SBuf s = {buf, 0, sizeof buf};
    if (!vs_generate(&s) || s.n >= s.cap) return;
    char path[512];
    snprintf(path, sizeof path, "%s/vs_%016llx.cpp", dir, (unsigned long long)h);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "#include \"vs_shim.h\"\n" VS_HELPERS "static void prog(const float4 *V_, const float4 *c, float4 *O_) {\n");
    for (int i = 0; i < 16; i++) fprintf(f, "  float4 V%d = V_[%d];\n", i, i);
    fwrite(buf, 1, s.n, f);
    for (int i = 0; i < 16; i++) fprintf(f, "  O_[%d] = O%d;\n", i, i);
    fprintf(f, "}\nstatic const float CONSTS[192 * 4] = {");
    for (int i = 0; i < VP_CONSTS; i++) fprintf(f, "%a, %a, %a, %a,%s", S.c[i].x, S.c[i].y, S.c[i].z, S.c[i].w, i % 4 == 3 ? "\n" : " ");
    fprintf(f, "};\nstatic const int NV = %d;\nstatic const float IN[][64] = {\n", nin);
    for (int v = 0; v < nin; v++) {
        fprintf(f, "{");
        for (int i = 0; i < 16; i++) fprintf(f, "%a, %a, %a, %a, ", ins[v * 16 + i].x, ins[v * 16 + i].y, ins[v * 16 + i].z, ins[v * 16 + i].w);
        fprintf(f, "},\n");
    }
    fprintf(f, "};\nstatic const float OUT[][64] = {\n");
    for (int v = 0; v < nin; v++) {
        V4 o[16];
        for (int i = 0; i < 16; i++) o[i] = (V4){0, 0, 0, 1};
        o[3] = (V4){1, 1, 1, 1};
        run_vertex_program_ref(&ins[v * 16], o);
        fprintf(f, "{");
        for (int i = 0; i < 16; i++) fprintf(f, "%a, %a, %a, %a, ", o[i].x, o[i].y, o[i].z, o[i].w);
        fprintf(f, "},\n");
    }
    fprintf(f, "};\n#include \"vs_main.h\"\n");
    fclose(f);
}

/* ===================================================== vertex assembly */
static int g_dbg_small = -1, g_dbg_left;
/* Thread-safe: reads S, g_vp and the inputs only (callers make sure the
 * program is decoded and does not write constants). */
static void compute_outputs(const V4 *in, V4 *o) {
    /* NV2A output registers start as (0, 0, 0, 1): screen-space shaders that
     * write only oPos.xy rely on w = 1 in D3D's divide-by-w epilogue. The
     * diffuse colour defaults to white. */
    for (int i = 0; i < 16; i++) o[i] = (V4){0, 0, 0, 1};
    o[3] = (V4){1, 1, 1, 1};
    if ((S.exec_mode & 3) == 2) {
        if (g_vp_check == 1 && !g_vp_writes_consts) {
            V4 init[16];
            memcpy(init, o, sizeof init);
            run_vertex_program(in, o);
            vp_check(in, o, init);
        } else if (g_vp_check == 2) run_vertex_program_ref(in, o);
        else run_vertex_program(in, o);
    } else {
        /* fixed function: position through the composite matrix (row-major,
         * includes the viewport), then the perspective divide. Lighting,
         * texgen and texture matrices are not modelled yet. */
        if (S.composite_set) {
            const float *m = S.composite;
            V4 p = in[0];
            o[0].x = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3] * p.w;
            o[0].y = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7] * p.w;
            o[0].z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11] * p.w;
            o[0].w = m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15] * p.w;
            if (o[0].w != 0) { float iw = 1.0f / o[0].w; o[0].x *= iw; o[0].y *= iw; o[0].z *= iw; }
        } else o[0] = in[0];
        o[3] = in[3];
        o[4] = in[4];
        for (int k = 0; k < 4; k++) o[9 + k] = in[9 + k];
    }
}
static inline void pack_out(const V4 *o, OutVtx *v) {
    float w = o[0].w;
    v->x = o[0].x; v->y = o[0].y; v->z = o[0].z; v->w = w == 0 ? 1.0f : w;
    v->d0 = o[3];
    v->d1 = o[4];
    for (int k = 0; k < 4; k++) v->t[k] = o[9 + k];
    v->fog = o[5].x;
}
static void vs_add_vertex(const V4 *in);
static int g_vs_mode;
static void emit_vertex(const V4 *in) {
    if (g_vs_mode) { vs_add_vertex(in); return; }
    V4 o[16];
    if ((S.exec_mode & 3) == 2) { if (g_vp_dirty) decode_program(); g_vp_runs++; g_vp_ins += (uint64_t)g_vp_len; }
    compute_outputs(in, o);
    int dbg = g_dbg_left > 0 && g_surf.pitch == 256;
    if (dbg) {
        g_dbg_left--;
        runtime_log("[dbg] mode %x start %u: in0 %g %g %g %g | in3 %g %g %g %g | in9 %g %g %g %g -> o0 %g %g %g %g o9 %g %g",
                    S.exec_mode, S.prog_start, in[0].x, in[0].y, in[0].z, in[0].w, in[3].x, in[3].y, in[3].z, in[3].w,
                    in[9].x, in[9].y, in[9].z, in[9].w, o[0].x, o[0].y, o[0].z, o[0].w, o[9].x, o[9].y);
        if (g_dbg_left == 0) {
            for (int i = 0; i < NATTR; i++)
                if (S.array_format[i] & 0xF0) runtime_log("[dbg]   attr %d format %08x offset %08x", i, S.array_format[i], S.array_offset[i]);
            for (int pc = (int)S.prog_start; pc < VP_SLOTS && pc < (int)S.prog_start + 16; pc++) {
                runtime_log("[dbg]   vp[%d] %08x %08x %08x %08x", pc, S.prog[pc][0], S.prog[pc][1], S.prog[pc][2], S.prog[pc][3]);
                if (S.prog[pc][3] & 1) break;
            }
            for (int c = 0; c < VP_CONSTS; c++)
                if (S.c[c].x != 0 || S.c[c].y != 0 || S.c[c].z != 0 || S.c[c].w != 0)
                    if (c < 8 || c >= 56) runtime_log("[dbg]   c%d %g %g %g %g", c, S.c[c].x, S.c[c].y, S.c[c].z, S.c[c].w);
        }
    }
    if (g_nv >= MAXV) return;
    pack_out(o, &g_vbuf[g_nv++]);
    S.verts++;
}

/* ================================================= register combiners */
/* Input byte: [3:0] register, [4] alpha/blue select, [7:5] mapping. */
enum { R_ZERO = 0, R_C0 = 1, R_C1 = 2, R_FOG = 3, R_V0 = 4, R_V1 = 5, R_T0 = 8, R_R0 = 12, R_R1 = 13,
       R_V1R0 = 14, R_EF = 15 };

static inline float map_in(float x, uint32_t m) {
    switch (m) {
    case 0: return x < 0 ? 0 : x;                       /* UNSIGNED_IDENTITY */
    case 1: return 1.0f - clamp01(x);                   /* UNSIGNED_INVERT */
    case 2: return 2.0f * (x < 0 ? 0 : x) - 1.0f;       /* EXPAND_NORMAL */
    case 3: return -2.0f * (x < 0 ? 0 : x) + 1.0f;      /* EXPAND_NEGATE */
    case 4: return (x < 0 ? 0 : x) - 0.5f;              /* HALFBIAS_NORMAL */
    case 5: return -(x < 0 ? 0 : x) + 0.5f;             /* HALFBIAS_NEGATE */
    case 6: return x;                                   /* SIGNED_IDENTITY */
    default: return -x;                                 /* SIGNED_NEGATE */
    }
}

static inline V4 comb_in_rgb(const V4 *reg, uint32_t sel) {
    V4 r = reg[sel & 15];
    uint32_t m = (sel >> 5) & 7;
    if (sel & 0x10) return (V4){map_in(r.w, m), map_in(r.w, m), map_in(r.w, m), 0};
    return (V4){map_in(r.x, m), map_in(r.y, m), map_in(r.z, m), 0};
}
static inline float comb_in_a(const V4 *reg, uint32_t sel) {
    V4 r = reg[sel & 15];
    return map_in((sel & 0x10) ? r.w : r.z, (sel >> 5) & 7);
}
static inline float out_map(float x, uint32_t op) {
    switch (op) {
    case 1: x = x - 0.5f; break;                        /* BIAS */
    case 2: x = x * 2.0f; break;                        /* SHIFTLEFT_1 */
    case 3: x = (x - 0.5f) * 2.0f; break;               /* SHIFTLEFT_1_BIAS */
    case 4: x = x * 4.0f; break;                        /* SHIFTLEFT_2 */
    case 6: x = x * 0.5f; break;                        /* SHIFTRIGHT_1 */
    default: break;
    }
    return x < -1 ? -1 : x > 1 ? 1 : x;
}

static V4 run_combiners(V4 *reg) {
    uint32_t nstages = S.comb_control & 0xFF;
    if (nstages > 8) nstages = 8;
    int msb_mux = (S.comb_control >> 8) & 1;
    for (uint32_t i = 0; i < nstages; i++) {
        reg[R_C0] = argb_to_v4(S.factor0[(S.comb_control >> 12) & 1 ? i : 0]);
        reg[R_C1] = argb_to_v4(S.factor1[(S.comb_control >> 16) & 1 ? i : 0]);
        /* colour portion */
        uint32_t icw = S.color_icw[i], ocw = S.color_ocw[i];
        V4 A = comb_in_rgb(reg, icw >> 24), B = comb_in_rgb(reg, icw >> 16 & 255),
           C = comb_in_rgb(reg, icw >> 8 & 255), D = comb_in_rgb(reg, icw & 255);
        V4 ab, cd, sum;
        if (ocw & (1u << 13)) { float d = A.x * B.x + A.y * B.y + A.z * B.z; ab = (V4){d, d, d, 0}; }
        else ab = (V4){A.x * B.x, A.y * B.y, A.z * B.z, 0};
        if (ocw & (1u << 12)) { float d = C.x * D.x + C.y * D.y + C.z * D.z; cd = (V4){d, d, d, 0}; }
        else cd = (V4){C.x * D.x, C.y * D.y, C.z * D.z, 0};
        int mux_sel = msb_mux ? reg[R_R0].w >= 0.5f : ((int)(reg[R_R0].w * 255.0f) & 1);
        if (ocw & (1u << 14)) sum = mux_sel ? cd : ab;
        else sum = (V4){ab.x + cd.x, ab.y + cd.y, ab.z + cd.z, 0};
        uint32_t op = (ocw >> 15) & 7;
        /* alpha portion (computed from the same inputs before writes) */
        uint32_t aicw = S.alpha_icw[i], aocw = S.alpha_ocw[i];
        float aA = comb_in_a(reg, aicw >> 24), aB = comb_in_a(reg, aicw >> 16 & 255),
              aC = comb_in_a(reg, aicw >> 8 & 255), aD = comb_in_a(reg, aicw & 255);
        float aab = aA * aB, acd = aC * aD;
        float asum = (aocw & (1u << 14)) ? (mux_sel ? acd : aab) : aab + acd;
        uint32_t aop = (aocw >> 15) & 7;
        /* writes */
        uint32_t d_cd = ocw & 15, d_ab = (ocw >> 4) & 15, d_sum = (ocw >> 8) & 15;
        if (d_ab) { reg[d_ab].x = out_map(ab.x, op); reg[d_ab].y = out_map(ab.y, op); reg[d_ab].z = out_map(ab.z, op);
                    if (ocw & (1u << 19)) reg[d_ab].w = reg[d_ab].z; }
        if (d_cd) { reg[d_cd].x = out_map(cd.x, op); reg[d_cd].y = out_map(cd.y, op); reg[d_cd].z = out_map(cd.z, op);
                    if (ocw & (1u << 18)) reg[d_cd].w = reg[d_cd].z; }
        if (d_sum) { reg[d_sum].x = out_map(sum.x, op); reg[d_sum].y = out_map(sum.y, op); reg[d_sum].z = out_map(sum.z, op); }
        uint32_t ad_cd = aocw & 15, ad_ab = (aocw >> 4) & 15, ad_sum = (aocw >> 8) & 15;
        if (ad_ab) reg[ad_ab].w = out_map(aab, aop);
        if (ad_cd) reg[ad_cd].w = out_map(acd, aop);
        if (ad_sum) reg[ad_sum].w = out_map(asum, aop);
    }
    /* final combiner: rgb = A*B + (1-A)*C + D, alpha = G; E*F and V1+R0 are extra inputs.
     * Its constants C0/C1 are the separate specular/fog factors, not the stage factors. */
    reg[R_C0] = argb_to_v4(S.specfog[0]);
    reg[R_C1] = argb_to_v4(S.specfog[1]);
    uint32_t f0 = S.final0, f1 = S.final1;
    V4 sum = {reg[R_V1].x + reg[R_R0].x, reg[R_V1].y + reg[R_R0].y, reg[R_V1].z + reg[R_R0].z, 0};
    if (f1 & 0x80) { sum.x = clamp01(sum.x); sum.y = clamp01(sum.y); sum.z = clamp01(sum.z); }
    reg[R_V1R0] = sum;
    #define FIN(sel) ({ V4 r_ = reg[(sel) & 15]; float k_[3] = {r_.x, r_.y, r_.z}; \
        if ((sel) & 0x10) k_[0] = k_[1] = k_[2] = r_.w; \
        for (int j_ = 0; j_ < 3; j_++) { k_[j_] = clamp01(k_[j_]); if ((sel) & 0x20) k_[j_] = 1.0f - k_[j_]; } \
        (V4){k_[0], k_[1], k_[2], 0}; })
    V4 E = FIN(f1 >> 24), F = FIN(f1 >> 16 & 255);
    reg[R_EF] = (V4){E.x * F.x, E.y * F.y, E.z * F.z, 0};
    V4 A = FIN(f0 >> 24), B = FIN(f0 >> 16 & 255), C = FIN(f0 >> 8 & 255), D = FIN(f0 & 255);
    #undef FIN
    uint32_t g = f1 >> 8 & 255;
    float ga = clamp01((g & 0x10) ? reg[g & 15].w : reg[g & 15].z);
    if (g & 0x20) ga = 1.0f - ga;
    return (V4){clamp01(A.x * B.x + (1 - A.x) * C.x + D.x), clamp01(A.y * B.y + (1 - A.y) * C.y + D.y),
                clamp01(A.z * B.z + (1 - A.z) * C.z + D.z), ga};
}

/* ===================================================== rasterization */
static inline uint32_t pack(V4 c) {
    return (uint32_t)(clamp01(c.w) * 255.0f + 0.5f) << 24 | (uint32_t)(clamp01(c.x) * 255.0f + 0.5f) << 16 |
           (uint32_t)(clamp01(c.y) * 255.0f + 0.5f) << 8 | (uint32_t)(clamp01(c.z) * 255.0f + 0.5f);
}

static inline float blend_factor(uint32_t f, float sa, float da, float sc, float dc, float cc, float ca) {
    switch (f) {
    case 0: return 0; case 1: return 1;
    case 0x300: return sc; case 0x301: return 1 - sc;
    case 0x302: return sa; case 0x303: return 1 - sa;
    case 0x304: return da; case 0x305: return 1 - da;
    case 0x306: return dc; case 0x307: return 1 - dc;
    case 0x308: return fminf(sa, 1 - da);                                /* SRC_ALPHA_SATURATE */
    case 0x8001: return cc; case 0x8002: return 1 - cc;                  /* CONSTANT_COLOR */
    case 0x8003: return ca; case 0x8004: return 1 - ca;                  /* CONSTANT_ALPHA */
    default: return 1;
    }
}

static inline float blend_eq(float s, float d) {
    switch (S.blend_eq) {
    case 0x800A: return s - d;          /* SUBTRACT */
    case 0x800B: return d - s;          /* REVERSE_SUBTRACT */
    case 0x8007: return fminf(s, d);
    case 0x8008: return fmaxf(s, d);
    default: return s + d;              /* FUNC_ADD */
    }
}

static inline int compare(uint32_t func, float a, float b) {
    switch (func & 7) {
    case 0: return 0; case 1: return a < b; case 2: return a == b; case 3: return a <= b;
    case 4: return a > b; case 5: return a != b; case 6: return a >= b; default: return 1;
    }
}

static int g_has_combiners;

static void shade_pixel(int x, int y, float z, V4 d0, V4 d1, const V4 *t, const V4 *dt, uint64_t *npix) {
    /* depth */
    uint8_t *zp = NULL;
    uint32_t zbits = 0, zmax = 0;
    if (g_surf.zeta_addr && S.depth_test && !g_surf.swizzled) {
        zp = g_mem + g_surf.zeta_addr + (uint32_t)y * g_surf.zeta_pitch;
        if (g_surf.zeta_fmt == 1) { zmax = 0xFFFF; zp += (uint32_t)x * 2; }
        else { zmax = 0xFFFFFF; zp += (uint32_t)x * 4; }
        float zc = z < 0 ? 0 : z > (float)zmax ? (float)zmax : z;
        zbits = (uint32_t)zc;
        if (S.depth_test) {
            uint32_t cur;
            if (g_surf.zeta_fmt == 1) { uint16_t h; memcpy(&h, zp, 2); cur = h; }
            else { memcpy(&cur, zp, 4); cur >>= 8; }
            if (!compare(S.depth_func, (float)zbits, (float)cur)) return;
        }
    }
    /* texture stages + combiners */
    V4 reg[16];
    memset(reg, 0, sizeof reg);
    reg[R_V0] = d0;
    reg[R_V1] = d1;
    reg[R_FOG] = argb_to_v4(S.fog_color);
    for (int i = 0; i < 4; i++)
        if (g_samp[i].enabled)
            reg[R_T0 + i] = g_samp[i].cube ? sample_cube(i, t[i].x, t[i].y, t[i].z) : sample_stage(i, t[i].x, t[i].y, dt ? &dt[i] : NULL);
    reg[R_R0].w = reg[R_T0].w;                                       /* r0.a starts as t0.a */
    V4 col;
    if (g_has_combiners) col = run_combiners(reg);
    else {
        col = d0;
        if (g_samp[0].enabled) { V4 tx = reg[R_T0]; col.x *= tx.x; col.y *= tx.y; col.z *= tx.z; col.w *= tx.w; }
    }
    /* alpha test */
    if (S.alpha_test_enable && !compare(S.alpha_func, col.w * 255.0f, (float)(S.alpha_ref & 255))) return;
    uint32_t *p = g_surf.swizzled
                      ? (uint32_t *)(g_mem + g_surf.color_addr) + swz((uint32_t)x, (uint32_t)y, 1u << g_surf.swz_log2w, 1u << g_surf.swz_log2h)
                      : (uint32_t *)(g_mem + g_surf.color_addr + (uint32_t)y * g_surf.pitch) + x;
    uint32_t d = *p;
    if (S.blend_enable) {
        V4 dc = argb_to_v4(d), cc = argb_to_v4(S.blend_color);
        float sa = col.w, da = dc.w;
        V4 o;
        o.x = blend_eq(col.x * blend_factor(S.blend_src, sa, da, col.x, dc.x, cc.x, cc.w), dc.x * blend_factor(S.blend_dst, sa, da, col.x, dc.x, cc.x, cc.w));
        o.y = blend_eq(col.y * blend_factor(S.blend_src, sa, da, col.y, dc.y, cc.y, cc.w), dc.y * blend_factor(S.blend_dst, sa, da, col.y, dc.y, cc.y, cc.w));
        o.z = blend_eq(col.z * blend_factor(S.blend_src, sa, da, col.z, dc.z, cc.z, cc.w), dc.z * blend_factor(S.blend_dst, sa, da, col.z, dc.z, cc.z, cc.w));
        o.w = blend_eq(sa * blend_factor(S.blend_src, sa, da, sa, da, cc.w, cc.w), da * blend_factor(S.blend_dst, sa, da, sa, da, cc.w, cc.w));
        col = o;
    }
    uint32_t n = pack(col), m = 0;
    if (S.color_mask & 0x000000FF) m |= 0x000000FF;
    if (S.color_mask & 0x0000FF00) m |= 0x0000FF00;
    if (S.color_mask & 0x00FF0000) m |= 0x00FF0000;
    if (S.color_mask & 0xFF000000) m |= 0xFF000000;
    *p = (d & ~m) | (n & m);
    if (zp && S.depth_mask) {
        if (g_surf.zeta_fmt == 1) { uint16_t h = (uint16_t)zbits; memcpy(zp, &h, 2); }
        else { uint32_t cur; memcpy(&cur, zp, 4); cur = (cur & 0xFF) | zbits << 8; memcpy(zp, &cur, 4); }
    }
    (*npix)++;
}

typedef struct { OutVtx a, b, c; int minx, maxx, miny, maxy; float ia; } Tri;
#define MAX_TRIS 2048
static Tri g_tris[MAX_TRIS];
static int g_ntris;
static Tri *tri_alloc(void);
static int g_cull_invert = -1;
static uint64_t g_prim_allculled, g_prim_noneculled, g_prim_mixed;
static void raster_screen(const OutVtx *a, const OutVtx *b, const OutVtx *c) {
    if (!g_surf.color_addr || !g_surf.pitch) return;
    float area = (b->x - a->x) * (c->y - a->y) - (c->x - a->x) * (b->y - a->y);
    if (fabsf(area) < 1e-6f || area != area) { g_tri_degenerate++; return; }
    if (S.cull_enable) {
        /* screen y points down: positive area = clockwise on screen */
        if (g_cull_invert < 0) g_cull_invert = getenv("HALO_CULL_INVERT") ? 1 : 0;
        int cw = (area > 0) ^ g_cull_invert;
        int front = (S.front_face == 0x900) ? cw : !cw;
        if ((S.cull_face == 0x405 && !front) || (S.cull_face == 0x404 && front) || S.cull_face == 0x408) {
            g_tri_culled++; return;
        }
    }
    int minx = (int)floorf(fminf(a->x, fminf(b->x, c->x))), maxx = (int)ceilf(fmaxf(a->x, fmaxf(b->x, c->x)));
    int miny = (int)floorf(fminf(a->y, fminf(b->y, c->y))), maxy = (int)ceilf(fmaxf(a->y, fmaxf(b->y, c->y)));
    if (minx < (int)g_surf.clip_x) minx = (int)g_surf.clip_x;
    if (miny < (int)g_surf.clip_y) miny = (int)g_surf.clip_y;
    if (maxx > (int)(g_surf.clip_x + g_surf.clip_w)) maxx = (int)(g_surf.clip_x + g_surf.clip_w);
    if (maxy > (int)(g_surf.clip_y + g_surf.clip_h)) maxy = (int)(g_surf.clip_y + g_surf.clip_h);
    if (minx >= maxx || miny >= maxy) { g_tri_offscreen++; return; }
    S.tris++;
    Tri *t = tri_alloc();
    t->a = *a; t->b = *b; t->c = *c;
    t->minx = minx; t->maxx = maxx; t->miny = miny; t->maxy = maxy;
    t->ia = 1.0f / area;
}

/* Pixel loop for the rows of one triangle that belong to band k of nb.
 * Rows are dealt out to threads in groups of 4, so every pixel is always
 * written by the same thread, in primitive order: results do not depend on
 * the number of threads. */
#define BAND_SHIFT 2
static void tri_rows(const Tri *tr, int k, int nb, uint64_t *npix) {
    const OutVtx *a = &tr->a, *b = &tr->b, *c = &tr->c;
    float ia = tr->ia;
    float iwa = 1.0f / a->w, iwb = 1.0f / b->w, iwc = 1.0f / c->w;
    /* edge functions, affine in (px, py): w0 = (b-p)x(c-p), w1 = (c-p)x(a-p) */
    float e0x = -(c->y - b->y) * ia, e1x = -(a->y - c->y) * ia;
    float e0y = (c->x - b->x) * ia, e1y = (a->x - c->x) * ia;
    int y = tr->miny;
    if (nb > 1) {
        int band = (y >> BAND_SHIFT) % nb;
        if (band != k) y = ((y >> BAND_SHIFT) + ((k - band + nb) % nb)) << BAND_SHIFT;
    }
    for (; y < tr->maxy; ) {
        float py = (float)y + 0.5f;
        float px0 = (float)tr->minx + 0.5f;
        float w0 = ((b->x - px0) * (c->y - py) - (c->x - px0) * (b->y - py)) * ia;
        float w1 = ((c->x - px0) * (a->y - py) - (a->x - px0) * (c->y - py)) * ia;
        for (int x = tr->minx; x < tr->maxx; x++, w0 += e0x, w1 += e1x) {
            float w2 = 1.0f - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            float z = a->z * w0 + b->z * w1 + c->z * w2;                 /* screen-space linear */
            float p0 = w0 * iwa, p1 = w1 * iwb, p2 = w2 * iwc, ps = p0 + p1 + p2;
            if (ps != 0) { float r = 1.0f / ps; p0 *= r; p1 *= r; p2 *= r; }
            #define LERP4(f) (V4){a->f.x * p0 + b->f.x * p1 + c->f.x * p2, a->f.y * p0 + b->f.y * p1 + c->f.y * p2, \
                                  a->f.z * p0 + b->f.z * p1 + c->f.z * p2, a->f.w * p0 + b->f.w * p1 + c->f.w * p2}
            V4 d0 = LERP4(d0), d1 = LERP4(d1);
            V4 t[4];
            for (int q = 0; q < 4; q++)
                if (g_samp[q].enabled) t[q] = (V4){a->t[q].x * p0 + b->t[q].x * p1 + c->t[q].x * p2,
                                                   a->t[q].y * p0 + b->t[q].y * p1 + c->t[q].y * p2,
                                                   a->t[q].z * p0 + b->t[q].z * p1 + c->t[q].z * p2, 0};
            #undef LERP4
            V4 dt[4];
            if (g_need_deriv) {
                /* texture coordinates one pixel to the right and one below */
                float q0 = (w0 + e0x) * iwa, q1 = (w1 + e1x) * iwb, q2 = (1.0f - w0 - e0x - w1 - e1x) * iwc, qs = q0 + q1 + q2;
                float r0 = (w0 + e0y) * iwa, r1 = (w1 + e1y) * iwb, r2 = (1.0f - w0 - e0y - w1 - e1y) * iwc, rs = r0 + r1 + r2;
                if (qs != 0) { qs = 1.0f / qs; q0 *= qs; q1 *= qs; q2 *= qs; }
                if (rs != 0) { rs = 1.0f / rs; r0 *= rs; r1 *= rs; r2 *= rs; }
                for (int q = 0; q < 4; q++)
                    if (g_samp[q].enabled)
                        dt[q] = (V4){a->t[q].x * q0 + b->t[q].x * q1 + c->t[q].x * q2 - t[q].x,
                                     a->t[q].y * q0 + b->t[q].y * q1 + c->t[q].y * q2 - t[q].y,
                                     a->t[q].x * r0 + b->t[q].x * r1 + c->t[q].x * r2 - t[q].x,
                                     a->t[q].y * r0 + b->t[q].y * r1 + c->t[q].y * r2 - t[q].y};
            }
            shade_pixel(x, y, z, d0, d1, t, g_need_deriv ? dt : NULL, npix);
        }
        y++;
        if (nb > 1 && (y & ((1 << BAND_SHIFT) - 1)) == 0) y += (nb - 1) << BAND_SHIFT;
    }
}

/* ------------------------------------------------ host GPU hand-off */
static uint32_t guest_addr_of(const uint8_t *p) { return p ? (uint32_t)(p - g_mem) : 0; }

static void fill_draw_state(NvDrawState *st) {
    memset(st, 0, sizeof *st);
    st->color_addr = g_surf.color_addr;
    st->pitch = g_surf.pitch;
    st->width = g_surf.pitch / 4;
    st->height = g_surf.clip_y + g_surf.clip_h;
    st->clip_x = g_surf.clip_x; st->clip_y = g_surf.clip_y; st->clip_w = g_surf.clip_w; st->clip_h = g_surf.clip_h;
    st->zeta_addr = g_surf.zeta_addr; st->zeta_pitch = g_surf.zeta_pitch; st->zeta_fmt = g_surf.zeta_fmt;
    for (int i = 0; i < 4; i++) {
        const Sampler *sp = &g_samp[i];
        NvTexture *t = &st->tex[i];
        if (!sp->enabled) continue;
        t->enabled = 1;
        t->addr = guest_addr_of(sp->base);
        t->fmt = sp->fmt; t->w = sp->w; t->h = sp->h; t->pitch = sp->pitch; t->linear = sp->linear;
        t->levels = g_levels[i];
        t->palette_addr = guest_addr_of(sp->palette);
        uint32_t bytes = 0;
        if (sp->linear) bytes = sp->pitch * sp->h;
        else {
            const Sampler *last = &g_lvl[i][g_levels[i] - 1];
            uint32_t lb = (sp->fmt == 0x0C || sp->fmt == 0x0E || sp->fmt == 0x0F)
                              ? ((last->w + 3) / 4) * ((last->h + 3) / 4) * (sp->fmt == 0x0C ? 8u : 16u)
                              : last->w * last->h * tex_bytes(sp->fmt);
            bytes = (uint32_t)(last->base - sp->base) + lb;
        }
        t->cube = sp->cube;
        t->face_bytes = sp->face_bytes;
        if (sp->cube) bytes = 6 * sp->face_bytes;
        t->bytes = bytes;
        t->addr_u = sp->addr_u; t->addr_v = sp->addr_v;
        resolve_filter(i, &t->mag_linear, &t->min_linear, &t->mip, &t->max_aniso);
        tex_stat(S.tex_filter[i], sp->fmt, g_levels[i], sp->linear, sp->w, sp->h);
    }
    st->specfog[0] = S.specfog[0]; st->specfog[1] = S.specfog[1];
    st->comb_control = S.comb_control; st->final0 = S.final0; st->final1 = S.final1; st->fog_color = S.fog_color;
    st->fog_code = 0;
    if (S.fog_enable && (S.exec_mode & 3) == 2) {           /* vertex programs write the fog coordinate (oFog) */
        switch (S.fog_mode) {
        case 0x2601: st->fog_code = 1; break;               /* LINEAR */
        case 0x0804: st->fog_code = 5; break;               /* LINEAR_ABS */
        case 0x0800: st->fog_code = 2; break;               /* EXP */
        case 0x0802: st->fog_code = 6; break;               /* EXP_ABS */
        case 0x0801: st->fog_code = 3; break;               /* EXP2 */
        case 0x0803: st->fog_code = 7; break;               /* EXP2_ABS */
        default: break;
        }
    }
    {   /* HALO_NO_FOG=1 turns fog off (comparison); the first settings seen are logged */
        static int nofog = -1;
        if (nofog < 0) nofog = getenv("HALO_NO_FOG") != NULL;
        if (nofog) st->fog_code = 0;
        static struct { uint32_t mode, gen; float b, m; } seen[8];
        static int nseen;
        if (S.fog_enable && nseen < 8) {
            int k;
            for (k = 0; k < nseen; k++)
                if (seen[k].mode == S.fog_mode && seen[k].gen == S.fog_gen && seen[k].b == S.fog_params[0] && seen[k].m == S.fog_params[1]) break;
            if (k == nseen) {
                seen[nseen].mode = S.fog_mode; seen[nseen].gen = S.fog_gen; seen[nseen].b = S.fog_params[0]; seen[nseen].m = S.fog_params[1]; nseen++;
                runtime_log("[fog] mode %04x gen %x params %g %g %g, program mode %u -> code %u", S.fog_mode, S.fog_gen,
                            S.fog_params[0], S.fog_params[1], S.fog_params[2], S.exec_mode & 3, st->fog_code);
            }
        }
    }
    st->fog_bias = S.fog_params[0]; st->fog_mult = S.fog_params[1];
    st->poly_offset_fill = S.poly_offset_fill; st->poly_scale = S.poly_scale; st->poly_bias = S.poly_bias;
    st->stencil_enable = S.stencil_enable && g_surf.zeta_fmt == 2;   /* only Z24S8 has a stencil byte */
    st->stencil_mask = S.stencil_mask; st->stencil_func = S.stencil_func; st->stencil_ref = S.stencil_ref;
    st->stencil_func_mask = S.stencil_func_mask;
    st->stencil_op_fail = S.stencil_op_fail; st->stencil_op_zfail = S.stencil_op_zfail; st->stencil_op_zpass = S.stencil_op_zpass;
    memcpy(st->color_icw, S.color_icw, sizeof st->color_icw);
    memcpy(st->color_ocw, S.color_ocw, sizeof st->color_ocw);
    memcpy(st->alpha_icw, S.alpha_icw, sizeof st->alpha_icw);
    memcpy(st->alpha_ocw, S.alpha_ocw, sizeof st->alpha_ocw);
    memcpy(st->factor0, S.factor0, sizeof st->factor0);
    memcpy(st->factor1, S.factor1, sizeof st->factor1);
    st->blend_enable = S.blend_enable; st->blend_src = S.blend_src; st->blend_dst = S.blend_dst;
    st->blend_eq = S.blend_eq; st->blend_color = S.blend_color;
    st->alpha_test_enable = S.alpha_test_enable; st->alpha_func = S.alpha_func; st->alpha_ref = S.alpha_ref;
    st->depth_test = S.depth_test; st->depth_func = S.depth_func; st->depth_mask = S.depth_mask;
    st->color_mask = S.color_mask;
}

void nv2a_decode_texture_level(const NvTexture *t, int face, int level, uint32_t *out, uint32_t *out_w, uint32_t *out_h) {
    Sampler L;
    memset(&L, 0, sizeof L);
    L.fmt = t->fmt; L.linear = t->linear; L.pitch = t->pitch; L.enabled = 1;
    L.palette = t->palette_addr ? g_mem + t->palette_addr : NULL;
    uint32_t w = t->w, h = t->h, off = 0;
    for (int l = 0; l < level; l++) {
        if (t->fmt == 0x0C || t->fmt == 0x0E || t->fmt == 0x0F) off += ((w + 3) / 4) * ((h + 3) / 4) * (t->fmt == 0x0C ? 8u : 16u);
        else off += w * h * tex_bytes(t->fmt);
        if (w > 1) w >>= 1;
        if (h > 1) h >>= 1;
    }
    L.base = g_mem + t->addr + (t->cube ? (uint32_t)face * t->face_bytes : 0) + off;
    L.w = w; L.h = h;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) out[y * w + x] = fetch(&L, x, y);
    *out_w = w; *out_h = h;
}

static OutVtx g_gpu_verts[MAX_TRIS * 3];
static int g_gpu_nverts;
static void gpu_flush(void) {
    if (!g_gpu_nverts) return;
    NvDrawState st;
    fill_draw_state(&st);
    g_gpu_backend->draw(&st, g_gpu_verts, g_gpu_nverts);
    g_gpu_nverts = 0;
}

void nv2a_raster_gpu_sync(void) { if (g_gpu_backend) gpu_flush(); }

/* ------------------------------------------------ parallel triangle drawing */
#include <pthread.h>
#include <unistd.h>
#define MAX_WORKERS 15
static int g_nthreads = 0;                      /* total bands, including the calling thread */
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pool_go = PTHREAD_COND_INITIALIZER, g_pool_done = PTHREAD_COND_INITIALIZER;
static unsigned g_pool_gen;
static int g_pool_pending;
static uint64_t g_pool_pix[MAX_WORKERS + 1];

static void draw_band(int k) {
    uint64_t n = 0;
    for (int i = 0; i < g_ntris; i++) tri_rows(&g_tris[i], k, g_nthreads, &n);
    g_pool_pix[k] += n;
}

static void *pool_worker(void *arg) {
    int k = (int)(intptr_t)arg;
    unsigned seen = 0;
    for (;;) {
        pthread_mutex_lock(&g_pool_mu);
        while (g_pool_gen == seen) pthread_cond_wait(&g_pool_go, &g_pool_mu);
        seen = g_pool_gen;
        pthread_mutex_unlock(&g_pool_mu);
        draw_band(k);
        pthread_mutex_lock(&g_pool_mu);
        if (--g_pool_pending == 0) pthread_cond_signal(&g_pool_done);
        pthread_mutex_unlock(&g_pool_mu);
    }
    return NULL;
}

static void pool_init(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    const char *env = getenv("HALO_RASTER_THREADS");
    if (env) n = atol(env);
    if (n < 1) n = 1;
    if (n > MAX_WORKERS + 1) n = MAX_WORKERS + 1;
    g_nthreads = (int)n;
    for (int k = 1; k < g_nthreads; k++) {
        pthread_t t;
        pthread_create(&t, NULL, pool_worker, (void *)(intptr_t)k);
        pthread_detach(t);
    }
    runtime_log("[raster] drawing with %d threads", g_nthreads);
}

static void draw_tris(void) {
    if (g_gpu_backend) gpu_flush();
    if (!g_ntris) return;
    if (!g_nthreads) pool_init();
    uint64_t area = 0;
    for (int i = 0; i < g_ntris; i++)
        area += (uint64_t)(g_tris[i].maxx - g_tris[i].minx) * (uint64_t)(g_tris[i].maxy - g_tris[i].miny);
    if (g_nthreads == 1 || area < 2048) {
        uint64_t n = 0;
        for (int i = 0; i < g_ntris; i++) tri_rows(&g_tris[i], 0, 1, &n);
        S.pixels += n;
    } else {
        memset(g_pool_pix, 0, sizeof g_pool_pix);
        pthread_mutex_lock(&g_pool_mu);
        g_pool_pending = g_nthreads - 1;
        g_pool_gen++;
        pthread_cond_broadcast(&g_pool_go);
        pthread_mutex_unlock(&g_pool_mu);
        draw_band(0);
        pthread_mutex_lock(&g_pool_mu);
        while (g_pool_pending) pthread_cond_wait(&g_pool_done, &g_pool_mu);
        pthread_mutex_unlock(&g_pool_mu);
        for (int k = 0; k < g_nthreads; k++) S.pixels += g_pool_pix[k];
    }
    g_ntris = 0;
}

static Tri *tri_alloc(void) {
    if (g_ntris == MAX_TRIS) draw_tris();
    return &g_tris[g_ntris++];
}

/* Near-plane clipping. Xbox vertex shaders do the perspective divide
 * themselves (oPos = clip.xyz / w * scale + offset, oPos.w = w), so a vertex
 * behind the eye (w <= 0) lands at a mirrored, meaningless screen position.
 * (screen * w, w) is linear in clip space, so clip there against w = W_NEAR and
 * divide again. Attributes are linear in clip space as well. */
#define W_NEAR 1e-4f
static uint64_t g_tri_clipped, g_tri_behind;
static OutVtx clip_lerp(const OutVtx *a, const OutVtx *b, float t) {
    OutVtx r;
    float hx = a->x * a->w + (b->x * b->w - a->x * a->w) * t;
    float hy = a->y * a->w + (b->y * b->w - a->y * a->w) * t;
    float hz = a->z * a->w + (b->z * b->w - a->z * a->w) * t;
    float w = a->w + (b->w - a->w) * t;
    if (w < W_NEAR) w = W_NEAR;
    r.x = hx / w; r.y = hy / w; r.z = hz / w; r.w = w;
    #define L4(f) (V4){a->f.x + (b->f.x - a->f.x) * t, a->f.y + (b->f.y - a->f.y) * t, \
                       a->f.z + (b->f.z - a->f.z) * t, a->f.w + (b->f.w - a->f.w) * t}
    r.d0 = L4(d0); r.d1 = L4(d1);
    for (int k = 0; k < 4; k++) r.t[k] = L4(t[k]);
    #undef L4
    return r;
}

/* GPU path: no CPU clipping. The triangle goes to the GPU in homogeneous form
 * (screen position x w, w) and the GPU clips it against w > 0 itself, which
 * keeps vertex values well-conditioned. Back faces are culled with the
 * homogeneous determinant, whose sign equals the screen-space winding times
 * sign(wa wb wc) and so stays correct for triangles that cross the camera plane. */
static void gpu_triangle(const OutVtx *a, const OutVtx *b, const OutVtx *c) {
    if (!g_surf.color_addr || !g_surf.pitch) return;
    if (a->w <= 0 && b->w <= 0 && c->w <= 0) { g_tri_behind++; return; }
    double ax = (double)a->x * a->w, ay = (double)a->y * a->w, bx = (double)b->x * b->w, by = (double)b->y * b->w;
    double cx = (double)c->x * c->w, cy = (double)c->y * c->w;
    double det = ax * (by * c->w - cy * b->w) - ay * (bx * c->w - cx * b->w) + a->w * (bx * cy - cx * by);
    if (det == 0 || det != det) { g_tri_degenerate++; return; }
    if (S.cull_enable) {
        if (g_cull_invert < 0) g_cull_invert = getenv("HALO_CULL_INVERT") ? 1 : 0;
        int cw = (det > 0) ^ g_cull_invert;
        int front = (S.front_face == 0x900) ? cw : !cw;
        if ((S.cull_face == 0x405 && !front) || (S.cull_face == 0x404 && front) || S.cull_face == 0x408) {
            g_tri_culled++; return;
        }
    }
    S.tris++;
    if (g_gpu_nverts + 3 > MAX_TRIS * 3) gpu_flush();
    g_gpu_verts[g_gpu_nverts++] = *a; g_gpu_verts[g_gpu_nverts++] = *b; g_gpu_verts[g_gpu_nverts++] = *c;
}

static void raster_triangle(const OutVtx *a, const OutVtx *b, const OutVtx *c) {
    if (g_gpu_backend) {
        gpu_triangle(a, b, c);
        if (!g_gpu_cpu_shadow) return;           /* diagnosis: also draw on the CPU */
    }
    static int noclip = -1;
    if (noclip < 0) noclip = getenv("HALO_NO_CLIP") != NULL;
    if (noclip) { raster_screen(a, b, c); return; }
    int ina = a->w >= W_NEAR, inb = b->w >= W_NEAR, inc = c->w >= W_NEAR;
    if (ina && inb && inc) { raster_screen(a, b, c); return; }
    if (!ina && !inb && !inc) { g_tri_behind++; return; }
    g_tri_clipped++;
    const OutVtx *in[3] = {a, b, c};
    OutVtx out[4];
    int n = 0;
    for (int i = 0; i < 3; i++) {
        const OutVtx *p = in[i], *q = in[(i + 1) % 3];
        int pin = p->w >= W_NEAR, qin = q->w >= W_NEAR;
        if (pin) out[n++] = *p;
        if (pin != qin) out[n++] = clip_lerp(p, q, (W_NEAR - p->w) / (q->w - p->w));
    }
    for (int i = 1; i + 1 < n; i++) raster_screen(&out[0], &out[i], &out[i + 1]);
}

static uint64_t g_raster_ns;
static int g_debug_prims = -1;
void nv2a_raster_debug_dump(void) {
    static uint64_t last_wall, last_raster;
    uint64_t now = host_time_ns();
    if (last_wall) runtime_log("[raster] time in triangle drawing: %.1f%% of wall time since last report",
                               100.0 * (double)(g_raster_ns - last_raster) / (double)(now - last_wall));
    last_wall = now; last_raster = g_raster_ns;
    runtime_log("[raster] prims: fixed %llu, program %llu | empty %llu, no surface %llu | tris degenerate %llu, offscreen %llu, culled %llu, near-clipped %llu, behind eye %llu | cull-enabled prims: all culled %llu, none %llu, mixed %llu",
                (unsigned long long)g_prim_by_mode[0], (unsigned long long)g_prim_by_mode[2],
                (unsigned long long)g_prim_novert, (unsigned long long)g_prim_nosurf,
                (unsigned long long)g_tri_degenerate, (unsigned long long)g_tri_offscreen, (unsigned long long)g_tri_culled, (unsigned long long)g_tri_clipped, (unsigned long long)g_tri_behind,
                (unsigned long long)g_prim_allculled, (unsigned long long)g_prim_noneculled, (unsigned long long)g_prim_mixed);
}

static void flush_primitive_body(void);
static void flush_primitive(void) {
    uint64_t t = host_time_ns();
    flush_primitive_body();
    g_raster_ns += host_time_ns() - t;
}
static void vs_flush(void);
static void flush_primitive_body(void) {
    if (g_vs_mode) { setup_samplers(); vs_flush(); g_vs_mode = 0; return; }
    int n = g_nv;
    const OutVtx *v = g_vbuf;
    g_prim_by_mode[S.exec_mode & 3]++;
    if (n == 0) g_prim_novert++;
    if (!g_surf.color_addr) g_prim_nosurf++;
    if (g_debug_prims < 0) g_debug_prims = getenv("HALO_DEBUG_PRIMS") ? atoi(getenv("HALO_DEBUG_PRIMS")) : 0;
    if (g_debug_prims > 0) {
        g_debug_prims--;
        runtime_log("[raster] prim %u, %d verts, mode %x, stages %u, texprog %05x", S.prim, n, S.exec_mode,
                    S.comb_control & 0xFF, S.shader_stage_prog);
        for (int i = 0; i < n && i < 3; i++)
            runtime_log("   v%d pos %.1f %.1f %.1f w %.3f d0 %.2f %.2f %.2f %.2f t0 %.3f %.3f", i, v[i].x, v[i].y, v[i].z, v[i].w,
                        v[i].d0.x, v[i].d0.y, v[i].d0.z, v[i].d0.w, v[i].t[0].x, v[i].t[0].y);
    }
    setup_samplers();
    uint64_t dbg_c0 = g_tri_culled, dbg_b0 = g_tri_behind, dbg_d0 = g_tri_degenerate, dbg_t0 = S.tris, dbg_o0 = g_tri_offscreen, dbg_k0 = g_tri_clipped;
    {   /* HALO_DEBUG_FRAME=n: log every draw of the n-th frame (buffer flip) */
        static long want = -2;
        if (want == -2) want = getenv("HALO_DEBUG_FRAME") ? atol(getenv("HALO_DEBUG_FRAME")) : -1;
        if (want >= 0 && (long)nv2a_stat_flips() == want) {
            static int idx;
            runtime_log("[frame] #%d -> %08x prim %u n %d | prog %05x comb %u final %08x %08x | atest %u %x %u | blend %u %x %x %x | depth %u %x mask %u | cmask %08x | t0 %08x/%02x t1 %08x t2 %08x t3 %08x",
                        idx++, g_surf.color_addr, S.prim, g_nv, S.shader_stage_prog, S.comb_control & 0xFF, S.final0, S.final1,
                        S.alpha_test_enable, S.alpha_func, S.alpha_ref, S.blend_enable, S.blend_src, S.blend_dst, S.blend_eq,
                        S.depth_test, S.depth_func, S.depth_mask, S.color_mask,
                        (S.tex_control0[0] >> 30) & 1 ? S.tex_offset[0] : 0, (S.tex_format[0] >> 8) & 0xFF,
                        (S.tex_control0[1] >> 30) & 1 ? S.tex_offset[1] : 0, (S.tex_control0[2] >> 30) & 1 ? S.tex_offset[2] : 0,
                        (S.tex_control0[3] >> 30) & 1 ? S.tex_offset[3] : 0);
            runtime_log("[frame]    comb_control %08x tex filters %08x %08x %08x %08x addr %08x %08x %08x %08x", S.comb_control,
                        S.tex_filter[0], S.tex_filter[1], S.tex_filter[2], S.tex_filter[3],
                        S.tex_address[0], S.tex_address[1], S.tex_address[2], S.tex_address[3]);
            for (uint32_t st = 0; st < (S.comb_control & 0xFF) && st < 8; st++)
                runtime_log("[frame]    stage %u: color icw %08x ocw %08x | alpha icw %08x ocw %08x | f0 %08x f1 %08x",
                            st, S.color_icw[st], S.color_ocw[st], S.alpha_icw[st], S.alpha_ocw[st], S.factor0[st], S.factor1[st]);
        }
    }
    {   /* HALO_DEBUG_STAGES: which texture-stage programs the game uses */
        static int dbg = -1; static uint32_t key[64], fmt[64]; static uint64_t cnt[64]; static int nk; static uint64_t total;
        if (dbg < 0) dbg = getenv("HALO_DEBUG_STAGES") ? 1 : 0;
        if (dbg) {
            uint32_t k = S.shader_stage_prog; int j;
            for (j = 0; j < nk; j++) if (key[j] == k) break;
            if (j == nk && nk < 64) { key[nk] = k; cnt[nk] = 0; nk++;
                fmt[j] = 0;
                for (int i = 0; i < 4; i++) fmt[j] |= ((S.tex_format[i] >> 8) & 0xFF) << (8 * i);
                runtime_log("[stages] new program %05x (modes %u %u %u %u) formats %08x tex_format0 %08x c0ctl %08x comb %u",
                            k, k & 31, k >> 5 & 31, k >> 10 & 31, k >> 15 & 31, fmt[j], S.tex_format[0], S.tex_control0[0], S.comb_control & 0xFF);
                runtime_log("[stages]   alpha test %u func %x ref %u | blend %u %x %x | depth %u %x mask %u | comb_control %08x final %08x %08x | texfmt %08x %08x %08x %08x",
                            S.alpha_test_enable, S.alpha_func, S.alpha_ref, S.blend_enable, S.blend_src, S.blend_dst, S.depth_test,
                            S.depth_func, S.depth_mask, S.comb_control, S.final0, S.final1, S.tex_format[0], S.tex_format[1], S.tex_format[2], S.tex_format[3]);
                for (uint32_t st = 0; st < (S.comb_control & 0xFF) && st < 8; st++)
                    runtime_log("[stages]   stage %u: color icw %08x ocw %08x | alpha icw %08x ocw %08x | f0 %08x f1 %08x",
                                st, S.color_icw[st], S.color_ocw[st], S.alpha_icw[st], S.alpha_ocw[st], S.factor0[st], S.factor1[st]); }
            if (j < nk) cnt[j]++;
            if (++total % 20000 == 0)
                for (int i = 0; i < nk; i++) runtime_log("[stages] %05x: %llu draws", key[i], (unsigned long long)cnt[i]);
        }
    }
    if (g_tex_mode < 0) g_tex_mode = getenv("HALO_TEX_FILTER") ? atoi(getenv("HALO_TEX_FILTER")) : 0;
    g_need_deriv = g_tex_mode != 1 && (g_samp[0].enabled || g_samp[1].enabled || g_samp[2].enabled || g_samp[3].enabled);
    uint64_t culled0 = g_tri_culled, drawn0 = S.tris;
    g_has_combiners = (S.comb_control & 0xFF) != 0;
    switch (S.prim) {
    case 5: for (int i = 0; i + 2 < n; i += 3) raster_triangle(&v[i], &v[i + 1], &v[i + 2]); break;
    case 6: for (int i = 0; i + 2 < n; i++) {                                                  /* strip: alternate winding */
                if (i & 1) raster_triangle(&v[i + 1], &v[i], &v[i + 2]);
                else raster_triangle(&v[i], &v[i + 1], &v[i + 2]); }
            break;
    case 7: case 10: for (int i = 1; i + 1 < n; i++) raster_triangle(&v[0], &v[i], &v[i + 1]); break;
    case 8: for (int i = 0; i + 3 < n; i += 4) {
                raster_triangle(&v[i], &v[i + 1], &v[i + 2]); raster_triangle(&v[i], &v[i + 2], &v[i + 3]); }
            break;
    case 9: for (int i = 0; i + 3 < n; i += 2) {
                raster_triangle(&v[i], &v[i + 1], &v[i + 3]); raster_triangle(&v[i], &v[i + 3], &v[i + 2]); }
            break;
    default: break;   /* points/lines: TODO */
    }
    draw_tris();
    static int dbg_ring = -1;
    if (dbg_ring < 0) dbg_ring = getenv("HALO_DEBUG_RING") != NULL;
    if (dbg_ring && (n == 290 || n == 1456)) {
        int bad = 0, badw = 0;
        for (int i = 0; i < n; i++) {
            const OutVtx *q = &g_vbuf[i];
            int nf = 0;
            for (int k = 0; k < 4; k++) nf |= !isfinite(q->t[k].x) || !isfinite(q->t[k].y) || !isfinite(q->t[k].z) || !isfinite(q->t[k].w);
            nf |= !isfinite(q->d0.x) || !isfinite(q->d0.w) || !isfinite(q->d1.x) || !isfinite(q->d1.w);
            if (nf) { bad++; static int shown; if (shown++ < 6) runtime_log("[ring]   vertex %d: pos %g %g %g w %g | t0 %g %g %g %g | t1 %g %g | t2 %g %g | t3 %g %g %g | d0 %g %g %g %g d1 %g %g %g %g",
                i, q->x, q->y, q->z, q->w, q->t[0].x, q->t[0].y, q->t[0].z, q->t[0].w, q->t[1].x, q->t[1].y, q->t[2].x, q->t[2].y,
                q->t[3].x, q->t[3].y, q->t[3].z, q->d0.x, q->d0.y, q->d0.z, q->d0.w, q->d1.x, q->d1.y, q->d1.z, q->d1.w); }
            if (!isfinite(q->x) || !isfinite(q->y) || !isfinite(q->z) || !isfinite(q->w)) badw++;
        }
        static int k2; if (k2++ < 6) runtime_log("[ring]   %d of %d vertices have non-finite attributes, %d non-finite positions", bad, n, badw);
        static int k;
        if (k++ < 6)
            runtime_log("[ring] n %d prim %u cull %u face %x front %x: drawn %llu culled %llu behind %llu degenerate %llu offscreen %llu nearclipped %llu (%s)",
                        n, S.prim, S.cull_enable, S.cull_face, S.front_face, (unsigned long long)(S.tris - dbg_t0),
                        (unsigned long long)(g_tri_culled - dbg_c0), (unsigned long long)(g_tri_behind - dbg_b0),
                        (unsigned long long)(g_tri_degenerate - dbg_d0), (unsigned long long)(g_tri_offscreen - dbg_o0),
                        (unsigned long long)(g_tri_clipped - dbg_k0), g_gpu_backend ? "gpu path" : "cpu path");
    }
    if (S.cull_enable && n >= 3) {
        uint64_t c = g_tri_culled - culled0, d = S.tris - drawn0;
        if (c && !d) g_prim_allculled++; else if (d && !c) g_prim_noneculled++; else if (c && d) g_prim_mixed++;
    }
    g_nv = 0;
}

/* ===================================================== clears */
void nv2a_raster_clear(uint32_t flags, uint32_t color, uint32_t zstencil, uint32_t rect_h, uint32_t rect_v) {
    if (g_npend) resolve_pending();
    uint32_t x0 = rect_h & 0xFFFF, x1 = rect_h >> 16, y0 = rect_v & 0xFFFF, y1 = rect_v >> 16;
    if (x1 < x0 || y1 < y0) return;
    uint32_t cw = g_surf.clip_x + g_surf.clip_w, ch = g_surf.clip_y + g_surf.clip_h;
    if (x1 >= cw) x1 = cw ? cw - 1 : 0;
    if (y1 >= ch) y1 = ch ? ch - 1 : 0;
    if (g_gpu_backend) {
        if (!g_surf.color_addr) return;
        NvDrawState st;
        memset(g_samp, 0, sizeof g_samp);            /* no textures needed */
        fill_draw_state(&st);
        g_gpu_backend->clear(&st, flags, color, zstencil, x0, y0, x1, y1);
        if (!g_gpu_cpu_shadow) return;
    }
    if ((flags & 0xF0) && g_surf.color_addr) {
        uint32_t m = 0;
        if (flags & 0x10) m |= 0x00FF0000; if (flags & 0x20) m |= 0x0000FF00;
        if (flags & 0x40) m |= 0x000000FF; if (flags & 0x80) m |= 0xFF000000;
        for (uint32_t y = y0; y <= y1; y++) {
            uint32_t *row = (uint32_t *)(g_mem + g_surf.color_addr + y * g_surf.pitch);
            for (uint32_t x = x0; x <= x1; x++) {
                uint32_t *px = g_surf.swizzled ? (uint32_t *)(g_mem + g_surf.color_addr) +
                                                     swz(x, y, 1u << g_surf.swz_log2w, 1u << g_surf.swz_log2h)
                                               : &row[x];
                *px = (*px & ~m) | (color & m);
            }
        }
    }
    if ((flags & 3) && g_surf.zeta_addr) {
        for (uint32_t y = y0; y <= y1; y++) {
            uint8_t *row = g_mem + g_surf.zeta_addr + y * g_surf.zeta_pitch;
            for (uint32_t x = x0; x <= x1; x++) {
                if (g_surf.zeta_fmt == 1) { if (flags & 1) { uint16_t h = (uint16_t)zstencil; memcpy(row + x * 2, &h, 2); } }
                else {
                    uint32_t cur; memcpy(&cur, row + x * 4, 4);
                    if (flags & 1) cur = (cur & 0xFF) | (zstencil & 0xFFFFFF00u);
                    if (flags & 2) cur = (cur & ~0xFFu) | (zstencil & 0xFF);
                    memcpy(row + x * 4, &cur, 4);
                }
            }
        }
    }
}

/* ===================================================== vertex arrays */
static inline float s10(uint32_t v, int bits) {        /* signed normalized from packed field */
    int32_t s = (int32_t)(v << (32 - bits)) >> (32 - bits);
    float d = (float)((1 << (bits - 1)) - 1);
    float f = (float)s / d;
    return f < -1 ? -1 : f;
}

static void array_bases(uint32_t *base) {
    for (int i = 0; i < NATTR; i++) {
        uint32_t off = S.array_offset[i];
        base[i] = ((S.array_format[i] >> 4) & 15) ? dma_base_cached((off & 0x80000000u) ? g_dma_b : g_dma_a) + (off & 0x7FFFFFFF) : 0;
    }
}
static void fetch_array_vertex_b(uint32_t index, V4 *in, const uint32_t *base) {
    for (int i = 0; i < NATTR; i++) {
        in[i] = S.attr[i];
        uint32_t f = S.array_format[i], type = f & 15, size = (f >> 4) & 15, stride = f >> 8;
        if (!size) continue;
        const uint8_t *p = phys_ptr(base[i] + index * stride);
        float c[4] = {0, 0, 0, 1};
        if (type == 6) {                                         /* CMP: 11:11:10 signed normalized */
            uint32_t v; memcpy(&v, p, 4);
            c[0] = s10(v, 11); c[1] = s10(v >> 11, 11); c[2] = s10(v >> 22, 10);
        } else {
            for (uint32_t k = 0; k < size && k < 4; k++) {
                switch (type) {
                case 2: memcpy(&c[k], p + 4 * k, 4); break;                                          /* F */
                case 1: { int16_t s; memcpy(&s, p + 2 * k, 2); c[k] = (float)s / 32767.0f; break; }  /* S1 */
                case 5: { int16_t s; memcpy(&s, p + 2 * k, 2); c[k] = (float)s; break; }             /* S32K */
                case 4: c[k] = (float)p[k] / 255.0f; break;                                          /* UB_OGL */
                case 0: c[k] = (float)p[k == 3 ? 3 : 2 - k] / 255.0f; break;                         /* UB_D3D (BGRA) */
                default: break;
                }
            }
            if (type == 0 && size < 4) c[3] = 1;
        }
        in[i] = (V4){c[0], c[1], c[2], c[3]};
    }
}

static void fetch_array_vertex(uint32_t index, V4 *in) {
    uint32_t base[NATTR];
    array_bases(base);
    fetch_array_vertex_b(index, in, base);
}

/* Post-transform cache for indexed vertices: within one begin/end, the same
 * index always transforms to the same result, so strips and lists that
 * share vertices run the program once per vertex instead of once per use. */
#define VCACHE 4096
static struct { uint32_t idx, gen; int32_t wk; OutVtx v; } g_vcache[VCACHE];
static uint32_t g_vcache_gen = 1;

static int g_reuse_on = -1, g_reuse_cur;
static uint64_t g_reuse_keys[4096];
static int g_reuse_nkeys;
static uint64_t g_reuse_draws, g_reuse_rep_draws, g_reuse_runs, g_reuse_rep_runs;
static uint64_t g_st_hit, g_st_miss, g_st_batches, g_st_par;

/* Indexed vertices are queued while a draw is described (only index words
 * arrive between begin and end) and transformed together in
 * resolve_pending(), which runs before any other method, so the result is
 * the same as transforming each one on arrival. Unique vertices are spread
 * over a small worker pool. */
static int32_t g_pend_pos[MAXV], g_pend_wk[MAXV];
static uint32_t g_work_idx[MAXV];
static OutVtx g_work_out[MAXV];
static int g_nwork;
static uint32_t g_work_base[NATTR];

static void transform_range(int lo, int hi) {
    if (g_xf_time < 0) g_xf_time = getenv("HALO_VSTATS") != NULL;
    uint64_t t0 = g_xf_time ? host_time_ns() : 0;
    V4 in[NATTR], o[16];
    for (int k = lo; k < hi; k++) {
        fetch_array_vertex_b(g_work_idx[k], in, g_work_base);
        compute_outputs(in, o);
        pack_out(o, &g_work_out[k]);
    }
    if (g_xf_time) { __atomic_add_fetch(&g_xf_ns, host_time_ns() - t0, __ATOMIC_RELAXED); __atomic_add_fetch(&g_xf_n, (uint64_t)(hi - lo), __ATOMIC_RELAXED); }
}

/* ---- worker pool
 * Most draws have a few hundred unique vertices (tens of microseconds of
 * work), so waking sleeping threads would cost more than it saves. Workers
 * spin for a short while after each job and only sleep when the GPU is idle;
 * the caller hands out chunks through an atomic counter and waits until all
 * of them are done. */
static int g_vpool_n = -1;
static pthread_mutex_t g_vpool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_vpool_go = PTHREAD_COND_INITIALIZER;
static unsigned g_vpool_seq;
static int g_vpool_sleepers, g_vpool_total, g_vpool_done;
static uint64_t g_vpool_ctr;            /* job number << 32 | next item: a worker only claims from the job it joined */
#define VPOOL_CHUNK 32
#define VPOOL_SPIN_NS 300000ull
static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause");
#endif
}
static void vpool_run_chunks(unsigned job) {
    for (;;) {
        uint64_t v = __atomic_load_n(&g_vpool_ctr, __ATOMIC_ACQUIRE);
        if ((unsigned)(v >> 32) != job) break;
        int k = (int)(uint32_t)v, total = __atomic_load_n(&g_vpool_total, __ATOMIC_ACQUIRE);
        if (k >= total) break;
        if (!__atomic_compare_exchange_n(&g_vpool_ctr, &v, v + VPOOL_CHUNK, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
        int hi = k + VPOOL_CHUNK < total ? k + VPOOL_CHUNK : total;
        transform_range(k, hi);
        __atomic_fetch_add(&g_vpool_done, hi - k, __ATOMIC_RELEASE);
    }
}
static void *vpool_main(void *arg) {
    (void)arg;
    unsigned seen = __atomic_load_n(&g_vpool_seq, __ATOMIC_ACQUIRE);
    for (;;) {
        uint64_t t0 = host_time_ns();
        for (int spins = 0; __atomic_load_n(&g_vpool_seq, __ATOMIC_SEQ_CST) == seen; spins++) {
            cpu_relax();
            if ((spins & 255) == 255 && host_time_ns() - t0 > VPOOL_SPIN_NS) break;
        }
        if (__atomic_load_n(&g_vpool_seq, __ATOMIC_SEQ_CST) == seen) {
            pthread_mutex_lock(&g_vpool_mu);
            __atomic_fetch_add(&g_vpool_sleepers, 1, __ATOMIC_SEQ_CST);
            while (__atomic_load_n(&g_vpool_seq, __ATOMIC_SEQ_CST) == seen) pthread_cond_wait(&g_vpool_go, &g_vpool_mu);
            __atomic_fetch_sub(&g_vpool_sleepers, 1, __ATOMIC_SEQ_CST);
            pthread_mutex_unlock(&g_vpool_mu);
        }
        seen = __atomic_load_n(&g_vpool_seq, __ATOMIC_SEQ_CST);
        vpool_run_chunks(seen);
    }
    return NULL;
}
static void vpool_init(void) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int n = (int)(ncpu / 2);           /* spinning workers: leave room for the game and Metal */
    if (n > 5) n = 5;
    const char *e = getenv("HALO_VERTEX_THREADS");
    if (e) n = atoi(e);
    if (n < 0) n = 0;
    for (int i = 0; i < n; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, vpool_main, NULL) != 0) { n = i; break; }
        pthread_detach(t);
    }
    g_vpool_n = n;
    runtime_log("[raster] %d vertex worker threads", n);
}
static void transform_work(int n) {
    if (g_vpool_n < 0) vpool_init();
    if (g_vpool_n == 0 || n < 2 * VPOOL_CHUNK) { transform_range(0, n); return; }
    g_st_par++;
    /* job data (g_work_*, S, g_vp) is complete; publish the size, then open the counter */
    unsigned job = __atomic_load_n(&g_vpool_seq, __ATOMIC_RELAXED) + 1;
    __atomic_store_n(&g_vpool_done, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_vpool_total, n, __ATOMIC_RELEASE);
    __atomic_store_n(&g_vpool_ctr, (uint64_t)job << 32, __ATOMIC_SEQ_CST);
    __atomic_store_n(&g_vpool_seq, job, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(&g_vpool_sleepers, __ATOMIC_SEQ_CST)) {
        pthread_mutex_lock(&g_vpool_mu);
        pthread_cond_broadcast(&g_vpool_go);
        pthread_mutex_unlock(&g_vpool_mu);
    }
    vpool_run_chunks(job);
    while (__atomic_load_n(&g_vpool_done, __ATOMIC_ACQUIRE) < n) cpu_relax();
}

static void resolve_pending(void) {
    int np = g_npend;
    g_npend = 0;
    if (!np) return;
    g_st_batches++;
    int n = g_nwork;
    g_nwork = 0;
    if (n) {
        if (g_reuse_on > 0) { g_reuse_runs += (uint64_t)n; if (g_reuse_cur) g_reuse_rep_runs += (uint64_t)n; }
        if ((S.exec_mode & 3) == 2) { if (g_vp_dirty) decode_program(); g_vp_runs += (uint64_t)n; g_vp_ins += (uint64_t)n * (uint64_t)g_vp_len; }
        array_bases(g_work_base);
        {   /* HALO_VS_TEST: dump new programs with sample inputs (diagnosis) */
            static int test = -1;
            if (test < 0) test = getenv("HALO_VS_TEST") != NULL;
            if (test && (S.exec_mode & 3) == 2) {
                V4 tin[8 * 16];
                int m = n < 8 ? n : 8;
                for (int k = 0; k < m; k++) fetch_array_vertex_b(g_work_idx[k], &tin[k * 16], g_work_base);
                vs_test_dump(tin, m);
            }
        }
        transform_work(n);
    }
    for (int i = 0; i < np; i++) g_vbuf[g_pend_pos[i]] = g_work_out[g_pend_wk[i]];
    for (int k = 0; k < n; k++) {
        uint32_t slot = g_work_idx[k] & (VCACHE - 1);
        if (g_vcache[slot].wk == k && g_vcache[slot].gen == g_vcache_gen) { g_vcache[slot].v = g_work_out[k]; g_vcache[slot].wk = -1; }
    }
}

/* ---- GPU vertex mode: while a draw is described, unique vertex inputs are
 * collected (16 per vertex) with a triangle index list; at the end the backend
 * runs the program on the GPU (draw_vs). If it cannot yet (shader compiling),
 * the collected vertices go through the CPU transform instead.
 * HALO_GPU_VS=0 turns this off. */
static int g_vs_enabled = -1, g_vs_n;
static V4 g_vs_in[MAXV * 16];
static uint32_t g_vs_idx[MAXV], g_vs_tri[MAXV * 3];
static uint32_t g_vs_base[NATTR];
static uint64_t g_st_vs_draws, g_st_vs_fallback;
static const char *vs_source_cb(void) {
    static char body[1 << 19], full[(1 << 19) + 16384];
    SBuf b = {body, 0, sizeof body};
    if (!vs_generate(&b) || b.n >= b.cap) return NULL;
    SBuf f = {full, 0, sizeof full};
    vs_msl(&f, body);
    return f.n < f.cap ? full : NULL;
}
static void vs_begin(void) {
    if (g_vs_enabled < 0) { const char *e = getenv("HALO_GPU_VS"); g_vs_enabled = !(e && *e == '0'); }
    g_vs_mode = 0;
    g_vs_n = 0;
    if (!g_vs_enabled || !g_gpu_backend || !g_gpu_backend->draw_vs) return;
    if ((S.exec_mode & 3) != 2 || g_gpu_cpu_shadow || g_dbg_left > 0 || S.prim < 5 || S.prim > 10 || !g_surf.color_addr) return;
    if (g_vp_dirty) decode_program();
    if (g_vp_writes_consts) return;
    if (!g_vp_hash_valid) { g_vp_hash = vs_hash(); g_vp_hash_valid = 1; }
    array_bases(g_vs_base);
    g_vs_mode = 1;
}
static void vs_add_index(uint32_t index) {
    if (g_nv >= MAXV) return;
    uint32_t slot = index & (VCACHE - 1);
    uint32_t v;
    if (g_vcache[slot].gen == g_vcache_gen && g_vcache[slot].idx == index) { v = (uint32_t)g_vcache[slot].wk; g_st_hit++; }
    else {
        if (g_vs_n >= MAXV) return;
        v = (uint32_t)g_vs_n++;
        fetch_array_vertex_b(index, &g_vs_in[v * 16], g_vs_base);
        g_vcache[slot].idx = index; g_vcache[slot].gen = g_vcache_gen; g_vcache[slot].wk = (int32_t)v;
        g_st_miss++;
    }
    g_vs_idx[g_nv++] = v;
    S.verts++;
}
static void vs_add_vertex(const V4 *in) {
    if (g_nv >= MAXV || g_vs_n >= MAXV) return;
    uint32_t v = (uint32_t)g_vs_n++;
    memcpy(&g_vs_in[v * 16], in, 16 * sizeof(V4));
    g_vs_idx[g_nv++] = v;
    S.verts++;
}
static void gpu_triangle(const OutVtx *a, const OutVtx *b, const OutVtx *c);
static void fill_draw_state(NvDrawState *st);
static void vs_flush(void) {
    int n = g_nv, nt = 0;
    const uint32_t *v = g_vs_idx;
    #define TRI(a, b, c) do { g_vs_tri[nt++] = v[a]; g_vs_tri[nt++] = v[b]; g_vs_tri[nt++] = v[c]; } while (0)
    switch (S.prim) {
    case 5: for (int i = 0; i + 2 < n; i += 3) TRI(i, i + 1, i + 2); break;
    case 6: for (int i = 0; i + 2 < n; i++) { if (i & 1) TRI(i + 1, i, i + 2); else TRI(i, i + 1, i + 2); } break;
    case 7: case 10: for (int i = 1; i + 1 < n; i++) TRI(0, i, i + 1); break;
    case 8: for (int i = 0; i + 3 < n; i += 4) { TRI(i, i + 1, i + 2); TRI(i, i + 2, i + 3); } break;
    case 9: for (int i = 0; i + 3 < n; i += 2) { TRI(i, i + 1, i + 3); TRI(i, i + 3, i + 2); } break;
    default: break;
    }
    #undef TRI
    g_nv = 0;
    if (!nt) return;
    gpu_flush();                                   /* earlier CPU-path triangles first */
    NvDrawState st;
    fill_draw_state(&st);
    NvVsDraw d = {g_vs_in, g_vs_n, g_vs_tri, nt, S.c, g_vp_hash, vs_source_cb, (int)S.cull_enable, S.cull_face, S.front_face};
    if (g_gpu_backend->draw_vs(&st, &d)) { g_st_vs_draws++; S.tris += (uint64_t)(nt / 3); return; }
    /* not ready: transform on the CPU and draw as before */
    g_st_vs_fallback++;
    V4 o[16];
    for (int k = 0; k < g_vs_n; k++) { compute_outputs(&g_vs_in[k * 16], o); pack_out(o, &g_vbuf[k]); }
    g_vp_runs += (uint64_t)g_vs_n;
    for (int i = 0; i + 2 < nt; i += 3) gpu_triangle(&g_vbuf[g_vs_tri[i]], &g_vbuf[g_vs_tri[i + 1]], &g_vbuf[g_vs_tri[i + 2]]);
    gpu_flush();
}

static void emit_indexed(uint32_t index) {
    if (g_vs_mode) { vs_add_index(index); return; }
    if ((S.exec_mode & 3) == 2 && g_vp_dirty) decode_program();
    int cacheable = !((S.exec_mode & 3) == 2 && g_vp_writes_consts) && g_dbg_left <= 0;
    if (!cacheable) {
        V4 in[NATTR];
        if (g_npend) resolve_pending();
        g_st_miss++;
        fetch_array_vertex(index, in);
        emit_vertex(in);
        return;
    }
    if (g_nv >= MAXV) return;
    uint32_t slot = index & (VCACHE - 1);
    if (g_vcache[slot].gen == g_vcache_gen && g_vcache[slot].idx == index) {
        g_st_hit++;
        if (g_vcache[slot].wk < 0) { g_vbuf[g_nv++] = g_vcache[slot].v; S.verts++; return; }
        g_pend_pos[g_npend] = g_nv++; g_pend_wk[g_npend++] = g_vcache[slot].wk; S.verts++;
        return;
    }
    g_st_miss++;
    int k = g_nwork++;
    g_work_idx[k] = index;
    g_vcache[slot].idx = index; g_vcache[slot].gen = g_vcache_gen; g_vcache[slot].wk = k;
    g_pend_pos[g_npend] = g_nv++; g_pend_wk[g_npend++] = k; S.verts++;
}

/* HALO_REUSE_STATS=1 (diagnosis): how many draws repeat the exact vertex
 * setup (program, constants, arrays) of an earlier draw since the last
 * report, and how many vertex transforms they account for. */
static uint64_t fnv64(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}
static void reuse_begin(void) {
    if (g_reuse_on < 0) g_reuse_on = getenv("HALO_REUSE_STATS") != NULL;
    if (!g_reuse_on) return;
    static uint32_t last_flip;
    if (nv2a_stat_flips() != last_flip) { last_flip = nv2a_stat_flips(); g_reuse_nkeys = 0; }   /* per frame */
    uint64_t h = 0xCBF29CE484222325ull;
    h = fnv64(&S.exec_mode, sizeof S.exec_mode, h);
    if ((S.exec_mode & 3) == 2) {
        if (g_vp_dirty) decode_program();
        h = fnv64(g_vp, sizeof(VPIns) * (size_t)g_vp_len, h);
        h = fnv64(S.c, sizeof S.c, h);
    } else h = fnv64(S.composite, sizeof S.composite, h);
    h = fnv64(S.array_format, sizeof S.array_format, h);
    h = fnv64(S.array_offset, sizeof S.array_offset, h);
    h = fnv64(S.attr, sizeof S.attr, h);
    h = fnv64(&g_dma_a, 4, h); h = fnv64(&g_dma_b, 4, h);
    g_reuse_cur = 0;
    for (int i = 0; i < g_reuse_nkeys; i++) if (g_reuse_keys[i] == h) { g_reuse_cur = 1; break; }
    if (!g_reuse_cur && g_reuse_nkeys < 4096) g_reuse_keys[g_reuse_nkeys++] = h;
    g_reuse_draws++;
    if (g_reuse_cur) g_reuse_rep_draws++;
}
static void reuse_report(void) {
    if (g_reuse_on <= 0) return;
    runtime_log("[reuse] draws %llu, repeating an earlier vertex setup %llu (%.0f%%); vertex runs %llu, in repeating draws %llu (%.0f%%)",
                (unsigned long long)g_reuse_draws, (unsigned long long)g_reuse_rep_draws, g_reuse_draws ? 100.0 * g_reuse_rep_draws / g_reuse_draws : 0.0,
                (unsigned long long)g_reuse_runs, (unsigned long long)g_reuse_rep_runs, g_reuse_runs ? 100.0 * g_reuse_rep_runs / g_reuse_runs : 0.0);
    g_reuse_draws = g_reuse_rep_draws = g_reuse_runs = g_reuse_rep_runs = 0; g_reuse_nkeys = 0;
}

/* Texture statistics, logged every 10 s as [tex] (HALO_TEX_STATS=1): which
 * filter settings, formats and mip counts the game draws with. */
static struct { uint32_t key, n; uint16_t w, h; } g_texst[64];
static void tex_stat(uint32_t tf, uint32_t fmt, int levels, int linear, uint32_t w, uint32_t h) {
    static int on = -1;
    if (on < 0) on = getenv("HALO_TEX_STATS") != NULL;
    if (!on) return;
    uint32_t key = ((tf >> 16) & 0x3F) | ((tf >> 24) & 15) << 6 | (uint32_t)(levels & 15) << 10 | (fmt & 0xFF) << 14 | (uint32_t)(linear != 0) << 22 | (tf & 0x1FFF) << 23;
    for (int k = 0; k < 64; k++) {
        if (g_texst[k].n && g_texst[k].key == key) { g_texst[k].n++; return; }
        if (!g_texst[k].n) { g_texst[k].key = key; g_texst[k].n = 1; g_texst[k].w = (uint16_t)w; g_texst[k].h = (uint16_t)h; return; }
    }
}
static void tex_stats_report(void) {
    if (!g_texst[0].n) return;
    runtime_log("[tex] texture uses since last report (min filter, mag filter, mip levels, format, linear, lod bias raw, example size):");
    for (int k = 0; k < 64 && g_texst[k].n; k++) {
        uint32_t key = g_texst[k].key;
        runtime_log("[tex]   %8u x  min %u mag %u levels %2u fmt %02x linear %u bias %04x  %ux%u", g_texst[k].n, key & 0x3F, (key >> 6) & 15,
                    (key >> 10) & 15, (key >> 14) & 0xFF, (key >> 22) & 1, key >> 23, g_texst[k].w, g_texst[k].h);
    }
    memset(g_texst, 0, sizeof g_texst);
}

/* HALO_VSTATS=1: vertex work per 10 s (cache hits/misses, program runs and length). */
void nv2a_raster_vstats(void) {
    tex_stats_report();
    if (g_vp_check == 1) runtime_log("[vpcheck] %u vertices compared, %u differ", g_vp_check_n, g_vp_check_bad);
    reuse_report();
    static int on = -1;
    if (on < 0) on = getenv("HALO_VSTATS") != NULL;
    if (!on) return;
    runtime_log("[vstats] indexed: %llu hits %llu misses | program runs %llu, %.1f instructions each | batches %llu, %llu parallel",
                (unsigned long long)g_st_hit, (unsigned long long)g_st_miss, (unsigned long long)g_vp_runs,
                g_vp_runs ? (double)g_vp_ins / (double)g_vp_runs : 0.0, (unsigned long long)g_st_batches, (unsigned long long)g_st_par);
    if (g_st_vs_draws || g_st_vs_fallback) runtime_log("[vstats] GPU vertex programs: %llu draws on the GPU, %llu on the CPU while compiling",
                                                     (unsigned long long)g_st_vs_draws, (unsigned long long)g_st_vs_fallback);
    g_st_vs_draws = g_st_vs_fallback = 0;
    if (g_xf_n) runtime_log("[vstats] transform: %.0f ns per vertex (thread time, %llu vertices)", (double)g_xf_ns / (double)g_xf_n, (unsigned long long)g_xf_n);
    g_xf_ns = g_xf_n = 0;
    g_st_hit = g_st_miss = g_vp_runs = g_vp_ins = g_st_batches = g_st_par = 0;
}

/* ===================================================== method entry */
int nv2a_raster_method(uint32_t m, uint32_t v) {
    if (g_npend && m != 0x1800 && m != 0x1808) resolve_pending();
    float f = u2f(v);
    if (m >= 0x1500 && m < 0x1510) {                                   /* SET_VERTEX3F */
        float *p = &S.attr[0].x; p[(m - 0x1500) / 4] = f;
        if (m == 0x1508) { S.attr[0].w = 1.0f; if (S.prim) emit_vertex(S.attr); }
        return 1;
    }
    if (m >= 0x1518 && m < 0x1528) {                                   /* SET_VERTEX4F */
        float *p = &S.attr[0].x; p[(m - 0x1518) / 4] = f;
        if (m == 0x1524 && S.prim) emit_vertex(S.attr);
        return 1;
    }
    if (m >= 0x1880 && m < 0x1900) {                                   /* SET_VERTEX_DATA2F_M */
        int i = (int)(m - 0x1880) / 8, k = (int)((m - 0x1880) / 4) & 1;
        float *p = &S.attr[i].x; p[k] = f;
        if (k == 1) { S.attr[i].z = 0; S.attr[i].w = 1; if (i == 0 && S.prim) emit_vertex(S.attr); }
        return 1;
    }
    if (m >= 0x1900 && m < 0x1940) {                                   /* SET_VERTEX_DATA2S */
        int i = (int)(m - 0x1900) / 4;
        S.attr[i] = (V4){(float)(int16_t)(v & 0xFFFF), (float)(int16_t)(v >> 16), 0, 1};
        if (i == 0 && S.prim) emit_vertex(S.attr);
        return 1;
    }
    if (m >= 0x1940 && m < 0x1980) {                                   /* SET_VERTEX_DATA4UB */
        int i = (int)(m - 0x1940) / 4;
        S.attr[i] = (V4){(float)(v & 255) / 255.0f, (float)(v >> 8 & 255) / 255.0f, (float)(v >> 16 & 255) / 255.0f, (float)(v >> 24) / 255.0f};
        if (i == 0 && S.prim) emit_vertex(S.attr);
        return 1;
    }
    if (m >= 0x1980 && m < 0x1A00) {                                   /* SET_VERTEX_DATA4S_M */
        int i = (int)(m - 0x1980) / 8, k = (int)((m - 0x1980) / 4) & 1;
        float *p = &S.attr[i].x;
        p[2 * k] = (float)(int16_t)(v & 0xFFFF); p[2 * k + 1] = (float)(int16_t)(v >> 16);
        if (k == 1 && i == 0 && S.prim) emit_vertex(S.attr);
        return 1;
    }
    if (m >= 0x1A00 && m < 0x1B00) {                                   /* SET_VERTEX_DATA4F_M */
        int i = (int)(m - 0x1A00) / 16, k = (int)((m - 0x1A00) / 4) & 3;
        float *p = &S.attr[i].x; p[k] = f;
        if (k == 3 && i == 0 && S.prim) emit_vertex(S.attr);
        return 1;
    }
    /* fixed-function transform registers share the vertex-program constant file */
    if (m >= 0x0680 && m < 0x06C0) {                                   /* COMPOSITE_MATRIX -> c0..c3 */
        uint32_t k = (m - 0x0680) / 4;
        S.composite[k] = f; S.composite_set = 1;
        float *p = &S.c[k / 4].x; p[k & 3] = f;
        return 1;
    }
    if (m >= 0x0AF0 && m < 0x0B00) { float *p = &S.c[58].x; p[(m - 0x0AF0) / 4] = f; return 1; }  /* VIEWPORT_SCALE  */
    if (m >= 0x0A20 && m < 0x0A30) { float *p = &S.c[59].x; p[(m - 0x0A20) / 4] = f; return 1; }  /* VIEWPORT_OFFSET */
    if (m >= 0x1720 && m < 0x1760) { S.array_offset[(m - 0x1720) / 4] = v; return 1; }
    if (m >= 0x1760 && m < 0x17A0) { S.array_format[(m - 0x1760) / 4] = v; return 1; }
    /* register combiners */
    if (m >= 0x0260 && m < 0x0280) { S.alpha_icw[(m - 0x0260) / 4] = v; return 1; }
    if (m >= 0x0A60 && m < 0x0A80) { S.factor0[(m - 0x0A60) / 4] = v; return 1; }
    if (m >= 0x0A80 && m < 0x0AA0) { S.factor1[(m - 0x0A80) / 4] = v; return 1; }
    if (m >= 0x0AA0 && m < 0x0AC0) { S.alpha_ocw[(m - 0x0AA0) / 4] = v; return 1; }
    if (m >= 0x0AC0 && m < 0x0AE0) { S.color_icw[(m - 0x0AC0) / 4] = v; return 1; }
    if (m >= 0x1E40 && m < 0x1E60) { S.color_ocw[(m - 0x1E40) / 4] = v; return 1; }
    switch (m) {
    case 0x0288: S.final0 = v; return 1;
    case 0x028C: S.final1 = v; return 1;
    case 0x02A8: S.fog_color = v; return 1;
    case 0x029C: S.fog_mode = v; return 1;                   /* SET_FOG_MODE */
    case 0x0338: S.poly_offset_fill = v; return 1;           /* SET_POLY_OFFSET_FILL_ENABLE */
    case 0x0384: memcpy(&S.poly_scale, &v, 4); return 1;     /* SET_POLYGON_OFFSET_SCALE_FACTOR */
    case 0x0388: memcpy(&S.poly_bias, &v, 4); return 1;      /* SET_POLYGON_OFFSET_BIAS */
    case 0x02A0: S.fog_gen = v; return 1;                    /* SET_FOG_GEN_MODE */
    case 0x02A4: S.fog_enable = v; return 1;                 /* SET_FOG_ENABLE */
    case 0x09C0: case 0x09C4: case 0x09C8: memcpy(&S.fog_params[(m - 0x09C0) / 4], &v, 4); return 1;   /* SET_FOG_PARAMS */
    case 0x1E60: S.comb_control = v; return 1;
    case 0x1E20: S.specfog[0] = v; return 1;
    case 0x1E24: S.specfog[1] = v; return 1;
    case 0x1E70: S.shader_stage_prog = v; S.stage_prog_set = 1; return 1;
    case 0x17FC:
        if (v == 0) { flush_primitive(); S.prim = 0; }
        else {
            S.prim = v; g_nv = 0;
            reuse_begin();
            {   /* HALO_FEATURE_STATS=1: which unmodelled features draws use (diagnosis) */
                static int on = -1; static uint64_t n, stencil, pts, lines, total; static uint64_t last;
                if (on < 0) on = getenv("HALO_FEATURE_STATS") != NULL;
                if (on) {
                    total++;
                    if (S.stencil_enable) stencil++;
                    if (v == 1) pts++;
                    if (v >= 2 && v <= 4) lines++;
                    if (++n - last >= 20000) {
                        last = n;
                        runtime_log("[features] draws %llu: stencil test %llu, points %llu, lines %llu", (unsigned long long)total,
                                    (unsigned long long)stencil, (unsigned long long)pts, (unsigned long long)lines);
                    }
                }
            }
            if (++g_vcache_gen == 0) { memset(g_vcache, 0, sizeof g_vcache); g_vcache_gen = 1; }
            if (g_dbg_small < 0) { g_dbg_small = getenv("HALO_DEBUG_SMALLRT") ? 1 : 0; g_dbg_left = g_dbg_small ? 6 : 0; }
            vs_begin();
        }
        return 0;
    case 0x1800: emit_indexed(v & 0xFFFF); emit_indexed(v >> 16); return 1;
    case 0x1808: emit_indexed(v); return 1;
    case 0x1810: {
        uint32_t start = v & 0xFFFFFF, count = (v >> 24) + 1;
        V4 in[NATTR];
        for (uint32_t i = 0; i < count; i++) { fetch_array_vertex(start + i, in); emit_vertex(in); }
        return 1;
    }
    case 0x1E94: S.exec_mode = v; g_vcache_gen++; return 1;
    case 0x1E9C: S.prog_load = v; S.prog_sub = 0; return 1;
    case 0x1EA0: S.prog_start = v; g_vp_dirty = 1; return 1;
    case 0x1EA4: S.const_load = v; S.const_sub = 0; return 1;
    /* raster ops */
    case 0x0300: S.alpha_test_enable = v; return 1;
    case 0x0304: S.blend_enable = v; return 1;
    case 0x0308: S.cull_enable = v; return 1;
    case 0x030C: S.depth_test = v; return 1;
    case 0x033C: S.alpha_func = v; return 1;
    case 0x0340: S.alpha_ref = v; return 1;
    case 0x0344: S.blend_src = v; return 1;
    case 0x0348: S.blend_dst = v; return 1;
    case 0x034C: S.blend_color = v; return 1;
    case 0x0350: S.blend_eq = v; return 1;
    case 0x0354: S.depth_func = v; return 1;
    case 0x0358: S.color_mask = v; return 1;
    case 0x035C: S.depth_mask = v; return 1;
    case 0x039C: S.cull_face = v; return 1;
    case 0x032C: S.stencil_enable = v; return 1;
    case 0x0360: S.stencil_mask = v; return 1;
    case 0x0364: S.stencil_func = v; return 1;
    case 0x0368: S.stencil_ref = v; return 1;
    case 0x036C: S.stencil_func_mask = v; return 1;
    case 0x0370: S.stencil_op_fail = v; return 1;
    case 0x0374: S.stencil_op_zfail = v; return 1;
    case 0x0378: S.stencil_op_zpass = v; return 1;
    case 0x03A0: S.front_face = v; return 1;
    default: break;
    }
    if (m >= 0x0B00 && m < 0x0B80) {
        if (S.prog_load < VP_SLOTS) S.prog[S.prog_load][S.prog_sub] = v;
        g_vp_dirty = 1;
        if (++S.prog_sub == 4) { S.prog_sub = 0; S.prog_load++; }
        return 1;
    }
    if (m >= 0x0B80 && m < 0x0C00) {
        if (S.const_load < VP_CONSTS) { float *p = &S.c[S.const_load].x; p[S.const_sub] = f; }
        if (++S.const_sub == 4) { S.const_sub = 0; S.const_load++; }
        return 1;
    }
    if (m >= 0x1B00 && m < 0x1C00) {                                   /* texture stages */
        int i = (int)(m - 0x1B00) / 0x40;
        switch ((m - 0x1B00) & 0x3F) {
        case 0x00: S.tex_offset[i] = v; break;
        case 0x04: S.tex_format[i] = v; break;
        case 0x08: S.tex_address[i] = v; break;
        case 0x0C: S.tex_control0[i] = v; break;
        case 0x10: S.tex_control1[i] = v; break;
        case 0x14: S.tex_filter[i] = v; break;
        case 0x1C: S.tex_rect[i] = v; break;
        case 0x20: S.tex_palette[i] = v; break;
        default: break;
        }
        return 1;
    }
    return 0;
}
