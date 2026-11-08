/* gpu_backend.h — hand-off from the NV2A model to a host GPU.
 *
 * The NV2A model (nv2a_raster.c) always runs vertex programs, primitive
 * assembly, near-plane clipping and culling on the CPU. What remains — pixel
 * work: texturing, register combiners, alpha/depth test, blending — is done
 * either by the CPU rasterizer or, when the host installs a backend here, by
 * the host GPU (platform/macos/metal_backend.m).
 *
 * With a backend, render targets live on the GPU, keyed by their guest
 * address: textures that point at a render target are sampled from it, and the
 * display shows it directly. Guest memory is only read (textures, palettes,
 * the initial contents of a new render target), never written.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float x, y, z, w; } V4;

/* One vertex after transform: screen space, z in depth-buffer units, w > 0. */
typedef struct {
    float x, y, z, w;
    V4 d0, d1;              /* diffuse, specular */
    V4 t[4];                /* texture coordinates (normalized, or texels for linear textures) */
    float fog;              /* fog coordinate (vertex program output oFog.x) */
} OutVtx;

typedef struct {
    int enabled;
    uint32_t addr;          /* guest address of level 0 (kernel mirror) */
    uint32_t fmt;           /* NV097 colour format */
    uint32_t w, h, pitch;   /* level 0 size; pitch for linear textures */
    int linear;             /* linear ("LU") format: texel coordinates, no mipmaps */
    int levels;
    uint32_t palette_addr;  /* guest address of the 256-entry palette, or 0 */
    uint32_t bytes;         /* bytes of the whole mip chain in guest memory */
    uint32_t addr_u, addr_v;/* 1 wrap, 2 mirror, 3+ clamp */
    int mag_linear, min_linear, mip; /* mip: 0 none, 1 nearest level, 2 linear between levels */
    int max_aniso;          /* 1, 2, 4, 8, 16 */
    int cube;               /* cube map: 6 faces (+X -X +Y -Y +Z -Z), sampled with (s, t, r) */
    uint32_t face_bytes;    /* distance between faces */
} NvTexture;

/* A draw whose vertex program runs on the GPU (backend draw_vs). */
typedef struct NvVsDraw {
    const V4 *inputs;              /* 16 inputs per vertex */
    int nverts;
    const uint32_t *indices;       /* triangle list into inputs */
    int nindices;
    const V4 *consts;              /* the 192 program constants */
    uint64_t hash;                 /* identifies the program */
    const char *(*source)(void);   /* Metal source of the program (vertex function nv_vs_prog) */
    int cull_enable;
    uint32_t cull_face, front_face;/* GL enums: 0x404 front, 0x405 back, 0x408 both; 0x900 CW, 0x901 CCW */
} NvVsDraw;

typedef struct {
    /* render target */
    uint32_t color_addr, pitch, width, height;          /* width = pitch / 4 */
    uint32_t clip_x, clip_y, clip_w, clip_h;
    uint32_t zeta_addr, zeta_pitch, zeta_fmt;          /* fmt 1 Z16, 2 Z24S8, 0 none */
    /* textures */
    NvTexture tex[4];
    /* register combiners (0 stages: texture 0 x diffuse) */
    uint32_t comb_control, final0, final1, fog_color;
    /* fog: code 0 off, 1 linear, 2 exp, 3 exp2, +4 on the absolute coordinate; bias and multiplier (FOG_PARAMS 0/1) */
    uint32_t fog_code;
    float fog_bias, fog_mult;
    /* polygon offset for filled triangles (decals): depth += bias * unit + scale * slope */
    uint32_t poly_offset_fill;
    float poly_scale, poly_bias;
    /* stencil (GL enums as the methods: funcs 0x200-0x207, ops 0x1E00 keep ...) */
    uint32_t stencil_enable, stencil_mask, stencil_func, stencil_ref, stencil_func_mask;
    uint32_t stencil_op_fail, stencil_op_zfail, stencil_op_zpass;
    uint32_t specfog[2];    /* final combiner constants C0/C1 */
    uint32_t color_icw[8], color_ocw[8], alpha_icw[8], alpha_ocw[8], factor0[8], factor1[8];
    /* raster ops */
    uint32_t blend_enable, blend_src, blend_dst, blend_eq, blend_color;
    uint32_t alpha_test_enable, alpha_func, alpha_ref;
    uint32_t depth_test, depth_func, depth_mask;
    uint32_t color_mask;
} NvDrawState;

typedef struct {
    /* Draw a triangle list (3 vertices per triangle, already clipped and culled). */
    void (*draw)(const NvDrawState *st, const OutVtx *v, int count);
    /* Clear the inclusive rectangle; flags as method 0x1D94 (bit 0 Z, bit 1 stencil, 0xF0 colour). */
    void (*clear)(const NvDrawState *st, uint32_t flags, uint32_t color, uint32_t zstencil,
                  uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1);
    /* End of a frame (buffer flip) or vertical blank: submit the queued GPU work. */
    void (*flush)(void);
    /* 2D copy (image blit): guest addresses of the source and destination
     * surfaces, their pitches, bytes per pixel, rectangle. Guest memory has
     * already been copied; the backend updates its GPU copies. May be NULL. */
    void (*blit)(uint32_t src, uint32_t src_pitch, uint32_t dst, uint32_t dst_pitch, uint32_t bpp,
                 uint32_t sx, uint32_t sy, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h);
    /* Buffer flip (FLIP_INCREMENT_WRITE): the frame is complete. May be NULL (then flush is called). */
    void (*flip)(void);
    /* Optional: draw with the vertex program run on the GPU. Returns 0 when it
     * cannot draw yet (shader still compiling); the CPU path then draws. */
    int (*draw_vs)(const NvDrawState *st, const struct NvVsDraw *d);
    /* Optional: visibility tests (pixel counts). op 0: counting on/off (arg),
     * 1: reset the count, 2: write the count so far into the 16-byte report at
     * guest address arg (+8 count, +12 status = 0 once done). Without it, reports
     * are written at once with a fixed "visible" count. */
    void (*report)(int op, uint32_t arg);
} GpuBackend;

/* Submit the triangles queued for the GPU backend (before a blit). */
void nv2a_raster_gpu_sync(void);

/* Set by the host before the game starts; NULL = CPU rasterizer. */
extern GpuBackend *g_gpu_backend;
/* Diagnosis: when set, the CPU rasterizer also draws into guest memory (to compare with the GPU). */
extern int g_gpu_cpu_shadow;

/* Decode one mip level (of one cube face; face 0 for 2D) of a texture to BGRA8 (A8R8G8B8 little-endian), w x h
 * of that level, rows packed. Used by backends to upload guest textures. */
void nv2a_decode_texture_level(const NvTexture *t, int face, int level, uint32_t *out, uint32_t *out_w, uint32_t *out_h);

#ifdef __cplusplus
}
#endif
