/* nv2a_raster.h — NV2A 3D pipeline (reference software rasterizer). */
#pragma once
#include <stdint.h>

typedef struct {
    uint32_t color_addr;          /* guest address of the colour surface (kernel mirror) */
    uint32_t pitch;               /* bytes per row */
    uint32_t clip_x, clip_y, clip_w, clip_h;
    uint32_t zeta_addr, zeta_pitch, zeta_fmt;   /* depth/stencil surface; fmt 1 = Z16, 2 = Z24S8 */
    uint32_t swizzled, swz_log2w, swz_log2h;     /* swizzled colour surface (render-to-texture) */
} Nv2aSurface;

void nv2a_raster_bind(uint32_t (*dma_lookup)(uint32_t handle, uint32_t *limit));
void nv2a_raster_set_surface(const Nv2aSurface *s);
void nv2a_raster_set_dma(uint32_t dma_a, uint32_t dma_b);
int  nv2a_raster_method(uint32_t method, uint32_t value);   /* returns 1 if consumed */
void nv2a_raster_clear(uint32_t flags, uint32_t color, uint32_t zstencil, uint32_t rect_h, uint32_t rect_v);
void nv2a_raster_stats(uint64_t *tris, uint64_t *pixels);
void nv2a_raster_debug_dump(void);
void nv2a_raster_set_texture_filter(int mode);   /* 0 game default, 1 fastest, 2 nearest, 3 bilinear, 4 trilinear, 5-8 anisotropic 2x-16x */
