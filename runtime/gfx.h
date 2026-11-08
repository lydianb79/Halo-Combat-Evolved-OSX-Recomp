/* gfx.h — renderer interface between the HLE Direct3D layer and the host.
 * The macOS shell implements it with Metal (platform/macos/MetalRenderer.m);
 * the headless runner implements it with no-ops (host/headless_main.c). */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void gfx_create_device(uint32_t width, uint32_t height);
void gfx_clear(uint32_t argb, uint32_t flags);
void gfx_present(void);
void gfx_wait_vblank(void);
/* Called at each vertical blank with the framebuffer the GPU is scanning out (BGRA/X8R8G8B8). */
void gfx_present_frame(const uint32_t *pixels, uint32_t width, uint32_t height, uint32_t pitch_bytes);

#ifdef __cplusplus
}
#endif
