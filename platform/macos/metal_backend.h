// metal_backend.h — NV2A pixel work on the Mac's GPU (see runtime/gpu_backend.h).
#pragma once
#import <Metal/Metal.h>
#include <stdint.h>

// Install the Metal backend as g_gpu_backend. Uses the presentation queue so
// that a frame's drawing is always submitted before it is displayed.
void metal_backend_install(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library);

// The GPU render target at this guest address (the scanned-out framebuffer),
// or nil if the GPU has not drawn there. Safe to call from the main thread.
id<MTLTexture> metal_backend_surface(uint32_t guestAddress);
// Render resolution: 1-4 times the Xbox's (applied at the next frame).
void metal_backend_set_render_scale(int scale);
