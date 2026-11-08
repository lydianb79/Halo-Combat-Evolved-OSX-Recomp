// graphics.h — display options for the macOS shell (menu bar: Graphics).
#pragma once
#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>

enum { GFX_FILTER_NEAREST, GFX_FILTER_BILINEAR, GFX_FILTER_SHARP, GFX_FILTER_BICUBIC, GFX_FILTER_LANCZOS, GFX_FILTER_COUNT };
// Texture filtering inside the game picture (software renderer): see nv2a_raster_set_texture_filter.
enum { TEX_GAME, TEX_FASTEST, TEX_NEAREST, TEX_BILINEAR, TEX_TRILINEAR, TEX_ANISO2, TEX_ANISO4, TEX_ANISO8, TEX_ANISO16, TEX_COUNT };
enum { GFX_RENDERER_METAL, GFX_RENDERER_CPU, GFX_RENDERER_COUNT };
enum { GFX_SCALE_FIT, GFX_SCALE_INTEGER, GFX_SCALE_FILL, GFX_SCALE_COUNT };
enum { GFX_ASPECT_4_3, GFX_ASPECT_16_9, GFX_ASPECT_STRETCH, GFX_ASPECT_COUNT };

typedef struct {
    int filter, scaling, aspect;
    int texFilter;            // TEX_*
    float scanlines;          // 0..1
    float brightness;         // 0.5..1.5
    float gamma;              // 0.6..1.6 (1 = unchanged)
    BOOL showStats, startFullScreen, hideCursor, escExitsFullScreen;
    int renderThreads;        // 0 = one per core (CPU renderer)
    int renderer;             // GFX_RENDERER_* (applies at the next start)
    int renderScale;          // Metal: draw at 1-4 times the Xbox's 640x480
} GraphicsSettings;

extern GraphicsSettings g_gfx;   // main thread only

// Uniforms for present_fs (Shaders.metal); layout must match PresentParams there.
typedef struct {
    float src_size[2], rect_origin[2], rect_size[2];
    int32_t filter;
    float scanlines, brightness, gamma;
} PresentParams;

// Load settings; call before the game thread starts (applies the render thread count).
void graphics_load(void);
// Hook up the game window and view: full screen, pointer hiding, performance overlay.
void graphics_install(NSWindow *gameWindow, MTKView *view);
// The "Graphics" page of the Settings window.
NSView *graphics_settings_view(void);
void graphics_toggle_full_screen(void);
// YES while the game window is in full screen.
BOOL graphics_is_full_screen(void);
// Where the game picture goes inside a drawable of `size` pixels.
CGRect graphics_output_rect(CGSize size, double srcWidth, double srcHeight);
