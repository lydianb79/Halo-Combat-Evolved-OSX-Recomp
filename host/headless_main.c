/* headless_main.c — command-line runner (macOS or Linux), no window.
 *
 *   ./build/halo-headless --xbe extracted/default.xbe --iso "../Halo.iso" [--trace-kernel] [--trace-calls]
 *
 * Useful for bring-up: it runs the recompiled game until the first trap and
 * prints exactly which instruction / kernel call / indirect target is missing. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../runtime/gfx.h"
#include "../runtime/runtime.h"

static unsigned g_frames;
void gfx_create_device(uint32_t w, uint32_t h) { runtime_log("[gfx] headless device %ux%u", w, h); }
void gfx_clear(uint32_t argb, uint32_t flags) { (void)argb; (void)flags; }
void gfx_present(void) {
    if (++g_frames % 60 == 0) runtime_log("[gfx] %u frames presented", g_frames);
}
void gfx_wait_vblank(void) { usleep(16666); }
static uint32_t g_scanouts, g_last_px;
void gfx_present_frame(const uint32_t *px, uint32_t w, uint32_t h, uint32_t pitch) {
    uint32_t center = px[(h / 2) * (pitch / 4) + w / 2];
    if (++g_scanouts % 120 == 0 || center != g_last_px)
        runtime_log("[gfx] scanout #%u %ux%u, centre pixel %08x", g_scanouts, w, h, center);
    g_last_px = center;
}

int main(int argc, char **argv) {
    RuntimeConfig cfg = {0};
    cfg.xbe_path = "extracted/default.xbe";
    cfg.save_root = "xbox_hdd";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--xbe") && i + 1 < argc) cfg.xbe_path = argv[++i];
        else if (!strcmp(argv[i], "--iso") && i + 1 < argc) cfg.iso_path = argv[++i];
        else if (!strcmp(argv[i], "--game-root") && i + 1 < argc) cfg.game_root = argv[++i];
        else if (!strcmp(argv[i], "--save-root") && i + 1 < argc) cfg.save_root = argv[++i];
        else if (!strcmp(argv[i], "--trace-kernel")) cfg.trace_kernel = 1;
        else if (!strcmp(argv[i], "--trace-calls")) cfg.trace_calls = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if (getenv("HALO_GPU_PROBE")) { void gpu_probe_install(void); gpu_probe_install(); }
    if (runtime_init(&cfg)) return 1;
    runtime_run();
    return 0;
}
