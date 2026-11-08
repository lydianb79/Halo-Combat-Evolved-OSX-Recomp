/* hle.c — high-level replacements for statically linked XDK library functions.
 *
 * The XBE carries no symbols, so library entry points are bound by name through
 * a map file (default: hle_map.txt next to the XBE), one "ADDRESS NAME" pair
 * per line, e.g.
 *
 *     0x0018a1b0 D3DDevice_Clear
 *     0x0018c4e0 D3DDevice_Swap
 *
 * Addresses are found by hand in a disassembler for now (an XDK 3911/3925
 * signature matcher is on the roadmap). Any name listed below gets the
 * host implementation; every other recompiled function runs as translated.
 */
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "runtime.h"

typedef struct { uint32_t addr; const char *name; HleFn fn; } HleEntry;
static HleEntry g_hle[512];
static int g_nhle;

void hle_register(uint32_t addr, const char *name, HleFn fn) {
    if (g_nhle >= 512) return;
    g_hle[g_nhle++] = (HleEntry){addr, name, fn};
}

HleFn hle_lookup(uint32_t addr, const char **name) {
    for (int i = 0; i < g_nhle; i++)
        if (g_hle[i].addr == addr) { if (name) *name = g_hle[i].name; return g_hle[i].fn; }
    return NULL;
}

/* ------------------------------------------------- Direct3D 8 (Xbox) */
/* Xbox D3D8 uses __stdcall for most D3DDevice_* functions and keeps the
 * device implicit (a global), so argument indices start at 0. */

static void hle_D3D_CreateDevice(CPU *c) {
    /* (Adapter, DeviceType, hFocusWindow, BehaviorFlags, pPresentationParameters, ppReturnedDevice) */
    uint32_t pp = hle_arg(c, 4);
    uint32_t w = rd32(pp), h = rd32(pp + 4);
    gfx_create_device(w ? w : 640, h ? h : 480);
    uint32_t dev = kdata_alloc(0x1000);
    wr32(hle_arg(c, 5), dev);
    runtime_log("[d3d] CreateDevice %ux%u -> %08x", w, h, dev);
    hle_return(c, 0, 6);
}
static void hle_D3DDevice_Clear(CPU *c) {
    /* (Count, pRects, Flags, Color, Z, Stencil) */
    uint32_t color = hle_arg(c, 3);
    gfx_clear(color, hle_arg(c, 2));
    hle_return(c, 0, 6);
}
static void hle_D3DDevice_Swap(CPU *c) {
    gfx_present();
    hle_return(c, 0, 1);
}
static void hle_D3DDevice_Present(CPU *c) {
    gfx_present();
    hle_return(c, 0, 4);
}
static void hle_D3DDevice_BlockUntilVerticalBlank(CPU *c) {
    gfx_wait_vblank();
    hle_return(c, 0, 0);
}
static void hle_noop_0(CPU *c) { hle_return(c, 0, 0); }

static const struct { const char *name; HleFn fn; } kImpls[] = {
    {"Direct3D_CreateDevice", hle_D3D_CreateDevice},
    {"D3DDevice_Clear", hle_D3DDevice_Clear},
    {"D3DDevice_Swap", hle_D3DDevice_Swap},
    {"D3DDevice_Present", hle_D3DDevice_Present},
    {"D3DDevice_BlockUntilVerticalBlank", hle_D3DDevice_BlockUntilVerticalBlank},
    {"NOOP_STDCALL0", hle_noop_0},
};

void hle_init(void) {
    xinput_hle_init();
    char path[1200];
    const char *xbe = g_cfg.xbe_path;
    const char *slash = strrchr(xbe, '/');
    snprintf(path, sizeof path, "%.*shle_map.txt", slash ? (int)(slash - xbe + 1) : 0, xbe);
    const char *env = getenv("HALO_HLE_MAP");
    FILE *f = fopen(env ? env : path, "r");
    if (!f) {
        runtime_log("[hle] no %s: all library code runs recompiled (graphics will not appear)", env ? env : path);
        return;
    }
    char line[256];
    int bound = 0, unknown = 0;
    while (fgets(line, sizeof line, f)) {
        char name[128];
        unsigned addr;
        if (line[0] == '#' || sscanf(line, "%x %127s", &addr, name) != 2) continue;
        int found = 0;
        for (size_t i = 0; i < sizeof kImpls / sizeof kImpls[0]; i++)
            if (!strcmp(kImpls[i].name, name)) {
                hle_register(addr, kImpls[i].name, kImpls[i].fn);
                found = 1;
                bound++;
            }
        if (!found) unknown++;
    }
    fclose(f);
    runtime_log("[hle] bound %d functions (%d mapped names have no host implementation yet)", bound, unknown);
}
