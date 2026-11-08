/* xinput.c — controller input: host replacements for the XDK's XInput API.
 *
 * Halo links the XDK input library (section XPP), which drives the Xbox's USB
 * controller hardware. Instead of emulating USB, the six entry points Halo
 * calls are replaced with host code that reports one gamepad on port 0 and
 * reads it from the platform layer (g_host.poll_input: a game controller or
 * the keyboard on macOS).
 *
 * The addresses are those of Halo: Combat Evolved (USA). Each binding is only
 * made if the code at that address matches the expected first bytes, so a
 * different build simply keeps running its recompiled (USB) input code.
 *
 * Headless testing: HALO_INPUT_SCRIPT="ms:buttons,ms:buttons,..." presses
 * buttons at the given times since start, e.g. "40000:down,40300:,41000:A,41300:".
 * Button names: up down left right start back A B X Y black white LT RT.
 */
#include <stdlib.h>
#include <string.h>

#include "runtime.h"

#define XDEVICE_TYPE_GAMEPAD 0x001CF4FCu   /* address of the XPP_DEVICE_TYPE descriptor in this build */
#define ERROR_SUCCESS 0
#define ERROR_DEVICE_NOT_CONNECTED 1167
#define ERROR_IO_PENDING 997

static int g_reported_insert;       /* port 0 insertion reported by XGetDeviceChanges */
static uint32_t g_handle;           /* guest-visible handle for the open gamepad */
static uint32_t g_packet;
static uint16_t g_last_buttons;
static uint8_t g_last_analog[8];
static int16_t g_last_axes[4];
static int g_log;
static uint64_t g_polls;

/* ---------------------------------------------------------- scripted input */
typedef struct { uint64_t ms; uint16_t buttons; uint8_t analog[8]; int16_t axes[4]; } ScriptStep;
static ScriptStep g_script[512];
static int g_nscript = -1;
static uint64_t g_t0;

static void parse_script(void) {
    g_nscript = 0;
    const char *s = getenv("HALO_INPUT_SCRIPT");
    if (!s) return;
    while (*s && g_nscript < 512) {
        ScriptStep st;
        memset(&st, 0, sizeof st);
        st.ms = strtoull(s, (char **)&s, 10);
        if (*s == ':') s++;
        while (*s && *s != ',') {
            char name[16]; int n = 0;
            while (*s && *s != ',' && *s != '+' && n < 15) name[n++] = *s++;
            name[n] = 0;
            if (*s == '+') s++;
            static const char *dig[] = {"up", "down", "left", "right", "start", "back"};
            static const char *ana[] = {"A", "B", "X", "Y", "black", "white", "LT", "RT"};
            for (int i = 0; i < 6; i++) if (!strcmp(name, dig[i])) st.buttons |= (uint16_t)(1u << i);
            for (int i = 0; i < 8; i++) if (!strcmp(name, ana[i])) st.analog[i] = 255;
            static const char *ax[] = {"lx=", "ly=", "rx=", "ry="};       /* sticks: e.g. ry=-20000 */
            for (int i = 0; i < 4; i++) if (!strncmp(name, ax[i], 3)) st.axes[i] = (int16_t)atoi(name + 3);
        }
        if (*s == ',') s++;
        g_script[g_nscript++] = st;
    }
    runtime_log("[xinput] input script with %d steps", g_nscript);
}

static void read_pad(uint16_t *buttons, uint8_t analog[8], int16_t axes[4]) {
    *buttons = 0;
    memset(analog, 0, 8);
    memset(axes, 0, 8);
    if (g_host.poll_input) g_host.poll_input(g_host.user, buttons, analog, axes);
    if (g_nscript < 0) { parse_script(); g_t0 = host_time_ns(); }
    if (g_nscript > 0) {
        uint64_t ms = (host_time_ns() - g_t0) / 1000000ull;
        const ScriptStep *cur = NULL;
        for (int i = 0; i < g_nscript; i++) if (g_script[i].ms <= ms) cur = &g_script[i];
        if (cur) {
            *buttons |= cur->buttons;
            for (int i = 0; i < 8; i++) if (cur->analog[i]) analog[i] = cur->analog[i];
            for (int i = 0; i < 4; i++) if (cur->axes[i]) axes[i] = cur->axes[i];
        }
    }
}

/* -------------------------------------------------------------- XInput API */
static void hle_XInitDevices(CPU *c) {            /* (dwPreallocTypeCount, PreallocTypes) */
    runtime_log("[xinput] XInitDevices: one gamepad on port 0");
    hle_return(c, 0, 2);
}

static void hle_XGetDeviceChanges(CPU *c) {       /* (DeviceType, *pdwInsertions, *pdwRemovals) -> BOOL */
    uint32_t type = hle_arg(c, 0), ins = 0;
    if (type == XDEVICE_TYPE_GAMEPAD && !g_reported_insert) { ins = 1; g_reported_insert = 1; }
    wr32(hle_arg(c, 1), ins);
    wr32(hle_arg(c, 2), 0);
    hle_return(c, ins != 0, 3);
}

static void hle_XInputOpen(CPU *c) {              /* (DeviceType, dwPort, dwSlot, pPollingParameters) -> HANDLE */
    uint32_t type = hle_arg(c, 0), port = hle_arg(c, 1);
    uint32_t h = 0;
    if (type == XDEVICE_TYPE_GAMEPAD && port == 0) {
        if (!g_handle) g_handle = kdata_alloc(0x100);
        h = g_handle;
    }
    runtime_log("[xinput] XInputOpen(port %u) -> %08x", port, h);
    hle_return(c, h, 4);
}

static void hle_XInputClose(CPU *c) {             /* (hDevice) */
    runtime_log("[xinput] XInputClose(%08x)", hle_arg(c, 0));
    hle_return(c, 0, 1);
}

static void hle_XInputGetState(CPU *c) {          /* (hDevice, PXINPUT_STATE) -> DWORD */
    uint32_t h = hle_arg(c, 0), st = hle_arg(c, 1);
    if (!h || h != g_handle) { hle_return(c, ERROR_DEVICE_NOT_CONNECTED, 2); return; }
    uint16_t buttons; uint8_t analog[8]; int16_t axes[4];
    g_polls++;
    read_pad(&buttons, analog, axes);
    if (buttons != g_last_buttons || memcmp(analog, g_last_analog, 8) || memcmp(axes, g_last_axes, 8)) {
        g_packet++;
        if (g_log < 200) {
            g_log++;
            runtime_log("[xinput] t=%llums poll %llu state %u: buttons %04x analog %02x %02x %02x %02x %02x %02x %02x %02x sticks %d %d %d %d",
                        (unsigned long long)((host_time_ns() - g_t0) / 1000000ull), (unsigned long long)g_polls, g_packet, buttons, analog[0], analog[1], analog[2], analog[3], analog[4], analog[5], analog[6],
                        analog[7], axes[0], axes[1], axes[2], axes[3]);
        }
        g_last_buttons = buttons; memcpy(g_last_analog, analog, 8); memcpy(g_last_axes, axes, 8);
    }
    /* XINPUT_STATE: DWORD dwPacketNumber; XINPUT_GAMEPAD { WORD wButtons; BYTE bAnalogButtons[8];
     * SHORT sThumbLX, sThumbLY, sThumbRX, sThumbRY; } */
    wr32(st, g_packet);
    wr16(st + 4, buttons);
    for (int i = 0; i < 8; i++) wr8(st + 6 + (uint32_t)i, analog[i]);
    for (int i = 0; i < 4; i++) wr16(st + 14 + 2 * (uint32_t)i, (uint16_t)axes[i]);
    hle_return(c, ERROR_SUCCESS, 2);
}

static void hle_XInputSetState(CPU *c) {          /* (hDevice, PXINPUT_FEEDBACK) -> DWORD: rumble */
    uint32_t h = hle_arg(c, 0), fb = hle_arg(c, 1);
    if (!h || h != g_handle) { hle_return(c, ERROR_DEVICE_NOT_CONNECTED, 2); return; }
    /* XINPUT_FEEDBACK_HEADER: dwStatus, hEvent. Complete at once. */
    wr32(fb, ERROR_SUCCESS);
    uint32_t ev = rd32(fb + 4);
    if (ev) kernel_set_event_handle(ev);
    hle_return(c, ERROR_IO_PENDING, 2);
}

/* ------------------------------------------------------------------ binding */
static const struct {
    uint32_t addr; const char *name; HleFn fn; uint8_t sig[8];
} kBind[] = {
    {0x001D129A, "XInitDevices",       hle_XInitDevices,       {0xe9, 0x76, 0xea, 0xff, 0xff, 0x55, 0x8b, 0xec}},
    {0x001CFD15, "XInitDevices_body",  hle_XInitDevices,       {0x53, 0x56, 0x57, 0x68, 0xb4, 0x00, 0x00, 0x00}},
    {0x001D129F, "XGetDeviceChanges",  hle_XGetDeviceChanges,  {0x55, 0x8b, 0xec, 0x56, 0x8b, 0x75, 0x08, 0x33}},
    {0x001D130C, "XInputOpen",         hle_XInputOpen,         {0x55, 0x8b, 0xec, 0x51, 0x83, 0x65, 0xfc, 0x00}},
    {0x001D1381, "XInputClose",        hle_XInputClose,        {0x8b, 0x4c, 0x24, 0x04, 0xe8, 0x6f, 0xf5, 0xff}},
    {0x001D138D, "XInputGetState",     hle_XInputGetState,     {0x53, 0x56, 0x33, 0xdb, 0xff, 0x15, 0xb8, 0x66}},
    {0x001D13FE, "XInputSetState",     hle_XInputSetState,     {0x8b, 0x4c, 0x24, 0x04, 0x8d, 0x81, 0xa3, 0x00}},
};

void xinput_hle_init(void) {
    int bound = 0;
    for (size_t i = 0; i < sizeof kBind / sizeof kBind[0]; i++) {
        int ok = 1;
        for (int k = 0; k < 8; k++) if (rd8(kBind[i].addr + (uint32_t)k) != kBind[i].sig[k]) ok = 0;
        if (ok) { hle_register(kBind[i].addr, kBind[i].name, kBind[i].fn); bound++; }
    }
    if (bound == (int)(sizeof kBind / sizeof kBind[0])) runtime_log("[xinput] controller input bound (%d entry points)", bound);
    else runtime_log("[xinput] only %d of %d input entry points matched this XBE; input stays on the recompiled USB code",
                     bound, (int)(sizeof kBind / sizeof kBind[0]));
}
