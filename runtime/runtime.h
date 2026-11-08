/* runtime.h — host-side runtime for recompiled Xbox code. */
#pragma once
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*GuestFn)(CPU *c);
typedef struct { uint32_t addr; GuestFn fn; } RecompFunc;

/* generated */
extern const RecompFunc g_recomp_funcs[];
extern const uint32_t g_recomp_func_count;
extern const uint32_t g_recomp_entry;

/* ------------------------------------------------------------ memory map */
/* Physical RAM (128 MB, twice the Xbox's 64 MB) is one shared memory object
 * mapped three times, like the real kernel's mappings: at 0x00000000 (user),
 * 0x80000000 (kernel identity mirror) and 0xF0000000 (write-combined mirror). */
#define GUEST_RAM_SIZE     0x08000000u
#define GUEST_RAM_MIRROR1  0x80000000u
#define GUEST_RAM_MIRROR2  0xF0000000u
#define GUEST_IMAGE_BASE   0x00010000u
/* Everything the guest can see lives in the Xbox's 64 MB of physical RAM, laid out
 * like the retail kernel: [0, 0x61000) kernel, the XBE image's pages near the top
 * (mapped at virtual 0x10000), GPU instance memory in the top 1 MB, and every other
 * page shared top-down by virtual (identity-mapped) and contiguous allocations. */
#define GUEST_PHYS_LIMIT   0x04000000u   /* 64 MB */
#define GUEST_KERNEL_PHYS  0x00061000u   /* physical pages used by the Xbox kernel itself */
#define GUEST_HEAP_BASE    0x003C0000u   /* lowest identity-mapped heap address (above the image's VA) */
#define GUEST_HEAP_END     GUEST_PHYS_LIMIT
#define GUEST_GPU_RESERVED 0x03FF0000u   /* top 64 KB of the 64 MB: GPU instance memory (MmClaimGpuInstanceMemory) */
#define GUEST_KDATA_BASE   0xD0000000u   /* kernel data exports, KPCR/KTHREAD blocks (not mirrored) */
#define GUEST_KDATA_END    0xD0400000u
#define KERNEL_THUNK_BASE  0xFEF00000u   /* thunk slot value = base + ordinal*4 */
#define HLE_TRAP_BASE      0xFEE00000u

uint32_t guest_alloc(uint32_t size, uint32_t align);   /* page allocator, returns 0 on failure */
uint32_t guest_free(uint32_t addr);                    /* returns bytes freed */
uint32_t guest_alloc_size(uint32_t addr);
void guest_reserve(uint32_t addr, uint32_t size);
uint32_t guest_alloc_contiguous(uint32_t size, uint32_t lowest, uint32_t highest, uint32_t align);
uint32_t guest_phys(uint32_t va);
uint32_t guest_heap_used(void);
uint32_t kdata_alloc(uint32_t size);                   /* never freed */

/* ---------------------------------------------------------- configuration */
typedef struct {
    const char *xbe_path;     /* extracted default.xbe */
    const char *game_root;    /* directory mapped to D:\ (extracted disc) */
    const char *iso_path;     /* optional: read D:\ files straight from the ISO */
    const char *save_root;    /* directory for T:\ U:\ Z:\ (saves, cache) */
    int trace_kernel;         /* log every kernel call */
    int trace_calls;          /* log every indirect call */
} RuntimeConfig;

extern RuntimeConfig g_cfg;

int  runtime_init(const RuntimeConfig *cfg);
void runtime_run(void);                 /* runs the game's entry point on this thread */
void runtime_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
GuestFn recomp_lookup(uint32_t addr);

/* ------------------------------------------------------------- threading */
/* All guest code runs under one global lock (like a GIL). Kernel calls that
 * block (waits, sleeps, yields) release it, so guest threads interleave at the
 * same points they would on a single-core Xbox. */
void guest_lock(void);
void guest_unlock(void);
CPU *cpu_new(uint32_t stack_size);
void cpu_setup_thread_block(CPU *c);
uint32_t current_kthread(CPU *c);
CPU *current_cpu(void);
void set_current_cpu(CPU *c);
uint64_t host_time_ns(void);
void guest_yield_lock(void);
extern volatile int g_vblank_waiting;   /* the ticker wants the guest lock for a due vblank */
void guest_wait_change(uint64_t deadline_ns);   /* GIL held; releases it while sleeping */
void guest_signal_change(void);

/* ------------------------------------------------------------------ kernel */
void kernel_init(void);
void kernel_call(CPU *c, uint32_t ordinal);
uint32_t kernel_export_address(uint32_t ordinal);

/* --------------------------------------------------------------------- HLE */
/* Replace a recompiled guest function with a host implementation. The handler
 * runs with the return address on the guest stack and must pop it plus any
 * stdcall arguments (use hle_return). */
typedef void (*HleFn)(CPU *c);
void hle_register(uint32_t guest_addr, const char *name, HleFn fn);
HleFn hle_lookup(uint32_t guest_addr, const char **name);
void hle_init(void);
void xinput_hle_init(void);
void kernel_set_event_handle(uint32_t handle);
static inline uint32_t hle_arg(CPU *c, int i) { return rd32(c->esp + 4 + 4 * (uint32_t)i); }
static inline void hle_return(CPU *c, uint32_t eax, int stdcall_args) {
    c->eax = eax; c->esp += 4 + 4 * (uint32_t)stdcall_args;
}

/* ---------------------------------------------------------------- platform */
/* Implemented by the host shell (macOS Metal app or headless runner). */
typedef struct {
    void (*present)(void *user, const uint32_t *bgra, int w, int h, int pitch);
    void (*clear)(void *user, float r, float g, float b, float a);
    /* Xbox gamepad state for port 0: digital buttons (XINPUT_GAMEPAD_* bits:
     * dpad up/down/left/right 0x1/2/4/8, start 0x10, back 0x20, thumbs 0x40/0x80),
     * analog buttons A B X Y black white LT RT (0-255), sticks LX LY RX RY. */
    void (*poll_input)(void *user, uint16_t *buttons, uint8_t analog[8], int16_t axes[4]);
    void *user;
} HostCallbacks;
extern HostCallbacks g_host;

/* apu.c */
int apu_owns(uint32_t a);
uint32_t apu_read(uint32_t a, int size, uint32_t mem_value);
int apu_write_pre(uint32_t a, uint32_t v, int size);
void apu_write(uint32_t a, uint32_t v, int size);
int apu_pull_audio(float *out, int frames);   /* host audio thread: 48 kHz interleaved stereo */
void apu_set_reverb_level(float level);       /* 0 off, 1 default, up to 4; HALO_REVERB overrides */
int apu_audio_buffered(void);
void apu_tick(uint64_t now_ns);
void apu_set_interrupt(uint32_t kinterrupt);

#ifdef __cplusplus
}
#endif
