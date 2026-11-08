/* kernel.c — high-level implementation of the xboxkrnl exports Halo imports.
 *
 * Every waitable object lives in guest memory as an NT DISPATCHER_HEADER, so
 * guest-initialized objects (KeInitializeEvent) and handle-based ones
 * (NtCreateEvent) share one wait implementation. Handles map to the guest
 * address of their object (or to a host file record).
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "kernel_names.h"
#include "runtime.h"
#include "vfs.h"
#include "nv2a.h"

#define STATUS_SUCCESS                0x00000000u
#define STATUS_TIMEOUT                0x00000102u
#define STATUS_PENDING                0x00000103u
#define STATUS_NO_MORE_FILES          0x80000006u
#define STATUS_UNSUCCESSFUL           0xC0000001u
#define STATUS_NOT_IMPLEMENTED        0xC0000002u
#define STATUS_INVALID_HANDLE         0xC0000008u
#define STATUS_INVALID_PARAMETER      0xC000000Du
#define STATUS_INVALID_DEVICE_REQUEST 0xC0000010u
#define STATUS_END_OF_FILE            0xC0000011u
#define STATUS_NO_MEMORY              0xC0000017u
#define STATUS_ACCESS_DENIED          0xC0000022u
#define STATUS_OBJECT_NAME_NOT_FOUND  0xC0000034u
#define STATUS_OBJECT_NAME_COLLISION  0xC0000035u
#define STATUS_OBJECT_PATH_NOT_FOUND  0xC000003Au
#define STATUS_FILE_IS_A_DIRECTORY    0xC00000BAu
#define STATUS_NOT_A_DIRECTORY        0xC0000103u

/* dispatcher object types */
enum { DO_NOTIFICATION_EVENT = 0, DO_SYNC_EVENT = 1, DO_MUTANT = 2, DO_SEMAPHORE = 5,
       DO_THREAD = 6, DO_NOTIFICATION_TIMER = 8, DO_SYNC_TIMER = 9 };

/* ------------------------------------------------------------- handles */
typedef enum { H_FREE, H_OBJECT, H_FILE, H_LINK } HKind;
typedef struct {
    HKind kind;
    uint32_t object;          /* guest address of dispatcher object / KTHREAD */
    /* files */
    int fd;                   /* host fd (or -1 for ISO-backed) */
    int iso_index;
    int is_dir;
    int delete_pending;       /* FileDispositionInformation / FILE_DELETE_ON_CLOSE: removed at close */
    uint64_t pos;
    char host[1024];
    char rel[512];
    int dir_cursor;
    char link_target[256];
} Handle;

#define MAX_HANDLES 4096
static Handle g_handles[MAX_HANDLES];
#define H2I(h) (((h) - 0x100u) / 4u)
#define I2H(i) ((uint32_t)(i) * 4u + 0x100u)
#define NT_CURRENT_THREAD 0xFFFFFFFEu

static uint32_t handle_new(HKind k) {
    for (int i = 1; i < MAX_HANDLES; i++)
        if (g_handles[i].kind == H_FREE) {
            memset(&g_handles[i], 0, sizeof(Handle));
            g_handles[i].kind = k;
            g_handles[i].fd = -1;
            g_handles[i].iso_index = -1;
            return I2H(i);
        }
    runtime_log("[kernel] out of handles");
    return 0;
}
static Handle *handle_get(uint32_t h) {
    uint32_t i = H2I(h);
    if (h < 0x100 || i >= MAX_HANDLES || g_handles[i].kind == H_FREE) return NULL;
    return &g_handles[i];
}

/* ---------------------------------------------------- kernel data area */
static uint32_t g_export_addr[400];   /* guest address for data exports */
static uint32_t g_tick_count_addr, g_obj_type_event, g_obj_type_thread;
static uint64_t g_boot_ns;

uint32_t kernel_export_address(uint32_t ordinal) {
    if (ordinal < 400 && g_export_addr[ordinal]) return g_export_addr[ordinal];
    return KERNEL_THUNK_BASE + ordinal * 4;
}

static uint32_t make_data(uint32_t ordinal, uint32_t size) {
    uint32_t a = kdata_alloc(size);
    memset(g_mem + a, 0, size);
    g_export_addr[ordinal] = a;
    return a;
}

static uint32_t make_ansi_string(const char *s) {
    uint32_t len = (uint32_t)strlen(s);
    uint32_t buf = kdata_alloc(len + 1);
    memcpy(g_mem + buf, s, len + 1);
    uint32_t str = kdata_alloc(8);
    wr16(str, (uint16_t)len);
    wr16(str + 2, (uint16_t)(len + 1));
    wr32(str + 4, buf);
    return str;
}

/* ------------------------------------------------------------- time */
static uint64_t now_100ns_since_1601(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t)ts.tv_sec + 11644473600ull) * 10000000ull + (uint64_t)ts.tv_nsec / 100;
}
static uint64_t interrupt_time_100ns(void) { return (host_time_ns() - g_boot_ns) / 100; }

/* Convert an NT timeout (LARGE_INTEGER*, 100 ns, negative = relative) to a
 * monotonic deadline. Returns 0 for "wait forever", 1 for "poll". */
static uint64_t timeout_deadline(uint32_t ptimeout) {
    if (!ptimeout) return 0;
    int64_t t = (int64_t)rd64(ptimeout);
    if (t == 0) return 1;
    if (t < 0) return host_time_ns() + (uint64_t)(-t) * 100ull;
    uint64_t now = now_100ns_since_1601();
    return host_time_ns() + ((uint64_t)t > now ? ((uint64_t)t - now) * 100ull : 0);
}

/* --------------------------------------------------------- dispatcher */
static uint8_t do_type(uint32_t o) { return rd8(o); }
static int32_t do_signal(uint32_t o) { return (int32_t)rd32(o + 4); }
static void do_set_signal(uint32_t o, int32_t v) { wr32(o + 4, (uint32_t)v); }

static void init_dispatcher(uint32_t o, uint8_t type, uint8_t size_dw, int32_t signal) {
    wr8(o, type);
    wr8(o + 1, 0);
    wr8(o + 2, size_dw);
    wr8(o + 3, 0);
    do_set_signal(o, signal);
    wr32(o + 8, o + 8);    /* empty wait list */
    wr32(o + 12, o + 8);
}

/* timers: KTIMER = header(16) DueTime(8) ListEntry(8) Dpc(4) Period(4) */
#define MAX_TIMERS 256
static uint32_t g_timers[MAX_TIMERS];
static uint64_t g_timer_due_ns[MAX_TIMERS];
static uint32_t g_timer_period_ms[MAX_TIMERS];
static int g_ntimers;

static int is_signaled(CPU *c, uint32_t o) {
    uint8_t t = do_type(o);
    if (t == DO_MUTANT) return do_signal(o) > 0 || rd32(o + 0x10) == current_kthread(c);
    return do_signal(o) > 0;
}

static void satisfy(CPU *c, uint32_t o) {
    switch (do_type(o)) {
    case DO_SYNC_EVENT: case DO_SYNC_TIMER: do_set_signal(o, 0); break;
    case DO_SEMAPHORE: do_set_signal(o, do_signal(o) - 1); break;
    case DO_MUTANT:
        do_set_signal(o, do_signal(o) - 1);
        wr32(o + 0x10, current_kthread(c));
        break;
    default: break;
    }
}

static uint32_t resolve_waitable(CPU *c, uint32_t h_or_obj, int is_handle) {
    if (!is_handle) return h_or_obj;
    if (h_or_obj == NT_CURRENT_THREAD) return current_kthread(c);
    Handle *h = handle_get(h_or_obj);
    if (!h || h->kind != H_OBJECT) return 0;
    return h->object;
}

/* ------------------------------------------------------- user APCs */
/* I/O completion routines (ReadFileEx/WriteFileEx) are queued per thread and
 * run when that thread enters an alertable wait, like the real kernel. */
#define STATUS_USER_APC 0x000000C0u
#define MAX_THREADS 64
#define MAX_APCS 64
typedef struct { uint32_t routine, ctx, iosb; } UserApc;
static UserApc g_apcq[MAX_THREADS][MAX_APCS];
static int g_apcn[MAX_THREADS];
static void run_guest(CPU *c, uint32_t fn, int nargs, const uint32_t *args);

static void queue_user_apc(CPU *c, uint32_t routine, uint32_t ctx, uint32_t iosb) {
    int t = c->thread_index % MAX_THREADS;
    if (g_apcn[t] < MAX_APCS) g_apcq[t][g_apcn[t]++] = (UserApc){routine, ctx, iosb};
    else runtime_log("[kernel] APC queue full for thread %d", c->thread_index);
}

/* Runs pending APCs for the current thread; returns how many ran. */
static int deliver_user_apcs(CPU *c) {
    int t = c->thread_index % MAX_THREADS, n = 0;
    while (g_apcn[t] > 0) {
        UserApc apc = g_apcq[t][0];
        memmove(&g_apcq[t][0], &g_apcq[t][1], sizeof(UserApc) * (size_t)(--g_apcn[t]));
        uint32_t saved_eax = c->eax;
        uint32_t args[3] = {apc.ctx, apc.iosb, 0};          /* PIO_APC_ROUTINE(ctx, iosb, reserved) */
        run_guest(c, apc.routine, 3, args);
        c->eax = saved_eax;
        n++;
    }
    return n;
}

/* Wait for one (wait_all=0: any) or all objects. Returns status. */
static uint32_t wait_objects_ex(CPU *c, int n, const uint32_t *objs, int wait_all, uint32_t ptimeout, int alertable) {
    uint64_t deadline = timeout_deadline(ptimeout);
    for (;;) {
        if (alertable && deliver_user_apcs(c)) return STATUS_USER_APC;
        if (wait_all) {
            int ok = 1;
            for (int i = 0; i < n; i++) ok &= is_signaled(c, objs[i]);
            if (ok) {
                for (int i = 0; i < n; i++) satisfy(c, objs[i]);
                return STATUS_SUCCESS;
            }
        } else {
            for (int i = 0; i < n; i++)
                if (is_signaled(c, objs[i])) {
                    satisfy(c, objs[i]);
                    return STATUS_SUCCESS + (uint32_t)i;
                }
        }
        if (deadline == 1) return STATUS_TIMEOUT;
        if (deadline && host_time_ns() >= deadline) return STATUS_TIMEOUT;
        guest_wait_change(deadline);
    }
}
static uint32_t wait_objects(CPU *c, int n, const uint32_t *objs, int wait_all, uint32_t ptimeout) {
    return wait_objects_ex(c, n, objs, wait_all, ptimeout, 0);
}

static void signal_object(uint32_t o) { do_set_signal(o, 1); guest_signal_change(); }

/* --------------------------------------------------------- ticker thread */
static CPU *g_dpc_cpu;
uint64_t g_pace_gil_max;
volatile int g_vblank_waiting;
uint64_t nv2a_next_vblank_ns(void);
unsigned g_pace_gil_over8;
static void run_guest(CPU *c, uint32_t fn, int nargs, const uint32_t *args) {
    for (int i = nargs - 1; i >= 0; i--) push32(c, args[i]);
    push32(c, 0xFFFFFFF0u);
    CPU *prev = current_cpu();
    set_current_cpu(c);
    call_indirect(c, fn);
    set_current_cpu(prev);
}

static void *ticker_main(void *arg) {
    (void)arg;
    for (;;) {
        usleep(1000);
        uint64_t w0 = host_time_ns();
        int urgent = w0 >= nv2a_next_vblank_ns();
        if (urgent) g_vblank_waiting = 1;
        guest_lock();
        g_vblank_waiting = 0;
        {   /* pacing diagnosis: how long the ticker waited for the guest lock */
            uint64_t w = host_time_ns() - w0;
            if (w > g_pace_gil_max) g_pace_gil_max = w;
            if (w > 8000000ull) g_pace_gil_over8++;
        }
        wr32(g_tick_count_addr, (uint32_t)((host_time_ns() - g_boot_ns) / 1000000ull));
        uint64_t now = host_time_ns();
        int fired = 0;
        for (int i = 0; i < g_ntimers; i++) {
            if (!g_timers[i] || !g_timer_due_ns[i] || now < g_timer_due_ns[i]) continue;
            uint32_t t = g_timers[i];
            do_set_signal(t, 1);
            fired = 1;
            uint32_t dpc = rd32(t + 0x20);
            if (g_timer_period_ms[i]) g_timer_due_ns[i] = now + g_timer_period_ms[i] * 1000000ull;
            else { g_timer_due_ns[i] = 0; g_timers[i] = 0; }
            if (dpc) {
                /* KDPC: Type(2) Inserted(1) pad(1) ListEntry(8) DeferredRoutine(+0xC) Context(+0x10) Arg1 Arg2 */
                uint32_t args[4] = {dpc, rd32(dpc + 0x10), (uint32_t)now, 0};
                run_guest(g_dpc_cpu, rd32(dpc + 0xC), 4, args);
            }
        }
        if (fired) guest_signal_change();
        nv2a_tick(now);
        guest_unlock();
    }
    return NULL;
}

static void timer_arm(uint32_t timer, uint64_t due_ns, uint32_t period_ms, uint32_t dpc) {
    wr32(timer + 0x20, dpc);
    wr32(timer + 0x24, period_ms);
    do_set_signal(timer, 0);
    int slot = -1;
    for (int i = 0; i < g_ntimers; i++) if (g_timers[i] == timer) slot = i;
    if (slot < 0) for (int i = 0; i < g_ntimers; i++) if (!g_timers[i]) { slot = i; break; }
    if (slot < 0 && g_ntimers < MAX_TIMERS) slot = g_ntimers++;
    if (slot < 0) { runtime_log("[kernel] too many timers"); return; }
    g_timers[slot] = timer;
    g_timer_due_ns[slot] = due_ns;
    g_timer_period_ms[slot] = period_ms;
}

static int timer_cancel(uint32_t timer) {
    for (int i = 0; i < g_ntimers; i++)
        if (g_timers[i] == timer) { int was = g_timer_due_ns[i] != 0; g_timers[i] = 0; return was; }
    return 0;
}

/* ------------------------------------------------------------ threads */
typedef struct { CPU *cpu; uint32_t routine, ctx, system_routine, kthread; int suspended; } ThreadStart;

static void *guest_thread_main(void *p) {
    ThreadStart *ts = p;
    guest_lock();
    while (rd32(ts->kthread + 0x1C0)) guest_wait_change(0);   /* suspend count */
    CPU *c = ts->cpu;
    set_current_cpu(c);
    runtime_log("[thread %d] start routine=%08x ctx=%08x", c->thread_index, ts->routine, ts->ctx);
    if (ts->system_routine) {
        uint32_t args[2] = {ts->routine, ts->ctx};
        run_guest(c, ts->system_routine, 2, args);
    } else {
        uint32_t args[1] = {ts->ctx};
        run_guest(c, ts->routine, 1, args);
    }
    signal_object(ts->kthread);
    runtime_log("[thread %d] exited", c->thread_index);
    guest_unlock();
    free(ts);
    return NULL;
}

/* ------------------------------------------------------------- strings */
static void read_ansi_string(uint32_t pstr, char *out, size_t cap) {
    if (!pstr) { out[0] = 0; return; }
    uint16_t len = rd16(pstr);
    uint32_t buf = rd32(pstr + 4);
    if (len >= cap) len = (uint16_t)(cap - 1);
    memcpy(out, g_mem + buf, len);
    out[len] = 0;
}

static void object_attributes_path(uint32_t poa, char *out, size_t cap) {
    /* OBJECT_ATTRIBUTES: RootDirectory, ObjectName (PSTRING), Attributes */
    uint32_t root = rd32(poa), name = rd32(poa + 4);
    char rel[512];
    read_ansi_string(name, rel, sizeof rel);
    Handle *rh = root ? handle_get(root) : NULL;
    if (rh && rh->kind == H_FILE) {
        /* relative to an open directory handle: rebuild an absolute disc/host path */
        if (rh->iso_index >= 0 || (rh->host[0] == 0))
            snprintf(out, cap, "\\Device\\CdRom0\\%s\\%s", rh->rel, rel);
        else
            snprintf(out, cap, "%s/%s", rh->host, rel);
        return;
    }
    snprintf(out, cap, "%s", rel);
}

/* ---------------------------------------------------------------- SHA-1 */
typedef struct { uint32_t h[5]; uint64_t len; uint8_t buf[64]; uint32_t n; } Sha1;
static uint32_t rol32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
static void sha1_block(Sha1 *s, const uint8_t *p) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
    for (int i = 16; i < 80; i++) w[i] = rol32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else { f = b ^ c ^ d; k = 0xCA62C1D6; }
        uint32_t t = rol32(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rol32(b, 30); b = a; a = t;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e;
}
static void sha1_init(Sha1 *s) {
    static const uint32_t iv[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    memcpy(s->h, iv, sizeof iv); s->len = 0; s->n = 0;
}
static void sha1_update(Sha1 *s, const uint8_t *p, uint32_t len) {
    s->len += len;
    while (len--) { s->buf[s->n++] = *p++; if (s->n == 64) { sha1_block(s, s->buf); s->n = 0; } }
}
static void sha1_final(Sha1 *s, uint8_t out[20]) {
    uint64_t bits = s->len * 8;
    uint8_t pad = 0x80;
    sha1_update(s, &pad, 1);
    pad = 0;
    while (s->n != 56) sha1_update(s, &pad, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha1_update(s, lb, 8);
    for (int i = 0; i < 5; i++) { out[4*i] = (uint8_t)(s->h[i] >> 24); out[4*i+1] = (uint8_t)(s->h[i] >> 16);
                                  out[4*i+2] = (uint8_t)(s->h[i] >> 8); out[4*i+3] = (uint8_t)s->h[i]; }
}
/* The guest SHA context is opaque (116 bytes on Xbox); we keep ours inside it. */
_Static_assert(sizeof(Sha1) <= 116, "SHA context too large");

/* ============================================================ exports */
typedef void (*KFn)(CPU *c, const uint32_t *a);
typedef struct { uint16_t ordinal; uint8_t nargs; uint8_t fastcall; KFn fn; } KExport;

#define K(name) static void k_##name(CPU *c, const uint32_t *a)
#define RET(v) do { c->eax = (uint32_t)(v); return; } while (0)
#define UNUSED_ARGS (void)a

/* ---- memory ---- */
K(NtAllocateVirtualMemory) {
    uint32_t pbase = a[0], psize = a[2];
    uint32_t base = rd32(pbase), size = rd32(psize);
    if (base) {   /* commit inside an existing reservation, or a fixed-address reserve */
        if (base >= GUEST_HEAP_BASE && base < GUEST_RAM_SIZE && !guest_alloc_size(base & ~0xFFFu))
            guest_reserve(base & ~0xFFFu, size);
        RET(STATUS_SUCCESS);
    }
    uint32_t got = guest_alloc(size, 0x1000);   /* page granularity: memory is tight on a 64 MB machine */
    if (!got) RET(STATUS_NO_MEMORY);
    wr32(pbase, got);
    wr32(psize, (size + 0xFFF) & ~0xFFFu);
    RET(STATUS_SUCCESS);
}
K(NtFreeVirtualMemory) {
    uint32_t base = rd32(a[0]);
    uint32_t type = a[2];
    if (type & 0x8000) guest_free(base);   /* MEM_RELEASE; MEM_DECOMMIT is a no-op */
    RET(STATUS_SUCCESS);
}
K(NtQueryVirtualMemory) {
    uint32_t base = a[0] & ~0xFFFu, mbi = a[1];
    uint32_t sz = guest_alloc_size(base);
    wr32(mbi + 0, base); wr32(mbi + 4, base); wr32(mbi + 8, 4);
    wr32(mbi + 12, sz ? sz : 0x1000); wr32(mbi + 16, sz ? 0x1000 : 0x10000 /* MEM_COMMIT : MEM_FREE */);
    wr32(mbi + 20, 4); wr32(mbi + 24, 0x20000);
    RET(STATUS_SUCCESS);
}
K(MmAllocateContiguousMemory) { RET(guest_alloc_contiguous(a[0], 0, 0xFFFFFFFFu, 0x1000)); }
K(MmAllocateContiguousMemoryEx) {
    /* (NumberOfBytes, LowestAcceptableAddress, HighestAcceptableAddress, Alignment, ProtectionType) */
    uint32_t r = guest_alloc_contiguous(a[0], a[1], a[2], a[3] > 0x1000 ? a[3] : 0x1000);
    if (a[0] >= 0x100000) runtime_log("[mem] contiguous %u KB in %08x-%08x -> %08x", a[0] >> 10, a[1], a[2], r);
    if (!r) {
        static unsigned n;
        if (++n <= 8) runtime_log("[mem]   (failed request from guest code at %08x, align %x, protect %x)", rd32(c->esp), a[3], a[4]);
    }
    RET(r);
}
K(MmFreeContiguousMemory) { guest_free(a[0]); RET(0); }
K(MmAllocateSystemMemory) { RET(guest_alloc(a[0], 0x1000)); }
K(MmFreeSystemMemory) { RET(guest_free(a[0])); }
K(MmQueryAllocationSize) { RET(guest_alloc_size(a[0])); }
K(MmPersistContiguousMemory) { UNUSED_ARGS; RET(0); }
K(MmSetAddressProtect) { UNUSED_ARGS; RET(0); }
K(MmQueryAddressProtect) { UNUSED_ARGS; RET(4 /* PAGE_READWRITE */); }
K(MmGetPhysicalAddress) { RET(guest_phys(a[0])); }
K(MmClaimGpuInstanceMemory) {
    if (a[1]) wr32(a[1], 0x10000);
    RET(0x83FF0000u);   /* top of 64 MB, tagged like the real kernel's GPU instance area */
}
K(MmLockUnlockBufferPages) { UNUSED_ARGS; RET(0); }
K(MmLockUnlockPhysicalPage) { UNUSED_ARGS; RET(0); }
K(MmQueryStatistics) {
    uint32_t s = a[0];
    if (rd32(s) != 36) RET(STATUS_INVALID_PARAMETER);
    uint32_t total = (GUEST_HEAP_END - GUEST_HEAP_BASE) / 4096;
    wr32(s + 4, 64u * 1024 * 1024 / 4096);
    wr32(s + 8, total - guest_heap_used() / 4096);
    wr32(s + 12, guest_heap_used());
    wr32(s + 16, guest_heap_used());
    for (int i = 20; i < 36; i += 4) wr32(s + (uint32_t)i, 0);
    RET(STATUS_SUCCESS);
}
/* Kernel pool: small blocks are carved out of 64 KB chunks in 16-byte size
 * classes (like the real pool), large ones get whole pages. A 16-byte header
 * before each small block records its size. */
#define POOL_CHUNK  0x10000u
#define POOL_MAXSML 2048u
static uint32_t g_pool_free[POOL_MAXSML / 16 + 1];   /* free-list heads per class */
static uint32_t g_pool_cur, g_pool_left;
static uint32_t pool_alloc(uint32_t size) {
    if (size == 0) size = 1;
    if (size > POOL_MAXSML) return guest_alloc(size, 16);
    uint32_t cls = (size + 15) / 16, bytes = cls * 16 + 16;
    uint32_t b = g_pool_free[cls];
    if (b) { g_pool_free[cls] = rd32(b); }
    else {
        if (g_pool_left < bytes) {
            g_pool_cur = guest_alloc(POOL_CHUNK, 16);
            if (!g_pool_cur) return 0;
            g_pool_left = POOL_CHUNK;
        }
        b = g_pool_cur + 16;
        g_pool_cur += bytes;
        g_pool_left -= bytes;
    }
    wr32(b - 16, 0x504F4F4Cu);    /* 'POOL' */
    wr32(b - 12, cls);
    memset(g_mem + b, 0, cls * 16);
    return b;
}
static int pool_is_small(uint32_t p) { return p >= 16 && rd32(p - 16) == 0x504F4F4Cu && rd32(p - 12) <= POOL_MAXSML / 16; }
static void pool_free(uint32_t p) {
    if (pool_is_small(p)) {
        uint32_t cls = rd32(p - 12);
        wr32(p - 16, 0);
        wr32(p, g_pool_free[cls]);
        g_pool_free[cls] = p;
    } else guest_free(p);
}
static uint32_t pool_size(uint32_t p) { return pool_is_small(p) ? rd32(p - 12) * 16 : guest_alloc_size(p); }

K(ExAllocatePool) { RET(pool_alloc(a[0])); }
K(ExAllocatePoolWithTag) { RET(pool_alloc(a[0])); }
K(ExFreePool) { pool_free(a[0]); RET(0); }
K(ExQueryPoolBlockSize) { RET(pool_size(a[0])); }

/* ---- critical sections (RTL_CRITICAL_SECTION: header(16) LockCount RecursionCount OwningThread) ---- */
K(RtlInitializeCriticalSection) {
    uint32_t cs = a[0];
    init_dispatcher(cs, DO_SYNC_EVENT, 4, 0);
    wr32(cs + 0x10, 0xFFFFFFFFu);
    wr32(cs + 0x14, 0);
    wr32(cs + 0x18, 0);
    RET(0);
}
K(RtlEnterCriticalSection) {
    uint32_t cs = a[0], me = current_kthread(c);
    if (rd32(cs + 0x18) == me) { wr32(cs + 0x14, rd32(cs + 0x14) + 1); wr32(cs + 0x10, rd32(cs + 0x10) + 1); RET(0); }
    while (rd32(cs + 0x18) != 0) guest_wait_change(0);
    wr32(cs + 0x18, me);
    wr32(cs + 0x14, 1);
    wr32(cs + 0x10, rd32(cs + 0x10) + 1);
    RET(0);
}
K(RtlLeaveCriticalSection) {
    uint32_t cs = a[0];
    uint32_t rc = rd32(cs + 0x14) - 1;
    wr32(cs + 0x14, rc);
    wr32(cs + 0x10, rd32(cs + 0x10) - 1);
    if (rc == 0) { wr32(cs + 0x18, 0); guest_signal_change(); }
    RET(0);
}

/* ---- events / mutants / semaphores ---- */
static uint32_t new_object_handle(uint32_t obj) {
    uint32_t h = handle_new(H_OBJECT);
    if (h) handle_get(h)->object = obj;
    return h;
}
K(NtCreateEvent) {
    uint32_t o = kdata_alloc(16);
    init_dispatcher(o, a[2] ? DO_SYNC_EVENT : DO_NOTIFICATION_EVENT, 4, a[3] ? 1 : 0);
    wr32(a[0], new_object_handle(o));
    RET(STATUS_SUCCESS);
}
K(NtSetEvent) {
    Handle *h = handle_get(a[0]);
    if (!h) RET(STATUS_INVALID_HANDLE);
    if (a[1]) wr32(a[1], (uint32_t)do_signal(h->object));
    signal_object(h->object);
    RET(STATUS_SUCCESS);
}
/* Used by HLE code (XInputSetState completion). */
void kernel_set_event_handle(uint32_t handle) {
    Handle *h = handle_get(handle);
    if (h) signal_object(h->object);
}
K(NtClearEvent) {
    Handle *h = handle_get(a[0]);
    if (!h) RET(STATUS_INVALID_HANDLE);
    do_set_signal(h->object, 0);
    RET(STATUS_SUCCESS);
}
K(NtCreateMutant) {
    uint32_t o = kdata_alloc(0x20);
    init_dispatcher(o, DO_MUTANT, 8, a[2] ? 0 : 1);
    wr32(o + 0x10, a[2] ? current_kthread(c) : 0);
    wr32(a[0], new_object_handle(o));
    RET(STATUS_SUCCESS);
}
K(NtReleaseMutant) {
    Handle *h = handle_get(a[0]);
    if (!h) RET(STATUS_INVALID_HANDLE);
    int32_t s = do_signal(h->object);
    if (a[1]) wr32(a[1], (uint32_t)s);
    do_set_signal(h->object, s + 1);
    if (s + 1 > 0) wr32(h->object + 0x10, 0);
    guest_signal_change();
    RET(STATUS_SUCCESS);
}
K(KeInitializeTimerEx) {
    init_dispatcher(a[0], a[1] ? DO_SYNC_TIMER : DO_NOTIFICATION_TIMER, 10, 0);
    wr64(a[0] + 0x10, 0);
    RET(0);
}
static uint64_t due_to_deadline(int64_t due) {
    if (due < 0) return host_time_ns() + (uint64_t)(-due) * 100ull;
    uint64_t now = now_100ns_since_1601();
    return host_time_ns() + ((uint64_t)due > now ? ((uint64_t)due - now) * 100ull : 0);
}
K(KeSetTimer) {    /* (Timer, DueTime.lo, DueTime.hi, Dpc) */
    int was = timer_cancel(a[0]);
    timer_arm(a[0], due_to_deadline((int64_t)((uint64_t)a[2] << 32 | a[1])), 0, a[3]);
    RET(was);
}
K(KeSetTimerEx) {  /* (Timer, DueTime.lo, DueTime.hi, Period, Dpc) */
    int was = timer_cancel(a[0]);
    timer_arm(a[0], due_to_deadline((int64_t)((uint64_t)a[2] << 32 | a[1])), a[3], a[4]);
    RET(was);
}
K(KeCancelTimer) { RET(timer_cancel(a[0])); }
K(KeInitializeDpc) {
    wr16(a[0], 19);     /* DpcObject */
    wr8(a[0] + 2, 0);
    wr32(a[0] + 0xC, a[1]);
    wr32(a[0] + 0x10, a[2]);
    RET(0);
}
K(KeInsertQueueDpc) {
    /* Run immediately on the DPC context (we have no interrupt levels). */
    uint32_t args[4] = {a[0], rd32(a[0] + 0x10), a[1], a[2]};
    run_guest(g_dpc_cpu, rd32(a[0] + 0xC), 4, args);
    RET(1);
}
K(KeRemoveQueueDpc) { UNUSED_ARGS; RET(0); }
K(KeSetEvent) {
    int32_t prev = do_signal(a[0]);
    signal_object(a[0]);
    RET(prev);
}
K(KeWaitForSingleObject) { uint32_t o = a[0]; RET(wait_objects_ex(c, 1, &o, 0, a[4], a[3] & 0xFF)); }
K(KeWaitForMultipleObjects) {
    uint32_t objs[64], n = a[0] > 64 ? 64 : a[0];
    for (uint32_t i = 0; i < n; i++) objs[i] = rd32(a[1] + 4 * i);
    RET(wait_objects(c, (int)n, objs, a[2] == 0 /* WaitAll */, a[6]));
}
K(NtWaitForSingleObject) {
    uint32_t o = resolve_waitable(c, a[0], 1);
    if (!o) RET(STATUS_INVALID_HANDLE);
    RET(wait_objects(c, 1, &o, 0, a[2]));
}
K(NtWaitForSingleObjectEx) {
    uint32_t o = resolve_waitable(c, a[0], 1);
    if (!o) RET(STATUS_INVALID_HANDLE);
    RET(wait_objects_ex(c, 1, &o, 0, a[3], a[2] & 0xFF));
}
K(NtWaitForMultipleObjectsEx) {
    uint32_t objs[64], n = a[0] > 64 ? 64 : a[0];
    for (uint32_t i = 0; i < n; i++) {
        objs[i] = resolve_waitable(c, rd32(a[1] + 4 * i), 1);
        if (!objs[i]) RET(STATUS_INVALID_HANDLE);
    }
    RET(wait_objects_ex(c, (int)n, objs, a[2] == 0, a[5], a[4] & 0xFF));
}

/* ---- threads ---- */
K(PsCreateSystemThreadEx) {
    /* (ThreadHandle*, ExtSize, KernelStackSize, TlsDataSize, ThreadId*, StartRoutine,
        StartContext, CreateSuspended, DebuggerThread, SystemRoutine) */
    uint32_t stack = a[2] ? a[2] : 0x10000;
    CPU *nc = cpu_new(stack < 0x10000 ? 0x10000 : stack);
    uint32_t kthread = current_kthread(nc);
    init_dispatcher(kthread, DO_THREAD, 0x40, 0);
    wr32(kthread + 0x1C0, a[7] ? 1 : 0);
    uint32_t h = new_object_handle(kthread);
    if (a[0]) wr32(a[0], h);
    if (a[4]) wr32(a[4], (uint32_t)nc->thread_index + 1);
    ThreadStart *ts = calloc(1, sizeof *ts);
    ts->cpu = nc; ts->routine = a[5]; ts->ctx = a[6]; ts->system_routine = a[9]; ts->kthread = kthread;
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 8u << 20);   /* host stack for deep recompiled call chains */
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_create(&th, &at, guest_thread_main, ts);
    RET(STATUS_SUCCESS);
}
K(PsTerminateSystemThread) {
    UNUSED_ARGS;
    uint32_t kt = current_kthread(c);
    runtime_log("[thread %d] PsTerminateSystemThread", c->thread_index);
    signal_object(kt);
    guest_unlock();
    pthread_exit(NULL);
}
K(NtResumeThread) {
    Handle *h = handle_get(a[0]);
    if (!h) RET(STATUS_INVALID_HANDLE);
    uint32_t cnt = rd32(h->object + 0x1C0);
    if (a[1]) wr32(a[1], cnt);
    if (cnt) wr32(h->object + 0x1C0, cnt - 1);
    guest_signal_change();
    RET(STATUS_SUCCESS);
}
K(KeSetBasePriorityThread) { UNUSED_ARGS; RET(0); }
K(ObReferenceObjectByHandle) {
    uint32_t o = resolve_waitable(c, a[0], 1);
    if (!o) RET(STATUS_INVALID_HANDLE);
    wr32(a[2], o);
    RET(STATUS_SUCCESS);
}
K(ObfDereferenceObject) { UNUSED_ARGS; RET(0); }
K(KeDelayExecutionThread) {
    /* (WaitMode, Alertable, Interval) */
    if ((a[1] & 0xFF) && deliver_user_apcs(c)) RET(STATUS_USER_APC);
    uint64_t deadline = timeout_deadline(a[2]);
    if (a[1] & 0xFF) {
        while (host_time_ns() < deadline || deadline <= 1) {
            if (deliver_user_apcs(c)) RET(STATUS_USER_APC);
            if (deadline <= 1) break;
            guest_wait_change(deadline);
        }
        RET(STATUS_SUCCESS);
    }
    if (deadline <= 1) { guest_unlock(); sched_yield(); guest_lock(); RET(STATUS_SUCCESS); }
    while (host_time_ns() < deadline) guest_wait_change(deadline);
    RET(STATUS_SUCCESS);
}
K(NtYieldExecution) { UNUSED_ARGS; guest_unlock(); sched_yield(); guest_lock(); RET(STATUS_SUCCESS); }
K(KeRaiseIrqlToDpcLevel) { UNUSED_ARGS; uint8_t old = rd8(c->fs_base + 0x24); wr8(c->fs_base + 0x24, 2); RET(old); }
K(KfRaiseIrql) { uint8_t old = rd8(c->fs_base + 0x24); wr8(c->fs_base + 0x24, (uint8_t)a[0]); RET(old); }
K(KfLowerIrql) { wr8(c->fs_base + 0x24, (uint8_t)a[0]); RET(0); }
K(KeSaveFloatingPointState) { UNUSED_ARGS; RET(STATUS_SUCCESS); }
K(KeRestoreFloatingPointState) { UNUSED_ARGS; RET(STATUS_SUCCESS); }
K(KeStallExecutionProcessor) {
    /* microsecond stalls: a host sleep lasts far longer than asked (and runs
     * with the guest lock held), so short ones spin */
    if (a[0] > 200) usleep(a[0]);
    else { uint64_t end = host_time_ns() + (uint64_t)a[0] * 1000u; while (host_time_ns() < end) { } }
    RET(0);
}

/* ---- interrupts (only reached by low-level driver code) ---- */
K(HalGetInterruptVector) { if (a[1]) wr8(a[1], (uint8_t)(a[0] + 1)); RET(a[0] + 0x30); }
/* KINTERRUPT (our layout inside the guest object): +0 ServiceRoutine, +4 ServiceContext,
 * +8 BusInterruptLevel(vector), +0xC connected */
K(KeInitializeInterrupt) {
    /* (Interrupt, ServiceRoutine, ServiceContext, Vector, Irql, InterruptMode, ShareVector) */
    wr32(a[0], a[1]); wr32(a[0] + 4, a[2]); wr32(a[0] + 8, a[3]); wr32(a[0] + 0xC, 0);
    RET(0);
}
K(KeConnectInterrupt) {
    wr32(a[0] + 0xC, 1);
    nv2a_register_interrupt(rd32(a[0] + 8) - 0x30, a[0]);
    RET(1);
}
K(KeDisconnectInterrupt) { wr32(a[0] + 0xC, 0); nv2a_register_interrupt(rd32(a[0] + 8) - 0x30, 0); RET(1); }
K(KeSynchronizeExecution) {
    uint32_t args[1] = {a[2]};
    run_guest(c, a[1], 1, args);
    /* eax holds the routine's result */
}

/* Called by device models (with the GIL held) to deliver a hardware interrupt:
 * runs ServiceRoutine(Interrupt, ServiceContext) on the DPC context. */
void kernel_raise_interrupt(uint32_t kinterrupt) {
    if (!kinterrupt || !rd32(kinterrupt + 0xC)) return;
    uint32_t args[2] = {kinterrupt, rd32(kinterrupt + 4)};
    run_guest(g_dpc_cpu, rd32(kinterrupt), 2, args);
    guest_signal_change();
}
K(HalRegisterShutdownNotification) { UNUSED_ARGS; RET(0); }
K(HalIsResetOrShutdownPending) { UNUSED_ARGS; RET(0); }
K(HalInitiateShutdown) { UNUSED_ARGS; runtime_log("[kernel] HalInitiateShutdown"); exit(0); }
K(HalReturnToFirmware) { runtime_log("[kernel] HalReturnToFirmware(%u): game requested reboot/exit", a[0]); exit(0); }
K(KeBugCheck) { recomp_trap(c, a[0], "KeBugCheck"); }

/* ---- time ---- */
K(KeQuerySystemTime) { wr64(a[0], now_100ns_since_1601()); RET(0); }
K(KeQueryInterruptTime) { UNUSED_ARGS; uint64_t t = interrupt_time_100ns(); c->edx = (uint32_t)(t >> 32); RET((uint32_t)t); }
K(KeQueryPerformanceCounter) {
    UNUSED_ARGS;
    uint32_t hi, lo = guest_rdtsc_lo(c, &hi);
    c->edx = hi;
    RET(lo);
}
K(KeQueryPerformanceFrequency) { UNUSED_ARGS; c->edx = 0; RET(733333333u); }
K(RtlTimeToTimeFields) {
    uint64_t t = rd64(a[0]);
    time_t secs = (time_t)(t / 10000000ull - 11644473600ull);
    struct tm tm;
    gmtime_r(&secs, &tm);
    uint32_t f = a[1];
    wr16(f + 0, (uint16_t)(tm.tm_year + 1900)); wr16(f + 2, (uint16_t)(tm.tm_mon + 1));
    wr16(f + 4, (uint16_t)tm.tm_mday); wr16(f + 6, (uint16_t)tm.tm_hour);
    wr16(f + 8, (uint16_t)tm.tm_min); wr16(f + 10, (uint16_t)tm.tm_sec);
    wr16(f + 12, (uint16_t)((t / 10000) % 1000)); wr16(f + 14, (uint16_t)tm.tm_wday);
    RET(0);
}
K(RtlTimeFieldsToTime) {
    uint32_t f = a[0];
    struct tm tm = {0};
    tm.tm_year = rd16(f) - 1900; tm.tm_mon = rd16(f + 2) - 1; tm.tm_mday = rd16(f + 4);
    tm.tm_hour = rd16(f + 6); tm.tm_min = rd16(f + 8); tm.tm_sec = rd16(f + 10);
    time_t secs = timegm(&tm);
    if (secs == (time_t)-1) RET(0);
    wr64(a[1], ((uint64_t)secs + 11644473600ull) * 10000000ull + rd16(f + 12) * 10000ull);
    RET(1);
}

/* ---- strings / rtl ---- */
K(RtlInitAnsiString) {
    uint32_t d = a[0], s = a[1];
    uint32_t len = s ? (uint32_t)strlen((const char *)g_mem + s) : 0;
    wr16(d, (uint16_t)len);
    wr16(d + 2, (uint16_t)(s ? len + 1 : 0));
    wr32(d + 4, s);
    RET(0);
}
K(RtlUnicodeStringToAnsiString) {
    uint32_t d = a[0], s = a[1];
    uint16_t len = rd16(s) / 2;
    uint32_t src = rd32(s + 4);
    if (a[2]) {
        uint32_t buf = guest_alloc(len + 1u, 16);
        wr32(d + 4, buf);
        wr16(d + 2, (uint16_t)(len + 1));
    }
    uint32_t buf = rd32(d + 4);
    uint16_t max = rd16(d + 2);
    uint16_t n = len < max ? len : (uint16_t)(max ? max - 1 : 0);
    for (uint16_t i = 0; i < n; i++) { uint16_t ch = rd16(src + 2u * i); wr8(buf + i, ch < 0x100 ? (uint8_t)ch : '?'); }
    if (n < max) wr8(buf + n, 0);
    wr16(d, n);
    RET(STATUS_SUCCESS);
}
K(RtlEqualString) {
    char x[512], y[512];
    read_ansi_string(a[0], x, sizeof x);
    read_ansi_string(a[1], y, sizeof y);
    RET(a[2] ? !strcasecmp(x, y) : !strcmp(x, y));
}
K(RtlCompareMemoryUlong) {
    uint32_t n = 0;
    while (n + 4 <= a[1] && rd32(a[0] + n) == a[2]) n += 4;
    RET(n);
}
static uint32_t status_to_dos(uint32_t st);
K(RtlNtStatusToDosError) { RET(status_to_dos(a[0])); }
/* XAPI's ReadFileEx/WriteFileEx pass this as the APC routine, with the
 * caller's completion routine as ApcContext and &OVERLAPPED as the IO status
 * block. It calls CompletionRoutine(dwErrorCode, dwBytes, lpOverlapped). */
K(NtUserIoApcDispatcher) {
    uint32_t iosb = a[1], st = rd32(iosb);
    uint32_t err = (st & 0xC0000000u) == 0xC0000000u ? status_to_dos(st) : 0;
    uint32_t args[3] = {err, rd32(iosb + 4), iosb};
    run_guest(c, a[0], 3, args);
    RET(0);
}
static uint32_t status_to_dos(uint32_t st) {
    switch (st) {
    case STATUS_SUCCESS: return 0;
    case STATUS_OBJECT_NAME_NOT_FOUND: return 2;
    case STATUS_OBJECT_PATH_NOT_FOUND: return 3;
    case STATUS_ACCESS_DENIED: return 5;
    case STATUS_INVALID_HANDLE: return 6;
    case STATUS_NO_MEMORY: return 8;
    case STATUS_END_OF_FILE: return 38;
    case STATUS_OBJECT_NAME_COLLISION: return 183;
    case STATUS_NO_MORE_FILES: return 18;
    case STATUS_TIMEOUT: return 1460;
    default: return 317 /* ERROR_MR_MID_NOT_FOUND */;
    }
}
K(RtlRaiseException) { recomp_trap(c, rd32(a[0]), "RtlRaiseException (guest SEH not implemented yet)"); }
K(RtlUnwind) { recomp_trap(c, a[1], "RtlUnwind (guest SEH not implemented yet)"); }
K(DbgPrint) {
    UNUSED_ARGS;
    const char *fmt = (const char *)g_mem + rd32(c->esp + 4);
    runtime_log("[DbgPrint] %.200s", fmt);
    RET(0);
}

/* ---- settings / A/V ---- */
K(ExQueryNonVolatileSetting) {
    /* (ValueIndex, Type*, Value*, ValueLength, ResultLength*) */
    uint32_t idx = a[0], val = 0;
    switch (idx) {
    case 0x7: val = 1; break;            /* language: English */
    case 0x8: val = 0x00400000; break;    /* video flags: widescreen off, 480p enabled */
    case 0x9: val = 0x00010001; break;    /* audio: stereo */
    case 0x103: val = 1; break;          /* game region: North America */
    case 0x104: val = 0x00400100; break;  /* AV region: NTSC-M */
    case 0xA: val = 0; break;            /* parental control */
    default: break;
    }
    if (a[1]) wr32(a[1], 4 /* REG_DWORD */);
    if (a[2] && a[3] >= 4) wr32(a[2], val);
    if (a[4]) wr32(a[4], 4);
    RET(STATUS_SUCCESS);
}
K(AvGetSavedDataAddress) { UNUSED_ARGS; RET(0); }
K(AvSetSavedDataAddress) { UNUSED_ARGS; RET(0); }
K(AvSendTVEncoderOption) {
    /* option 6 = query AV capabilities; report a standard composite/S-Video pack */
    if (a[3]) wr32(a[3], a[1] == 6 ? 0x00000001u /* AV_PACK_STANDARD */ : 0);
    RET(0);
}
K(AvSetDisplayMode) {
    /* (RegisterBase, Step, Mode, Format, Pitch, FrameBuffer) */
    nv2a_set_display(a[5], a[4], a[2], a[3]);
    RET(0);
}

/* ---- sections ---- */
K(XeLoadSection) { wr32(a[0] + 0x18, rd32(a[0] + 0x18) + 1); RET(STATUS_SUCCESS); } /* all sections preloaded */
K(XeUnloadSection) { UNUSED_ARGS; RET(STATUS_SUCCESS); }

/* ---- crypto ---- */
K(XcSHAInit) { Sha1 s; sha1_init(&s); memcpy(g_mem + a[0], &s, sizeof s); RET(0); }
K(XcSHAUpdate) { Sha1 s; memcpy(&s, g_mem + a[0], sizeof s); sha1_update(&s, g_mem + a[1], a[2]); memcpy(g_mem + a[0], &s, sizeof s); RET(0); }
K(XcSHAFinal) { Sha1 s; memcpy(&s, g_mem + a[0], sizeof s); sha1_final(&s, g_mem + a[1]); RET(0); }
K(XcHMAC) {
    /* (Key, KeyLen, Data1, Len1, Data2, Len2, Digest) — HMAC-SHA1 */
    uint8_t k[64] = {0}, ipad[64], opad[64], inner[20];
    if (a[1] > 64) { Sha1 t; sha1_init(&t); sha1_update(&t, g_mem + a[0], a[1]); sha1_final(&t, k); }
    else memcpy(k, g_mem + a[0], a[1]);
    for (int i = 0; i < 64; i++) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5C; }
    Sha1 s;
    sha1_init(&s); sha1_update(&s, ipad, 64);
    if (a[2]) sha1_update(&s, g_mem + a[2], a[3]);
    if (a[4]) sha1_update(&s, g_mem + a[4], a[5]);
    sha1_final(&s, inner);
    sha1_init(&s); sha1_update(&s, opad, 64); sha1_update(&s, inner, 20);
    sha1_final(&s, g_mem + a[6]);
    RET(0);
}
K(XcRC4Key) {
    uint8_t *S = g_mem + a[0], *key = g_mem + a[2];
    for (int i = 0; i < 256; i++) S[i] = (uint8_t)i;
    for (int i = 0, j = 0; i < 256; i++) {
        j = (j + S[i] + key[i % a[1]]) & 255;
        uint8_t t = S[i]; S[i] = S[j]; S[j] = t;
    }
    S[256] = S[257] = 0;
    RET(0);
}
K(XcRC4Crypt) {
    uint8_t *S = g_mem + a[0], *d = g_mem + a[2];
    uint8_t i = S[256], j = S[257];
    for (uint32_t n = 0; n < a[1]; n++) {
        i++; j = (uint8_t)(j + S[i]);
        uint8_t t = S[i]; S[i] = S[j]; S[j] = t;
        d[n] ^= S[(uint8_t)(S[i] + S[j])];
    }
    S[256] = i; S[257] = j;
    RET(0);
}

/* Network crypto (system link). Single-player never sends anything, so the
 * DES/3DES and modexp calls are pass-through stand-ins for now; only the key
 * parity routine is real. TODO: real implementations before enabling LAN play. */
K(XcDESKeyParity) {
    uint8_t *k = g_mem + a[0];
    for (uint32_t i = 0; i < a[1]; i++) {
        uint8_t b = k[i] & 0xFE;
        k[i] = (uint8_t)(b | !__builtin_parity(b));
    }
    RET(0);
}
K(XcKeyTable) { /* (dwCipher, pbKeyTable, pbKey) */ memcpy(g_mem + a[1], g_mem + a[2], a[0] ? 24 : 8); RET(0); }
K(XcBlockCryptCBC) {
    /* (dwCipher, dwInputLength, pbOutput, pbInput, pbKeyTable, dwOp, pbFeedback) */
    memmove(g_mem + a[2], g_mem + a[3], a[1]);
    RET(0);
}
K(XcModExp) {
    /* (pA, pB, pC, pD, dwN): A = B^C mod D, dwN dwords. Stand-in: copies B. */
    memmove(g_mem + a[0], g_mem + a[1], a[4] * 4u);
    RET(1);
}

/* ---- network PHY (system link): report no cable ---- */
K(PhyGetLinkState) { UNUSED_ARGS; RET(0); }
K(PhyInitialize) { UNUSED_ARGS; RET(STATUS_SUCCESS); }

/* ---- files ---- */
static void set_iosb(uint32_t iosb, uint32_t status, uint32_t info) {
    if (iosb) { wr32(iosb, status); wr32(iosb + 4, info); }
}

static uint32_t open_common(CPU *c, uint32_t phandle, uint32_t poa, uint32_t iosb,
                            uint32_t disposition, uint32_t options, uint32_t access) {
    (void)c;
    char xp[1024];
    object_attributes_path(poa, xp, sizeof xp);
    VfsPath vp;
    int is_host_abs = xp[0] == '/';
    if (is_host_abs) { memset(&vp, 0, sizeof vp); vp.kind = VFS_HOST; snprintf(vp.host, sizeof vp.host, "%s", xp); }
    else if (vfs_resolve(xp, &vp)) {
        if (g_cfg.trace_kernel) runtime_log("[file] cannot resolve '%s'", xp);
        set_iosb(iosb, STATUS_OBJECT_PATH_NOT_FOUND, 0);
        return STATUS_OBJECT_PATH_NOT_FOUND;
    }
    uint32_t h = handle_new(H_FILE);
    Handle *hd = handle_get(h);
    snprintf(hd->rel, sizeof hd->rel, "%s", vp.rel);
    uint32_t info = 1 /* FILE_OPENED */;
    if (vp.kind == VFS_DISC && vfs_have_iso()) {
        if (vp.iso_index < 0 && !vp.is_dir) { hd->kind = H_FREE; set_iosb(iosb, STATUS_OBJECT_NAME_NOT_FOUND, 0); return STATUS_OBJECT_NAME_NOT_FOUND; }
        if (disposition != 1 && disposition != 3) { hd->kind = H_FREE; return STATUS_ACCESS_DENIED; }
        hd->iso_index = vp.iso_index;
        hd->is_dir = vp.is_dir;
    } else {
        snprintf(hd->host, sizeof hd->host, "%s", vp.host);
        struct stat st;
        int exists = stat(vp.host, &st) == 0;
        int dir = exists ? S_ISDIR(st.st_mode) : (options & 1);
        if (!exists && (disposition == 1 || disposition == 4)) {
            if (g_cfg.trace_kernel) runtime_log("[file] not found: '%s'", xp);
            hd->kind = H_FREE;
            set_iosb(iosb, STATUS_OBJECT_NAME_NOT_FOUND, 5);
            return STATUS_OBJECT_NAME_NOT_FOUND;
        }
        if (exists && disposition == 2) { hd->kind = H_FREE; set_iosb(iosb, STATUS_OBJECT_NAME_COLLISION, 4); return STATUS_OBJECT_NAME_COLLISION; }
        if (dir) {
            if (!exists) { mkdir(vp.host, 0755); info = 2; }
            hd->is_dir = 1;
        } else {
            if ((options & 1)) { hd->kind = H_FREE; return STATUS_NOT_A_DIRECTORY; }
            int flags = (access & 0x40000000u /* GENERIC_WRITE */) || (access & 0x2 /* FILE_WRITE_DATA */) ? O_RDWR : O_RDONLY;
            if (disposition == 0 || disposition == 2 || disposition == 3 || disposition == 5) flags |= O_CREAT;
            if (disposition == 0 || disposition == 4 || disposition == 5) flags |= O_TRUNC;
            if (vp.kind == VFS_DISC) flags = O_RDONLY;
            hd->fd = open(vp.host, flags, 0644);
            if (hd->fd < 0 && (flags & O_RDWR)) hd->fd = open(vp.host, O_RDONLY);
            if (hd->fd < 0) {
                if (g_cfg.trace_kernel) runtime_log("[file] not found: '%s' (%s)", xp, vp.host);
                hd->kind = H_FREE; set_iosb(iosb, STATUS_OBJECT_NAME_NOT_FOUND, 5); return STATUS_OBJECT_NAME_NOT_FOUND;
            }
            if (!exists) info = 2;
            else if (flags & O_TRUNC) info = 3;
        }
    }
    if ((options & 0x1000 /* FILE_DELETE_ON_CLOSE */) && hd->host[0] && vp.kind != VFS_DISC) hd->delete_pending = 1;
    if (g_cfg.trace_kernel) runtime_log("[file] open '%s' -> handle %x", xp, h);
    wr32(phandle, h);
    set_iosb(iosb, STATUS_SUCCESS, info);
    return STATUS_SUCCESS;
}

K(NtCreateFile) { RET(open_common(c, a[0], a[2], a[3], a[7], a[8], a[1])); }
K(NtOpenFile) { RET(open_common(c, a[0], a[2], a[3], 1, a[5], a[1])); }

static uint64_t file_size(Handle *h) {
    if (h->iso_index >= 0) {
        VfsPath vp;
        char p[600];
        snprintf(p, sizeof p, "\\Device\\CdRom0\\%s", h->rel);
        if (!vfs_resolve(p, &vp)) return vp.size;
        return 0;
    }
    struct stat st;
    if (h->fd >= 0 && !fstat(h->fd, &st)) return (uint64_t)st.st_size;
    return 0;
}

static void complete_io(CPU *c, uint32_t event, uint32_t apc, uint32_t apc_ctx, uint32_t iosb) {
    if (event) {
        Handle *e = handle_get(event);
        if (e && e->kind == H_OBJECT) signal_object(e->object);
    }
    if (apc) queue_user_apc(c, apc, apc_ctx, iosb);
}

K(NtReadFile) {
    /* (FileHandle, Event, ApcRoutine, ApcContext, IoStatusBlock, Buffer, Length, ByteOffset*) */
    Handle *h = handle_get(a[0]);
    if (!h || h->kind != H_FILE) RET(STATUS_INVALID_HANDLE);
    uint64_t pos = a[7] ? rd64(a[7]) : h->pos;
    if (a[7] && (int64_t)pos == -2) pos = h->pos;    /* FILE_USE_FILE_POINTER_POSITION */
    uint32_t got = 0;
    if (h->is_dir && h->fd < 0 && h->iso_index < 0) {          /* raw read of a partition volume */
        memset(g_mem + a[5], 0, a[6]);
        got = a[6];
    } else if (h->iso_index >= 0) vfs_iso_read(h->iso_index, pos, g_mem + a[5], a[6], &got);
    else if (h->fd >= 0) { ssize_t r = pread(h->fd, g_mem + a[5], a[6], (off_t)pos); got = r > 0 ? (uint32_t)r : 0; }
    h->pos = pos + got;
    uint32_t st = (got == 0 && a[6] != 0) ? STATUS_END_OF_FILE : STATUS_SUCCESS;
    set_iosb(a[4], st, got);
    complete_io(c, a[1], a[2], a[3], a[4]);
    RET(st);
}
K(NtWriteFile) {
    Handle *h = handle_get(a[0]);
    if (h && h->kind == H_FILE && h->is_dir && h->fd < 0 && h->iso_index < 0) {
        /* raw sector write to a partition volume (e.g. XAPI formatting the utility
         * drive). Partitions are host folders, so the write is accepted and dropped. */
        set_iosb(a[4], STATUS_SUCCESS, a[6]);
        complete_io(c, a[1], a[2], a[3], a[4]);
        RET(STATUS_SUCCESS);
    }
    if (!h || h->kind != H_FILE || h->fd < 0) RET(STATUS_ACCESS_DENIED);
    uint64_t pos = a[7] ? rd64(a[7]) : h->pos;
    if (a[7] && (int64_t)pos == -2) pos = h->pos;
    ssize_t r = pwrite(h->fd, g_mem + a[5], a[6], (off_t)pos);
    uint32_t put = r > 0 ? (uint32_t)r : 0;
    h->pos = pos + put;
    set_iosb(a[4], STATUS_SUCCESS, put);
    complete_io(c, a[1], a[2], a[3], a[4]);
    RET(STATUS_SUCCESS);
}
static void write_network_open_info(uint32_t p, uint64_t size, int dir) {
    uint64_t t = now_100ns_since_1601();
    for (int i = 0; i < 4; i++) wr64(p + 8u * (uint32_t)i, t);
    wr64(p + 32, (size + 0xFFF) & ~0xFFFull);
    wr64(p + 40, size);
    wr32(p + 48, dir ? 0x10 : 0x80);
}
K(NtQueryInformationFile) {
    /* (FileHandle, IoStatusBlock, FileInformation, Length, FileInformationClass) */
    Handle *h = handle_get(a[0]);
    if (!h || h->kind != H_FILE) RET(STATUS_INVALID_HANDLE);
    uint32_t p = a[2];
    uint64_t sz = file_size(h);
    switch (a[4]) {
    case 14: wr64(p, h->pos); set_iosb(a[1], 0, 8); RET(STATUS_SUCCESS);           /* FilePositionInformation */
    case 5:                                                                           /* FileStandardInformation */
        wr64(p, (sz + 0xFFF) & ~0xFFFull); wr64(p + 8, sz); wr32(p + 16, 1); wr8(p + 20, 0); wr8(p + 21, (uint8_t)h->is_dir);
        set_iosb(a[1], 0, 24); RET(STATUS_SUCCESS);
    case 34: write_network_open_info(p, sz, h->is_dir); set_iosb(a[1], 0, 56); RET(STATUS_SUCCESS);
    case 4: {                                                                         /* FileBasicInformation */
        uint64_t t = now_100ns_since_1601();
        for (int i = 0; i < 4; i++) wr64(p + 8u * (uint32_t)i, t);
        wr32(p + 32, h->is_dir ? 0x10 : 0x80);
        set_iosb(a[1], 0, 40); RET(STATUS_SUCCESS);
    }
    default:
        runtime_log("[file] NtQueryInformationFile class %u not implemented", a[4]);
        RET(STATUS_INVALID_PARAMETER);
    }
}
K(NtSetInformationFile) {
    Handle *h = handle_get(a[0]);
    if (!h || h->kind != H_FILE) RET(STATUS_INVALID_HANDLE);
    switch (a[4]) {
    case 14: h->pos = rd64(a[2]); break;                                    /* position */
    case 20: case 19: if (h->fd >= 0 && ftruncate(h->fd, (off_t)rd64(a[2]))) {} break; /* EOF / allocation */
    case 13: {                                                               /* FileDispositionInformation */
        int del = rd8(a[2]) != 0;
        if (del && h->is_dir && h->host[0]) {                                /* a directory must be empty */
            DIR *dp = opendir(h->host);
            int empty = 1;
            if (dp) {
                struct dirent *e;
                while ((e = readdir(dp))) if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) { empty = 0; break; }
                closedir(dp);
            }
            if (!empty) { set_iosb(a[1], 0xC0000101u /* STATUS_DIRECTORY_NOT_EMPTY */, 0); RET(0xC0000101u); }
        }
        if (h->host[0]) h->delete_pending = del;                             /* the disc is read-only */
        break;
    }
    case 4: break;                                                           /* basic info: ignore */
    default: runtime_log("[file] NtSetInformationFile class %u ignored", a[4]); break;
    }
    set_iosb(a[1], STATUS_SUCCESS, 0);
    RET(STATUS_SUCCESS);
}
K(NtQueryFullAttributesFile) {
    char xp[1024];
    object_attributes_path(a[0], xp, sizeof xp);
    VfsPath vp;
    if (vfs_resolve(xp, &vp)) RET(STATUS_OBJECT_PATH_NOT_FOUND);
    if (vp.kind == VFS_DISC && vfs_have_iso()) {
        if (vp.iso_index < 0 && !vp.is_dir) RET(STATUS_OBJECT_NAME_NOT_FOUND);
        write_network_open_info(a[1], vp.size, vp.is_dir);
        RET(STATUS_SUCCESS);
    }
    struct stat st;
    if (stat(vp.host, &st)) RET(STATUS_OBJECT_NAME_NOT_FOUND);
    write_network_open_info(a[1], (uint64_t)st.st_size, S_ISDIR(st.st_mode));
    RET(STATUS_SUCCESS);
}
K(NtQueryVolumeInformationFile) {
    uint32_t p = a[2];
    if (a[4] == 3) {   /* FileFsSizeInformation: report a roomy 8 GB drive */
        wr64(p, 0x200000); wr64(p + 8, 0x100000); wr32(p + 16, 8); wr32(p + 20, 512);
        set_iosb(a[1], 0, 24);
        RET(STATUS_SUCCESS);
    }
    runtime_log("[file] NtQueryVolumeInformationFile class %u not implemented", a[4]);
    RET(STATUS_INVALID_PARAMETER);
}
K(NtQueryDirectoryFile) {
    /* (FileHandle, Event, Apc, ApcCtx, IoStatusBlock, FileInformation, Length, Class, FileMask, RestartScan) */
    Handle *h = handle_get(a[0]);
    if (!h || h->kind != H_FILE || !h->is_dir) RET(STATUS_INVALID_HANDLE);
    if (a[9]) h->dir_cursor = 0;
    char mask[256] = "*";
    if (a[8]) read_ansi_string(a[8], mask, sizeof mask);
    char name[256];
    uint32_t size = 0;
    int is_dir = 0;
    for (;;) {
        int ok = 0;
        if (h->iso_index >= 0 || (!h->host[0] && vfs_have_iso())) {
            ok = vfs_iso_list(h->rel, h->dir_cursor, name, sizeof name, &size, &is_dir);
        } else {
            DIR *d = opendir(h->host);
            if (d) {
                struct dirent *de;
                int i = 0;
                while ((de = readdir(d))) {
                    if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
                    if (i++ == h->dir_cursor) {
                        snprintf(name, sizeof name, "%s", de->d_name);
                        char full[1300];
                        snprintf(full, sizeof full, "%s/%s", h->host, de->d_name);
                        struct stat st;
                        if (!stat(full, &st)) { size = (uint32_t)st.st_size; is_dir = S_ISDIR(st.st_mode); }
                        ok = 1;
                        break;
                    }
                }
                closedir(d);
            }
        }
        if (!ok) { set_iosb(a[4], STATUS_NO_MORE_FILES, 0); RET(STATUS_NO_MORE_FILES); }
        h->dir_cursor++;
        /* simple mask support: "*" or exact name or "*.ext" */
        if (!strcmp(mask, "*") || !strcmp(mask, "*.*") || !strcasecmp(mask, name)) break;
        if (mask[0] == '*' && strlen(name) >= strlen(mask) - 1 &&
            !strcasecmp(name + strlen(name) - (strlen(mask) - 1), mask + 1)) break;
    }
    /* FILE_DIRECTORY_INFORMATION (Xbox: ANSI names) */
    uint32_t p = a[5], nl = (uint32_t)strlen(name);
    if (a[6] < 0x40 + nl) RET(0x80000005u /* STATUS_BUFFER_OVERFLOW */);
    wr32(p, 0); wr32(p + 4, 0);
    uint64_t t = now_100ns_since_1601();
    for (int i = 0; i < 4; i++) wr64(p + 8 + 8u * (uint32_t)i, t);
    wr64(p + 40, size); wr64(p + 48, (size + 0xFFF) & ~0xFFFull);
    wr32(p + 56, is_dir ? 0x10 : 0x80); wr32(p + 60, nl);
    memcpy(g_mem + p + 64, name, nl);
    set_iosb(a[4], STATUS_SUCCESS, 64 + nl);
    RET(STATUS_SUCCESS);
}
K(NtFlushBuffersFile) { Handle *h = handle_get(a[0]); if (h && h->fd >= 0) fsync(h->fd); set_iosb(a[1], 0, 0); RET(STATUS_SUCCESS); }
K(NtFsControlFile) {
    /* FSCTL_DISMOUNT_VOLUME and friends: volumes are host folders, nothing to do */
    if (g_cfg.trace_kernel) runtime_log("[file] NtFsControlFile fsctl %08x", a[5]);
    set_iosb(a[4], STATUS_SUCCESS, 0);
    RET(STATUS_SUCCESS);
}
K(NtDeviceIoControlFile) {
    /* (FileHandle, Event, Apc, ApcCtx, IoStatusBlock, IoControlCode, In, InLen, Out, OutLen) */
    uint32_t out = a[8], outlen = a[9];
    const uint64_t part_bytes = 0x80000000ull;   /* 2 GB virtual partitions */
    switch (a[5]) {
    case 0x00070000: /* IOCTL_DISK_GET_DRIVE_GEOMETRY */
        if (outlen < 24) break;
        wr64(out, part_bytes / (255ull * 63 * 512)); wr32(out + 8, 12 /* FixedMedia */);
        wr32(out + 12, 255); wr32(out + 16, 63); wr32(out + 20, 512);
        set_iosb(a[4], STATUS_SUCCESS, 24);
        RET(STATUS_SUCCESS);
    case 0x00074004: /* IOCTL_DISK_GET_PARTITION_INFO */
        if (outlen < 32) break;
        wr64(out, 0x80000); wr64(out + 8, part_bytes); wr32(out + 16, 0); wr32(out + 20, 1);
        wr8(out + 24, 0x0C); wr8(out + 25, 0); wr8(out + 26, 1); wr8(out + 27, 0);
        set_iosb(a[4], STATUS_SUCCESS, 32);
        RET(STATUS_SUCCESS);
    case 0x00070024: /* IOCTL_DISK_VERIFY */
        set_iosb(a[4], STATUS_SUCCESS, 0);
        RET(STATUS_SUCCESS);
    default: break;
    }
    runtime_log("[file] NtDeviceIoControlFile ioctl %08x (not implemented)", a[5]);
    set_iosb(a[4], STATUS_INVALID_DEVICE_REQUEST, 0);
    RET(STATUS_INVALID_DEVICE_REQUEST);
}
K(NtClose) {
    Handle *h = handle_get(a[0]);
    if (!h) RET(STATUS_INVALID_HANDLE);
    if (h->kind == H_FILE && h->fd >= 0) close(h->fd);
    if (h->kind == H_FILE && h->delete_pending && h->host[0]) {
        /* the name goes away when the handle that asked for it closes (others may still be open on the host: fine) */
        int r = h->is_dir ? rmdir(h->host) : unlink(h->host);
        if (r && g_cfg.trace_kernel) runtime_log("[file] delete '%s' failed: %s", h->host, strerror(errno));
    }
    h->kind = H_FREE;
    RET(STATUS_SUCCESS);
}
K(IoCreateSymbolicLink) {
    char name[256], target[256];
    read_ansi_string(a[0], name, sizeof name);
    read_ansi_string(a[1], target, sizeof target);
    vfs_add_link(name, target);
    RET(STATUS_SUCCESS);
}
K(NtOpenSymbolicLinkObject) {
    char name[512];
    object_attributes_path(a[1], name, sizeof name);
    const char *t = vfs_query_link(name);
    if (!t) RET(STATUS_OBJECT_NAME_NOT_FOUND);
    uint32_t h = handle_new(H_LINK);
    snprintf(handle_get(h)->link_target, 256, "%s", t);
    wr32(a[0], h);
    RET(STATUS_SUCCESS);
}
K(NtQuerySymbolicLinkObject) {
    Handle *h = handle_get(a[0]);
    if (!h || h->kind != H_LINK) RET(STATUS_INVALID_HANDLE);
    uint32_t len = (uint32_t)strlen(h->link_target), max = rd16(a[1] + 2), buf = rd32(a[1] + 4);
    if (len > max) RET(0xC0000023u /* STATUS_BUFFER_TOO_SMALL */);
    memcpy(g_mem + buf, h->link_target, len);
    wr16(a[1], (uint16_t)len);
    if (a[2]) wr32(a[2], len);
    RET(STATUS_SUCCESS);
}

/* ---- I/O manager (only used by low-level drivers; not expected to run) ---- */
K(IoCreateDevice) {
    /* (DriverObject, DeviceExtensionSize, DeviceName, DeviceType, Exclusive, DeviceObject*)
     * Best-effort DEVICE_OBJECT: Type, Size, DriverObject, DeviceExtension (+0x18),
     * DeviceType (+0x1C), StackSize (+0x1E). Used by the XDK's USB/input drivers. */
    uint32_t size = 0x60 + a[1];
    uint32_t dev = guest_alloc(size, 16);
    wr16(dev, 3); wr16(dev + 2, (uint16_t)size); wr32(dev + 4, 1); wr32(dev + 8, a[0]);
    wr32(dev + 0xC, dev); wr32(dev + 0x18, dev + 0x60); wr8(dev + 0x1C, (uint8_t)a[3]); wr8(dev + 0x1E, 1);
    wr32(dev + 0x24, 512);
    if (a[5]) wr32(a[5], dev);
    runtime_log("[kernel] IoCreateDevice type %u ext %u -> %08x", a[3], a[1], dev);
    RET(STATUS_SUCCESS);
}

/* ------------------------------------------------------------- table */
#define E(ord, n, name) {ord, n, 0, k_##name}
#define F(ord, n, name) {ord, n, 1, k_##name}
static const KExport g_exports[] = {
    E(1, 0, AvGetSavedDataAddress), E(2, 4, AvSendTVEncoderOption), E(3, 6, AvSetDisplayMode),
    E(4, 1, AvSetSavedDataAddress), E(8, 0, DbgPrint),
    E(14, 1, ExAllocatePool), E(15, 2, ExAllocatePoolWithTag), E(17, 1, ExFreePool),
    E(23, 1, ExQueryPoolBlockSize), E(24, 5, ExQueryNonVolatileSetting),
    E(44, 2, HalGetInterruptVector), E(47, 2, HalRegisterShutdownNotification),
    E(49, 1, HalReturnToFirmware), E(65, 6, IoCreateDevice), E(67, 2, IoCreateSymbolicLink),
    E(95, 1, KeBugCheck), E(97, 1, KeCancelTimer), E(98, 1, KeConnectInterrupt),
    E(99, 3, KeDelayExecutionThread), E(100, 1, KeDisconnectInterrupt), E(107, 3, KeInitializeDpc),
    E(109, 7, KeInitializeInterrupt), E(113, 2, KeInitializeTimerEx), E(119, 3, KeInsertQueueDpc),
    E(125, 0, KeQueryInterruptTime), E(126, 0, KeQueryPerformanceCounter),
    E(127, 0, KeQueryPerformanceFrequency), E(128, 1, KeQuerySystemTime),
    E(129, 0, KeRaiseIrqlToDpcLevel), E(137, 1, KeRemoveQueueDpc), E(139, 1, KeRestoreFloatingPointState),
    E(142, 1, KeSaveFloatingPointState), E(143, 2, KeSetBasePriorityThread), E(145, 3, KeSetEvent),
    E(149, 4, KeSetTimer), E(150, 5, KeSetTimerEx), E(151, 1, KeStallExecutionProcessor),
    E(153, 3, KeSynchronizeExecution), E(158, 8, KeWaitForMultipleObjects),
    E(159, 5, KeWaitForSingleObject), F(160, 1, KfRaiseIrql), F(161, 1, KfLowerIrql),
    E(165, 1, MmAllocateContiguousMemory), E(166, 5, MmAllocateContiguousMemoryEx),
    E(167, 2, MmAllocateSystemMemory), E(168, 2, MmClaimGpuInstanceMemory),
    E(171, 1, MmFreeContiguousMemory), E(172, 2, MmFreeSystemMemory), E(173, 1, MmGetPhysicalAddress),
    E(175, 3, MmLockUnlockBufferPages), E(176, 2, MmLockUnlockPhysicalPage),
    E(178, 3, MmPersistContiguousMemory), E(179, 1, MmQueryAddressProtect),
    E(180, 1, MmQueryAllocationSize), E(181, 1, MmQueryStatistics), E(182, 3, MmSetAddressProtect),
    E(184, 5, NtAllocateVirtualMemory), E(186, 1, NtClearEvent), E(187, 1, NtClose),
    E(189, 4, NtCreateEvent), E(190, 9, NtCreateFile), E(192, 3, NtCreateMutant),
    E(196, 10, NtDeviceIoControlFile), E(198, 2, NtFlushBuffersFile), E(199, 3, NtFreeVirtualMemory),
    E(200, 10, NtFsControlFile), E(202, 6, NtOpenFile), E(203, 2, NtOpenSymbolicLinkObject),
    E(207, 10, NtQueryDirectoryFile), E(210, 2, NtQueryFullAttributesFile),
    E(211, 5, NtQueryInformationFile), E(215, 3, NtQuerySymbolicLinkObject),
    E(217, 2, NtQueryVirtualMemory), E(218, 5, NtQueryVolumeInformationFile), E(219, 8, NtReadFile),
    E(221, 2, NtReleaseMutant), E(224, 2, NtResumeThread), E(225, 2, NtSetEvent),
    E(226, 5, NtSetInformationFile), E(232, 3, NtUserIoApcDispatcher), E(233, 3, NtWaitForSingleObject),
    E(234, 4, NtWaitForSingleObjectEx), E(235, 6, NtWaitForMultipleObjectsEx), E(236, 8, NtWriteFile),
    E(238, 0, NtYieldExecution), E(246, 3, ObReferenceObjectByHandle), F(250, 1, ObfDereferenceObject),
    E(252, 1, PhyGetLinkState), E(253, 2, PhyInitialize), E(255, 10, PsCreateSystemThreadEx),
    E(258, 1, PsTerminateSystemThread), E(269, 3, RtlCompareMemoryUlong),
    E(277, 1, RtlEnterCriticalSection), E(279, 3, RtlEqualString), E(289, 2, RtlInitAnsiString),
    E(291, 1, RtlInitializeCriticalSection), E(294, 1, RtlLeaveCriticalSection),
    E(301, 1, RtlNtStatusToDosError), E(302, 1, RtlRaiseException), E(304, 2, RtlTimeFieldsToTime),
    E(305, 2, RtlTimeToTimeFields), E(308, 3, RtlUnicodeStringToAnsiString), E(312, 4, RtlUnwind),
    E(327, 1, XeLoadSection), E(328, 1, XeUnloadSection),
    E(335, 1, XcSHAInit), E(336, 3, XcSHAUpdate), E(337, 2, XcSHAFinal),
    E(338, 3, XcRC4Key), E(339, 3, XcRC4Crypt), E(340, 7, XcHMAC),
    E(345, 5, XcModExp), E(346, 2, XcDESKeyParity), E(347, 3, XcKeyTable), E(349, 7, XcBlockCryptCBC),
    E(358, 0, HalIsResetOrShutdownPending), E(360, 0, HalInitiateShutdown),
};
static const KExport *g_by_ordinal[400];
static uint32_t g_call_counts[400];

void kernel_call(CPU *c, uint32_t ordinal) {
    const KExport *e = ordinal < 400 ? g_by_ordinal[ordinal] : NULL;
    const char *name = ordinal < 367 ? kKernelExportNames[ordinal] : "?";
    if (!e) {
        runtime_log("[kernel] unimplemented kernel function %u %s (called from %08x)", ordinal, name, rd32(c->esp));
        recomp_trap(c, rd32(c->esp), "unimplemented kernel export");
    }
    uint32_t a[12] = {0};
    int stack_args = e->nargs;
    if (e->fastcall) {
        a[0] = c->ecx; a[1] = c->edx;
        stack_args = e->nargs > 2 ? e->nargs - 2 : 0;
        for (int i = 0; i < stack_args; i++) a[2 + i] = rd32(c->esp + 4 + 4u * (uint32_t)i);
    } else {
        for (int i = 0; i < e->nargs && i < 12; i++) a[i] = rd32(c->esp + 4 + 4u * (uint32_t)i);
    }
    g_call_counts[ordinal]++;
    if (g_cfg.trace_kernel)
        runtime_log("[kernel] %s(%08x, %08x, %08x, %08x) from %08x", name, a[0], a[1], a[2], a[3], rd32(c->esp));
    e->fn(c, a);
    /* stdcall: callee pops return address + arguments. DbgPrint is cdecl (nargs 0). */
    c->esp += 4 + 4u * (uint32_t)stack_args;
}

void kernel_init(void) {
    g_boot_ns = host_time_ns();
    for (size_t i = 0; i < sizeof g_exports / sizeof g_exports[0]; i++) g_by_ordinal[g_exports[i].ordinal] = &g_exports[i];

    /* data exports */
    g_tick_count_addr = make_data(156, 4);                         /* KeTickCount */
    uint32_t hw = make_data(322, 16);                              /* XboxHardwareInfo */
    wr32(hw, 0x00000020);                                          /* flags */
    wr8(hw + 4, 0xA2);                                             /* GPU revision (NV2A A2) */
    wr8(hw + 5, 0xD3);                                             /* MCPX revision */
    make_data(164, 4);                                             /* LaunchDataPage (NULL) */
    make_data(323, 16);                                            /* XboxHDKey */
    make_data(325, 16);                                            /* XboxSignatureKey */
    make_data(353, 16);                                            /* XboxLANKey */
    wr32(make_data(40, 4), 3);                                     /* HalDiskCachePartitionCount */
    wr32(make_data(356, 4), 0);                                    /* HalBootSMCVideoMode */
    wr32(make_data(41, 4), make_ansi_string("HALORECOMP VIRTUAL HDD"));  /* HalDiskModelNumber */
    wr32(make_data(42, 4), make_ansi_string("0000000000000001"));        /* HalDiskSerialNumber */
    g_obj_type_event = make_data(16, 0x40);                        /* ExEventObjectType */
    g_obj_type_thread = make_data(259, 0x40);                      /* PsThreadObjectType */
    make_data(324, 8);                                             /* XboxKrnlVersion */
    wr16(g_export_addr[324], 1); wr16(g_export_addr[324] + 2, 0);
    wr16(g_export_addr[324] + 4, 5838); wr16(g_export_addr[324] + 6, 1);

    vfs_init();

    /* context used for DPC / timer callbacks */
    g_dpc_cpu = cpu_new(0x20000);
    pthread_t th;
    pthread_create(&th, NULL, ticker_main, NULL);
    pthread_detach(th);
}
