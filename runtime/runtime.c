/* runtime.c — guest memory, XBE loader, dispatch, threads, traps. */
#define _GNU_SOURCE
#include "runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0   /* macOS commits anonymous memory lazily anyway */
#endif

uint8_t *g_mem;
RuntimeConfig g_cfg;
HostCallbacks g_host;

/* ------------------------------------------------------------------ log */
static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;
void runtime_log(const char *fmt, ...) {
    va_list ap;
    pthread_mutex_lock(&g_log_lock);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
    pthread_mutex_unlock(&g_log_lock);
}

/* --------------------------------------------------------------- memory */
/* Physical RAM is a shared memory object (GUEST_RAM_SIZE) mapped:
 *   - identity at virtual 0 (user heap: NtAllocateVirtualMemory, pools, stacks)
 *   - at 0x80000000 and 0xF0000000 (the kernel's physical mirrors)
 *   - the XBE image's own physical pages (GUEST_IMAGE_PHYS) overlay virtual
 *     0x10000..image end, exactly like the real kernel's mapping, so physical
 *     low memory stays free for contiguous allocations (framebuffers etc.).
 * One page bitmap tracks physical pages for both kinds of allocation. */
#define PAGE 4096u
#define RAM_PAGES (GUEST_RAM_SIZE / PAGE)
static uint32_t *g_page_run;      /* at the first page of an allocation: its page count; else 0 */
static uint8_t *g_page_used;
static uint32_t g_heap_used;
static uint32_t g_kdata_next = GUEST_KDATA_BASE;
static pthread_mutex_t g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_ram_fd = -1;
static uint32_t g_image_base, g_image_end, g_image_phys;

static void mark_used(uint32_t phys, uint32_t size) {
    uint32_t first = phys / PAGE, n = (size + PAGE - 1) / PAGE;
    if (first + n > RAM_PAGES) n = RAM_PAGES - first;
    memset(g_page_used + first, 1, n);
}

static int mem_init(void) {
    /* Reserve the whole 32-bit guest address space; pages are committed lazily
     * by the host OS on first touch, so this costs nothing up front. */
    void *p = mmap(NULL, 1ull << 32, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        runtime_log("[mem] failed to reserve 4 GiB guest space: %s", strerror(errno));
        return -1;
    }
    g_mem = p;

    char name[32];
    snprintf(name, sizeof name, "/halorecomp.%d", (int)getpid());
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { runtime_log("[mem] shm_open failed: %s", strerror(errno)); return -1; }
    shm_unlink(name);
    if (ftruncate(fd, GUEST_RAM_SIZE)) { runtime_log("[mem] ftruncate failed: %s", strerror(errno)); return -1; }
    const uint32_t views[3] = {0, GUEST_RAM_MIRROR1, GUEST_RAM_MIRROR2};
    for (int i = 0; i < 3; i++) {
        void *v = mmap(g_mem + views[i], GUEST_RAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
        if (v == MAP_FAILED) { runtime_log("[mem] mapping RAM view %08x failed: %s", views[i], strerror(errno)); return -1; }
    }
    g_ram_fd = fd;

    g_page_run = calloc(RAM_PAGES, sizeof(uint32_t));
    g_page_used = calloc(RAM_PAGES, 1);
    /* The kernel's own pages. Our kernel lives on the host, so only the first
     * 128 KB stay reserved; the rest of the range is the only memory left
     * below 16 MB once Halo claims its fixed pools, and D3D needs a few pages
     * there for visibility-test reports (MmAllocateContiguousMemoryEx with a
     * 16 MB limit, 4 KB each). */
    mark_used(0, 0x20000);
    mark_used(GUEST_GPU_RESERVED, GUEST_PHYS_LIMIT - GUEST_GPU_RESERVED);  /* GPU instance memory */
    /* RAM above the Xbox's 64 MB (up to GUEST_RAM_SIZE) backs ordinary virtual
     * allocations only (guest_alloc); contiguous / physical allocations stay
     * below 64 MB, where the GPU and audio hardware expect them. The extra
     * room absorbs what this runtime uses beyond a real Xbox kernel. */
    return 0;
}

/* Called by the XBE loader: give the image its own physical pages and map
 * them at the image's virtual address. */
static int mem_map_image(uint32_t base, uint32_t size) {
    size = (size + PAGE - 1) & ~(PAGE - 1);
    g_image_phys = (GUEST_GPU_RESERVED - size) & ~0xFFFFu;       /* just below the GPU area */
    void *v = mmap(g_mem + base, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, g_ram_fd, g_image_phys);
    if (v == MAP_FAILED) { runtime_log("[mem] mapping image failed: %s", strerror(errno)); return -1; }
    mark_used(g_image_phys, size);
    runtime_log("[mem] XBE image: virtual %08x-%08x on physical %08x", base, base + size, g_image_phys);
    g_image_base = base;
    g_image_end = base + size;
    return 0;
}

/* Same, searching downward from the top (how the kernel places contiguous memory). */
static uint32_t alloc_pages_top(uint32_t pages, uint32_t step, uint32_t lo, uint32_t hi) {
    if (hi < lo + pages) return UINT32_MAX;
    uint32_t i = hi - pages;
    i -= i % step;
    for (;;) {
        uint32_t j = 0;
        while (j < pages && !g_page_used[i + j]) j++;
        if (j == pages) {
            memset(g_page_used + i, 1, pages);
            g_page_run[i] = pages;
            g_heap_used += pages * PAGE;
            return i;
        }
        if (i < lo + step) return UINT32_MAX;
        i -= step;
    }
}

/* Virtual (identity-mapped) allocation above GUEST_HEAP_BASE. */
uint32_t guest_alloc(uint32_t size, uint32_t align) {
    if (!size) size = 1;
    uint32_t pages = (size + PAGE - 1) / PAGE, step = align > PAGE ? align / PAGE : 1;
    pthread_mutex_lock(&g_alloc_lock);
    uint32_t i = alloc_pages_top(pages, step, GUEST_PHYS_LIMIT / PAGE, GUEST_RAM_SIZE / PAGE);   /* extra RAM first */
    if (i == UINT32_MAX) i = alloc_pages_top(pages, step, GUEST_HEAP_BASE / PAGE, GUEST_PHYS_LIMIT / PAGE);
    pthread_mutex_unlock(&g_alloc_lock);
    if (i == UINT32_MAX) {
        runtime_log("[mem] out of guest memory allocating %u bytes (%u KB in use)", size, g_heap_used / 1024);
        return 0;
    }
    uint32_t a = i * PAGE;
    memset(g_mem + a, 0, (size_t)pages * PAGE);
    return a;
}

/* Physically contiguous allocation within [lowest, highest]; returns the
 * kernel-mirror address (0x80000000 | phys), like MmAllocateContiguousMemoryEx. */
uint32_t guest_alloc_contiguous(uint32_t size, uint32_t lowest, uint32_t highest, uint32_t align) {
    if (!size) size = 1;
    uint32_t pages = (size + PAGE - 1) / PAGE, step = align > PAGE ? align / PAGE : 1;
    uint32_t lo = (lowest + PAGE - 1) / PAGE;
    uint32_t hi = highest >= GUEST_PHYS_LIMIT - 1 ? GUEST_PHYS_LIMIT / PAGE : (highest + 1) / PAGE;
    if (lo % step) lo += step - lo % step;
    pthread_mutex_lock(&g_alloc_lock);
    uint32_t i = alloc_pages_top(pages, step, lo, hi);
    pthread_mutex_unlock(&g_alloc_lock);
    if (i == UINT32_MAX) {
        static unsigned failures;   /* Halo keeps retrying small requests below 16 MB: log a few */
        if (++failures <= 10 || (failures & (failures - 1)) == 0)
            runtime_log("[mem] no contiguous memory for %u bytes in %08x-%08x (failure %u)", size, lowest, highest, failures);
        return 0;
    }
    memset(g_mem + GUEST_RAM_MIRROR1 + i * PAGE, 0, (size_t)pages * PAGE);
    if (getenv("HALO_TRACE_MEM"))
        runtime_log("[mem] contiguous %08x-%08x (%u bytes, range %08x-%08x)", i * PAGE, i * PAGE + pages * PAGE, size, lowest, highest);
    return GUEST_RAM_MIRROR1 | (i * PAGE);
}

/* Physical address of a guest virtual address. */
uint32_t guest_phys(uint32_t va) {
    if (va >= g_image_base && va < g_image_end) return g_image_phys + (va - g_image_base);
    if ((va >= GUEST_RAM_MIRROR1 && va < GUEST_RAM_MIRROR1 + GUEST_RAM_SIZE) ||
        (va >= GUEST_RAM_MIRROR2 && va < GUEST_RAM_MIRROR2 + GUEST_RAM_SIZE))
        return va & (GUEST_RAM_SIZE - 1);
    return va < GUEST_RAM_SIZE ? va : va & (GUEST_RAM_SIZE - 1);
}

static int page_index(uint32_t addr, uint32_t *idx) {
    uint32_t ph = guest_phys(addr);
    if (ph >= GUEST_RAM_SIZE) return 0;
    *idx = ph / PAGE;
    return 1;
}

/* Mark a fixed range as allocated (NtAllocateVirtualMemory with a base address). */
void guest_reserve(uint32_t addr, uint32_t size) {
    uint32_t i;
    if (!page_index(addr, &i)) return;
    pthread_mutex_lock(&g_alloc_lock);
    mark_used(i * PAGE, size);
    pthread_mutex_unlock(&g_alloc_lock);
}

uint32_t guest_alloc_size(uint32_t addr) {
    uint32_t i;
    return page_index(addr, &i) ? g_page_run[i] * PAGE : 0;
}

uint32_t guest_free(uint32_t addr) {
    uint32_t i;
    if (!page_index(addr, &i)) return 0;
    pthread_mutex_lock(&g_alloc_lock);
    uint32_t n = g_page_run[i];
    if (n && getenv("HALO_TRACE_MEM")) runtime_log("[mem] free %08x-%08x", i * PAGE, (i + n) * PAGE);
    if (n) {
        memset(g_page_used + i, 0, n);
        g_page_run[i] = 0;
        g_heap_used -= n * PAGE;
    }
    pthread_mutex_unlock(&g_alloc_lock);
    return n * PAGE;
}

uint32_t guest_heap_used(void) { return g_heap_used; }

uint32_t kdata_alloc(uint32_t size) {
    pthread_mutex_lock(&g_alloc_lock);
    uint32_t a = (g_kdata_next + 15) & ~15u;
    g_kdata_next = a + size;
    pthread_mutex_unlock(&g_alloc_lock);
    if (g_kdata_next > GUEST_KDATA_END) {
        runtime_log("[mem] kernel data area exhausted");
        abort();
    }
    return a;
}

/* 80-bit extended <-> double (x87 tbyte loads/stores) */
double rdf80(uint32_t a) {
    uint64_t mant = rd64(a);
    uint16_t se = rd16(a + 8);
    int sign = se >> 15, exp = se & 0x7FFF;
    if (exp == 0 && mant == 0) return sign ? -0.0 : 0.0;
    if (exp == 0x7FFF) return (mant << 1) ? NAN : (sign ? -INFINITY : INFINITY);
    double v = ldexp((double)mant, exp - 16383 - 63);
    return sign ? -v : v;
}
void wrf80(uint32_t a, double v) {
    uint16_t se = signbit(v) ? 0x8000 : 0;
    uint64_t mant = 0;
    if (isnan(v)) { se |= 0x7FFF; mant = 0xC000000000000000ull; }
    else if (isinf(v)) { se |= 0x7FFF; mant = 0x8000000000000000ull; }
    else if (v != 0) {
        int e;
        double f = frexp(fabs(v), &e);     /* v = f * 2^e, f in [0.5,1) */
        mant = (uint64_t)ldexp(f, 64);
        se |= (uint16_t)(e - 1 + 16383);
    }
    wr64(a, mant);
    wr16(a + 8, se);
}

/* --------------------------------------------------------------- loader */
#include "../tools/xbe_layout.h"

static int load_xbe(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { runtime_log("[xbe] cannot open %s", path); return -1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = malloc((size_t)len);
    if (fread(d, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(d); return -1; }
    fclose(f);
    if (memcmp(d, "XBEH", 4) != 0) { runtime_log("[xbe] bad magic"); free(d); return -1; }
    uint32_t base, hdr_size, nsec, sec_addr;
    memcpy(&base, d + XBE_BASE, 4);
    memcpy(&hdr_size, d + XBE_HEADERS_SIZE, 4);
    memcpy(&nsec, d + XBE_NUM_SECTIONS, 4);
    memcpy(&sec_addr, d + XBE_SECTION_HEADERS, 4);
    if (base != GUEST_IMAGE_BASE) runtime_log("[xbe] warning: unusual base %#x", base);
    {
        uint32_t isz;
        memcpy(&isz, d + XBE_IMAGE_SIZE, 4);
        if (mem_map_image(base, isz)) { free(d); return -1; }
    }
    memcpy(g_mem + base, d, hdr_size);
    for (uint32_t i = 0; i < nsec; i++) {
        const uint8_t *s = d + (sec_addr - base) + i * 56;
        uint32_t flags, va, vs, ra, rs;
        memcpy(&flags, s, 4); memcpy(&va, s + 4, 4); memcpy(&vs, s + 8, 4);
        memcpy(&ra, s + 12, 4); memcpy(&rs, s + 16, 4);
        /* Load every section, including non-preload ones: XeLoadSection then
         * becomes a no-op, which is simpler and only costs a little memory. */
        memcpy(g_mem + va, d + ra, rs);
        if (vs > rs) memset(g_mem + va + rs, 0, vs - rs);
    }
    /* kernel thunk table: retail/debug key chosen by entry point validity */
    uint32_t enc_entry, enc_thunk, image_size;
    memcpy(&enc_entry, d + XBE_ENTRY, 4);
    memcpy(&enc_thunk, d + XBE_KERNEL_THUNK, 4);
    memcpy(&image_size, d + XBE_IMAGE_SIZE, 4);
    int retail = ((enc_entry ^ XBE_ENTRY_KEY_RETAIL) - base) < image_size;
    uint32_t thunk = enc_thunk ^ (retail ? XBE_THUNK_KEY_RETAIL : XBE_THUNK_KEY_DEBUG);
    int n = 0;
    for (uint32_t a = thunk; rd32(a); a += 4, n++) {
        uint32_t ordinal = rd32(a) & 0x7FFFFFFF;
        wr32(a, kernel_export_address(ordinal));
    }
    runtime_log("[xbe] loaded %s: %u sections, %d kernel imports patched", path, nsec, n);
    free(d);
    return 0;
}

/* ------------------------------------------------------------- dispatch */
GuestFn recomp_lookup(uint32_t addr) {
    uint32_t lo = 0, hi = g_recomp_func_count;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (g_recomp_funcs[mid].addr < addr) lo = mid + 1;
        else hi = mid;
    }
    if (lo < g_recomp_func_count && g_recomp_funcs[lo].addr == addr) return g_recomp_funcs[lo].fn;
    return NULL;
}

/* Diagnostics: map a host code address back to the recompiled guest function
 * containing it (functions are laid out in address order within each file). */
static int cmp_fnptr(const void *a, const void *b) {
    uintptr_t x = (uintptr_t)((const RecompFunc *)a)->fn, y = (uintptr_t)((const RecompFunc *)b)->fn;
    return x < y ? -1 : x > y;
}
static uint32_t guest_func_for_host(void *ra) {
    static RecompFunc *sorted;
    if (!sorted) {
        sorted = malloc(sizeof(RecompFunc) * g_recomp_func_count);
        memcpy(sorted, g_recomp_funcs, sizeof(RecompFunc) * g_recomp_func_count);
        qsort(sorted, g_recomp_func_count, sizeof(RecompFunc), cmp_fnptr);
    }
    uint32_t best = 0;
    for (uint32_t i = 0; i < g_recomp_func_count; i++) {
        if ((uintptr_t)sorted[i].fn <= (uintptr_t)ra) best = sorted[i].addr;
        else break;
    }
    return best;
}

/* HALO_TRACE_SCRIPT=file ("hexaddr name" lines, e.g. Halo's script function
 * table): log indirect calls to those addresses (first 30 per name) with the
 * first stack argument and the result. */
typedef struct { uint32_t addr; char name[48]; uint32_t n; } TraceFn;
static TraceFn *g_tfn;
static int g_tfn_n = -1;
static int tfn_cmp(const void *a, const void *b) {
    uint32_t x = ((const TraceFn *)a)->addr, y = ((const TraceFn *)b)->addr;
    return x < y ? -1 : x > y;
}
static TraceFn *trace_fn(uint32_t a) {
    if (g_tfn_n < 0) {
        g_tfn_n = 0;
        const char *path = getenv("HALO_TRACE_SCRIPT");
        FILE *f = path ? fopen(path, "r") : NULL;
        if (f) {
            char line[128];
            int cap = 0;
            while (fgets(line, sizeof line, f)) {
                unsigned long ad; char nm[64];
                if (sscanf(line, "%lx %63s", &ad, nm) != 2) continue;
                if (g_tfn_n == cap) { cap = cap ? cap * 2 : 256; g_tfn = realloc(g_tfn, (size_t)cap * sizeof *g_tfn); }
                g_tfn[g_tfn_n].addr = (uint32_t)ad; g_tfn[g_tfn_n].n = 0;
                snprintf(g_tfn[g_tfn_n].name, sizeof g_tfn[0].name, "%s", nm);
                g_tfn_n++;
            }
            fclose(f);
            qsort(g_tfn, (size_t)g_tfn_n, sizeof *g_tfn, tfn_cmp);
            runtime_log("[script] tracing %d functions", g_tfn_n);
        }
    }
    if (!g_tfn_n) return NULL;
    TraceFn key = {a, "", 0};
    return bsearch(&key, g_tfn, (size_t)g_tfn_n, sizeof *g_tfn, tfn_cmp);
}

void call_indirect(CPU *c, uint32_t target) {
    TraceFn *tf = trace_fn(target);
    if (tf && tf->n++ < 30) {
        uint32_t a1 = rd32(c->esp + 4), a2 = rd32(c->esp + 8);
        GuestFn f = recomp_lookup(target);
        if (f) {
            f(c);
            runtime_log("[script] %s(%08x, %08x) -> %08x (call %u)", tf->name, a1, a2, c->eax, tf->n);
            return;
        }
    }
    if (target >= KERNEL_THUNK_BASE && target < KERNEL_THUNK_BASE + 4 * 400) {
        kernel_call(c, (target - KERNEL_THUNK_BASE) / 4);
        return;
    }
    const char *name = NULL;
    HleFn h = hle_lookup(target, &name);
    if (h) {
        if (g_cfg.trace_calls) runtime_log("[hle] %s", name);
        h(c);
        return;
    }
    GuestFn f = recomp_lookup(target);
    if (f) {
        if (g_cfg.trace_calls) runtime_log("[call] %08x", target);
        f(c);
        return;
    }
    runtime_log("[dispatch] call came from recompiled function f_%08x", guest_func_for_host(__builtin_return_address(0)));
    runtime_log("[dispatch] no recompiled function at %08x (called from %08x).\n"
                "  Appended to missing_seeds.txt - add it to seeds.txt and re-run the recompiler.",
                target, rd32(c->esp));
    FILE *ms = fopen("missing_seeds.txt", "a");
    if (ms) { fprintf(ms, "%08x\n", target); fclose(ms); }
    recomp_trap(c, target, "unknown indirect call target");
}

static void dump_cpu(CPU *c) {
    runtime_log("  eax=%08x ecx=%08x edx=%08x ebx=%08x esp=%08x ebp=%08x esi=%08x edi=%08x",
                c->eax, c->ecx, c->edx, c->ebx, c->esp, c->ebp, c->esi, c->edi);
    runtime_log("  stack: %08x %08x %08x %08x %08x %08x", rd32(c->esp), rd32(c->esp + 4),
                rd32(c->esp + 8), rd32(c->esp + 12), rd32(c->esp + 16), rd32(c->esp + 20));
}

void recomp_trap(CPU *c, uint32_t eip, const char *what) {
    runtime_log("[trap] %s at %08x (thread %d)", what, eip, c->thread_index);
    dump_cpu(c);
    exit(3);
}

void recomp_unimpl(CPU *c, uint32_t eip, const char *insn) {
    runtime_log("[trap] instruction not supported by the recompiler: %s", insn);
    dump_cpu(c);
    exit(4);
}

void guest_int3(CPU *c, uint32_t eip) {
    runtime_log("[int3] breakpoint at %08x (ignored)", eip);
    (void)c;
}

void guest_port_io(CPU *c, uint32_t eip, int is_out, uint32_t port, int size) {
    /* 0x80C0: MCPX GPIO/field status, read by D3D's vblank handler before
     * completing a flip (bit 5 clear = ready). Other ports: absent device. */
    uint32_t v = port == 0x80C0 ? 0 : 0xFFFFFFFFu;
    if (!is_out) {
        if (size == 1) set8l(&c->eax, v);
        else if (size == 2) set16(&c->eax, v);
        else c->eax = v;
    }
    if (port != 0x80C0) runtime_log("[io] %s port %#x size %d at %08x", is_out ? "out" : "in", port, size, eip);
}

uint32_t guest_rdtsc_lo(CPU *c, uint32_t *hi) {
    (void)c;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    /* Xbox CPU runs at 733 MHz */
    uint64_t t = (uint64_t)ts.tv_sec * 733333333ull + (uint64_t)ts.tv_nsec * 733ull / 1000ull;
    *hi = (uint32_t)(t >> 32);
    return (uint32_t)t;
}

void guest_cpuid(CPU *c) {
    /* Pentium III (Coppermine) with MMX + SSE, like the Xbox's CPU. */
    uint32_t leaf = c->eax;
    if (leaf == 0) { c->eax = 2; c->ebx = 0x756e6547; c->edx = 0x49656e69; c->ecx = 0x6c65746e; }
    else if (leaf == 1) { c->eax = 0x683; c->ebx = 0; c->ecx = 0; c->edx = 0x0383F9FF; }
    else { c->eax = c->ebx = c->ecx = c->edx = 0; }
}

/* --------------------------------------------------------------- threads */
static pthread_mutex_t g_gil;
static pthread_cond_t g_kcond;
static __thread CPU *t_cpu;
static int g_thread_count;

/* The guest lock is handed off explicitly at preemption points: plain mutexes
 * are not fair, so a guest thread busy-waiting with brief unlocks would
 * otherwise starve the kernel timer thread (no vblank, no timers). */
static volatile int g_gil_waiters;
static volatile uint32_t g_gil_generation;

void guest_lock(void) {
    __atomic_add_fetch(&g_gil_waiters, 1, __ATOMIC_SEQ_CST);
    pthread_mutex_lock(&g_gil);
    __atomic_sub_fetch(&g_gil_waiters, 1, __ATOMIC_SEQ_CST);
    g_gil_generation++;
}

/* HALO_PREEMPT_PROFILE=1: which recompiled loops burn the time slices
 * (link with -rdynamic so dladdr can name them). */
#include <dlfcn.h>
static void preempt_profile(void *ra) {
    static int on = -1;
    static void *addr[64]; static unsigned cnt[64]; static uint64_t next;
    if (on < 0) on = getenv("HALO_PREEMPT_PROFILE") != NULL;
    if (!on) return;
    unsigned h = (unsigned)(((uintptr_t)ra >> 4) % 64);
    for (unsigned i = 0; i < 64; i++) {
        unsigned k = (h + i) % 64;
        if (addr[k] == ra || !addr[k]) { addr[k] = ra; cnt[k]++; break; }
    }
    uint64_t now = host_time_ns();
    if (!next) next = now + 10000000000ull;
    if (now < next) return;
    next = now + 10000000000ull;
    runtime_log("[profile] time slices used up in:");
    for (unsigned i = 0; i < 64; i++) if (cnt[i] > 20) {
        Dl_info di;
        const char *name = dladdr(addr[i], &di) && di.dli_sname ? di.dli_sname : "?";
        runtime_log("[profile]   %6u  %s+%lx", cnt[i], name, di.dli_saddr ? (unsigned long)((char *)addr[i] - (char *)di.dli_saddr) : 0ul);
    }
    memset(cnt, 0, sizeof cnt); memset(addr, 0, sizeof addr);
}

void guest_preempt(CPU *c) {
    preempt_profile(__builtin_return_address(0));
    c->budget = 200000;
    uint32_t gen = g_gil_generation;
    int waiting = __atomic_load_n(&g_gil_waiters, __ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&g_gil);
    /* let a waiting thread actually take the lock before we retake it */
    for (int i = 0; waiting && i < 10000 && __atomic_load_n(&g_gil_generation, __ATOMIC_SEQ_CST) == gen; i++)
        sched_yield();
    if (!waiting) sched_yield();
    guest_lock();
}
void guest_unlock(void) { pthread_mutex_unlock(&g_gil); }
/* Hand the guest lock to a waiting thread and take it back (host code that
 * runs long with the lock held, such as the pushbuffer reader, uses this so
 * the vblank interrupt is not delayed). */
void guest_yield_lock(void) {
    uint32_t gen = g_gil_generation;
    pthread_mutex_unlock(&g_gil);
    for (int i = 0; i < 10000 && __atomic_load_n(&g_gil_generation, __ATOMIC_SEQ_CST) == gen; i++) sched_yield();
    guest_lock();
}
CPU *current_cpu(void) { return t_cpu; }
void set_current_cpu(CPU *c) { t_cpu = c; }

uint64_t host_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Called with the GIL held: sleeps until any kernel object changes or until
 * `deadline_ns` (monotonic; 0 = no deadline). Other guest threads run meanwhile. */
void guest_wait_change(uint64_t deadline_ns) {
    if (!deadline_ns) {
        /* bounded so timers/DPCs serviced by the ticker are re-checked */
        deadline_ns = host_time_ns() + 5000000ull;
    }
    struct timespec now_rt;
    clock_gettime(CLOCK_REALTIME, &now_rt);
    uint64_t now = host_time_ns();
    uint64_t delta = deadline_ns > now ? deadline_ns - now : 0;
    uint64_t abs_ns = (uint64_t)now_rt.tv_sec * 1000000000ull + (uint64_t)now_rt.tv_nsec + delta;
    struct timespec ts = {(time_t)(abs_ns / 1000000000ull), (long)(abs_ns % 1000000000ull)};
    pthread_cond_timedwait(&g_kcond, &g_gil, &ts);
    g_gil_generation++;
}
void guest_signal_change(void) { pthread_cond_broadcast(&g_kcond); }

/* Xbox KPCR layout (subset): fs:[0x00] exception list, fs:[0x04] stack base,
 * fs:[0x08] stack limit, fs:[0x18] self, fs:[0x1C] self-pcr, fs:[0x20] prcb,
 * fs:[0x24] irql (byte), fs:[0x28] current KTHREAD. KTHREAD+0x28 = TLS data. */
void cpu_setup_thread_block(CPU *c) {
    uint32_t kpcr = kdata_alloc(0x300);
    uint32_t kthread = kdata_alloc(0x200);
    memset(g_mem + kpcr, 0, 0x300);
    memset(g_mem + kthread, 0, 0x200);
    wr32(kpcr + 0x00, 0xFFFFFFFF);
    wr32(kpcr + 0x04, c->stack_base);
    wr32(kpcr + 0x08, c->stack_limit);
    wr32(kpcr + 0x18, kpcr);
    wr32(kpcr + 0x1C, kpcr);
    wr32(kpcr + 0x20, kpcr + 0x28);
    wr32(kpcr + 0x28, kthread);
    /* TLS block: XBE TLS template + zero fill */
    uint32_t tls_dir = rd32(GUEST_IMAGE_BASE + XBE_TLS);
    uint32_t tls_size = 0x100;
    if (tls_dir) {
        uint32_t s = rd32(tls_dir), e = rd32(tls_dir + 4), zf = rd32(tls_dir + 16);
        tls_size = (e - s) + zf + 0x10;
        uint32_t tls = kdata_alloc(tls_size + 4) + 4;
        wr32(tls - 4, tls);    /* pointer slot used by __declspec(thread) code */
        if (e > s) memcpy(g_mem + tls, g_mem + s, e - s);
        wr32(kthread + 0x28, tls);
        uint32_t idx = rd32(tls_dir + 8);
        if (idx) wr32(idx, 0);
    } else {
        wr32(kthread + 0x28, kdata_alloc(tls_size));
    }
    c->fs_base = kpcr;
}

uint32_t current_kthread(CPU *c) { return rd32(c->fs_base + 0x28); }

CPU *cpu_new(uint32_t stack_size) {
    CPU *c = calloc(1, sizeof(CPU));
    if (stack_size < 0x10000) stack_size = 0x10000;
    uint32_t stack = guest_alloc(stack_size, PAGE);
    c->stack_limit = stack;
    c->stack_base = stack + stack_size;
    c->esp = c->stack_base - 0x10;
    c->fcw = 0x027F;
    c->mxcsr = 0x1F80;
    c->thread_index = __atomic_fetch_add(&g_thread_count, 1, __ATOMIC_SEQ_CST);
    cpu_setup_thread_block(c);
    return c;
}

/* ------------------------------------------------------------------ init */
/* HALO_SAMPLE_PROFILE=file (Linux): sample the program counter of whichever
 * thread is running 500 times a CPU-second and write "offset count" lines
 * (offsets into the executable) to `file` every 10 s; map them to
 * functions with `nm -n`. */
#if defined(__linux__)
#include <signal.h>
#include <sys/time.h>
#include <ucontext.h>
#define PROF_SLOTS 65536
static uintptr_t g_prof_pc[PROF_SLOTS];
static uint32_t g_prof_n[PROF_SLOTS];
static uintptr_t g_prof_base;
static void prof_signal(int sig, siginfo_t *si, void *uc_) {
    (void)sig; (void)si;
    ucontext_t *uc = uc_;
#if defined(__aarch64__)
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.pc;
#elif defined(__x86_64__)
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
#else
    uintptr_t pc = 0;
#endif
    pc -= g_prof_base;
    uint32_t h = (uint32_t)((pc >> 2) * 2654435761u) % PROF_SLOTS;
    for (int i = 0; i < 64; i++, h = (h + 1) % PROF_SLOTS) {
        if (g_prof_pc[h] == pc || !g_prof_pc[h]) { g_prof_pc[h] = pc; __atomic_add_fetch(&g_prof_n[h], 1, __ATOMIC_RELAXED); return; }
    }
}
static void *prof_writer(void *arg) {
    const char *path = arg;
    for (;;) {
        sleep(10);
        FILE *f = fopen(path, "w");
        if (!f) continue;
        for (int i = 0; i < PROF_SLOTS; i++) if (g_prof_n[i]) fprintf(f, "%lx %u\n", (unsigned long)g_prof_pc[i], g_prof_n[i]);
        fclose(f);
    }
    return NULL;
}
static void prof_start(void) {
    const char *path = getenv("HALO_SAMPLE_PROFILE");
    if (!path) return;
    Dl_info di;
    if (dladdr((void *)prof_start, &di)) g_prof_base = (uintptr_t)di.dli_fbase;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = prof_signal;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGPROF, &sa, NULL);
    struct itimerval it = {{0, 2000}, {0, 2000}};
    setitimer(ITIMER_PROF, &it, NULL);
    pthread_t t;
    pthread_create(&t, NULL, prof_writer, (void *)path);
    runtime_log("[profile] sampling into %s", path);
}
#else
static void prof_start(void) {}
#endif

int runtime_init(const RuntimeConfig *cfg) {
    prof_start();
    g_cfg = *cfg;
    pthread_mutex_init(&g_gil, NULL);
    pthread_cond_init(&g_kcond, NULL);
    if (mem_init()) return -1;
    kernel_init();
    if (load_xbe(cfg->xbe_path)) return -1;
    hle_init();
    runtime_log("[runtime] %u recompiled functions, entry %08x", g_recomp_func_count, g_recomp_entry);
    return 0;
}

void runtime_run(void) {
    uint32_t stack_commit = rd32(GUEST_IMAGE_BASE + XBE_STACK_COMMIT);
    CPU *c = cpu_new(stack_commit > 0x40000 ? stack_commit : 0x40000);
    t_cpu = c;
    guest_lock();
    push32(c, 0xFFFFFFF0u);   /* sentinel return address */
    GuestFn entry = recomp_lookup(g_recomp_entry);
    if (!entry) { runtime_log("[runtime] entry point was not recompiled"); exit(2); }
    runtime_log("[runtime] jumping to entry point %08x", g_recomp_entry);
    entry(c);
    runtime_log("[runtime] startup thread finished (eax=%08x); game threads keep running", c->eax);
    for (;;) guest_wait_change(host_time_ns() + 1000000000ull);
}
