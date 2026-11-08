/*
 * cpu.h — guest CPU model used by statically recompiled Xbox (x86-32) code.
 *
 * Every recompiled guest function has the signature `void f_XXXXXXXX(CPU *c)`.
 * Guest memory is a flat 4 GiB host reservation (g_mem); a guest address is
 * simply an offset into it, so any 32-bit address is safe to dereference.
 *
 * Flags are evaluated lazily: arithmetic records (op, a, b, result, size) and a
 * condition is computed only when a jcc/setcc/cmov/adc needs it. Because the
 * stores and loads are adjacent, an optimizing compiler forwards them and folds
 * the switch away, so a `cmp; jl` pair compiles to one compare-and-branch.
 */
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef union {
    float f[4];
    uint32_t u[4];
    int32_t i[4];
    uint64_t q[2];
    double d[2];
} XMMReg;

typedef struct CPU {
    uint32_t eax, ecx, edx, ebx, esp, ebp, esi, edi;
    /* lazy flags */
    uint32_t lf_op, lf_sz, lf_a, lf_b, lf_res, lf_cin;
    uint32_t eflags; /* explicit flags, valid when lf_op == LF_NONE */
    uint32_t df;     /* direction flag (kept separately) */
    uint32_t fs_base;
    /* x87 (modelled in double precision) */
    double st[8];
    uint32_t ftop;
    uint16_t fsw, fcw;
    /* SSE */
    XMMReg xmm[8];
    uint32_t mxcsr;
    /* host bookkeeping */
    int32_t budget;            /* back-edges left before a preemption point */
    int thread_index;
    uint32_t stack_base, stack_limit;
    /* MMX (aliases the x87 registers on real hardware; kept separate here) */
    uint64_t mm[8];
} CPU;

extern uint8_t *g_mem;

/* ---------------------------------------------------------------- memory */
/* 0xFD000000-0xFEFFFFFF is hardware (NV2A GPU, APU, ...): accesses there go to
 * the device models in nv2a.c. One predictable compare per access. */
#define MMIO_BASE 0xFD000000u
#define MMIO_SIZE 0x01F00000u
#define IS_MMIO(a) __builtin_expect((uint32_t)((a) - MMIO_BASE) < MMIO_SIZE, 0)
uint32_t mmio_read(uint32_t a, int size);
void mmio_write(uint32_t a, uint32_t v, int size);

static inline uint8_t  rd8 (uint32_t a) { if (IS_MMIO(a)) return (uint8_t)mmio_read(a, 1); return g_mem[a]; }
static inline uint16_t rd16(uint32_t a) { if (IS_MMIO(a)) return (uint16_t)mmio_read(a, 2); uint16_t v; memcpy(&v, g_mem + a, 2); return v; }
static inline uint32_t rd32(uint32_t a) { if (IS_MMIO(a)) return mmio_read(a, 4); uint32_t v; memcpy(&v, g_mem + a, 4); return v; }
static inline uint64_t rd64(uint32_t a) { uint64_t v; memcpy(&v, g_mem + a, 8); return v; }
static inline void wr8 (uint32_t a, uint8_t v)  { if (IS_MMIO(a)) { mmio_write(a, v, 1); return; } g_mem[a] = v; }
static inline void wr16(uint32_t a, uint16_t v) { if (IS_MMIO(a)) { mmio_write(a, v, 2); return; } memcpy(g_mem + a, &v, 2); }
static inline void wr32(uint32_t a, uint32_t v) { if (IS_MMIO(a)) { mmio_write(a, v, 4); return; } memcpy(g_mem + a, &v, 4); }
static inline void wr64(uint32_t a, uint64_t v) { memcpy(g_mem + a, &v, 8); }
static inline float  rdf32(uint32_t a) { float v;  memcpy(&v, g_mem + a, 4); return v; }
static inline double rdf64(uint32_t a) { double v; memcpy(&v, g_mem + a, 8); return v; }
static inline void wrf32(uint32_t a, float v)  { memcpy(g_mem + a, &v, 4); }
static inline void wrf64(uint32_t a, double v) { memcpy(g_mem + a, &v, 8); }
double rdf80(uint32_t a);
void   wrf80(uint32_t a, double v);

static inline void* guest_ptr(uint32_t a) { return g_mem + a; }

/* --------------------------------------------------------- sub-registers */
static inline void set8l(uint32_t *r, uint32_t v) { *r = (*r & 0xFFFFFF00u) | (v & 0xFFu); }
static inline void set8h(uint32_t *r, uint32_t v) { *r = (*r & 0xFFFF00FFu) | ((v & 0xFFu) << 8); }
static inline void set16(uint32_t *r, uint32_t v) { *r = (*r & 0xFFFF0000u) | (v & 0xFFFFu); }

/* ------------------------------------------------------------------ stack */
static inline void push32(CPU *c, uint32_t v) { c->esp -= 4; wr32(c->esp, v); }
static inline uint32_t pop32(CPU *c) { uint32_t v = rd32(c->esp); c->esp += 4; return v; }
static inline void push16(CPU *c, uint32_t v) { c->esp -= 2; wr16(c->esp, (uint16_t)v); }
static inline uint32_t pop16(CPU *c) { uint32_t v = rd16(c->esp); c->esp += 2; return v; }

/* ------------------------------------------------------------------ flags */
enum { LF_NONE = 0, LF_ADD, LF_ADC, LF_SUB, LF_SBB, LF_LOGIC, LF_INC, LF_DEC,
       LF_NEG, LF_SHL, LF_SHR, LF_SAR, LF_MUL };
enum { FL_CF = 0x1, FL_PF = 0x4, FL_AF = 0x10, FL_ZF = 0x40, FL_SF = 0x80,
       FL_DF = 0x400, FL_OF = 0x800 };

static inline uint32_t sz_mask(uint32_t sz) { return sz == 4 ? 0xFFFFFFFFu : sz == 2 ? 0xFFFFu : 0xFFu; }
static inline uint32_t sz_sign(uint32_t sz) { return 1u << (sz * 8 - 1); }

static inline void setf(CPU *c, uint32_t op, uint32_t sz, uint32_t a, uint32_t b, uint32_t r) {
    c->lf_op = op; c->lf_sz = sz; c->lf_a = a; c->lf_b = b; c->lf_res = r;
}

static inline int f_cf(const CPU *c) {
    uint32_t m = sz_mask(c->lf_sz), a = c->lf_a & m, b = c->lf_b & m, r = c->lf_res & m;
    switch (c->lf_op) {
    case LF_NONE: return c->eflags & FL_CF;
    case LF_ADD:  return r < a;
    case LF_ADC:  return c->lf_cin ? r <= a : r < a;
    case LF_SUB:  return a < b;
    case LF_SBB:  return c->lf_cin ? a <= b : a < b;
    case LF_LOGIC: return 0;
    case LF_INC: case LF_DEC: return (int)c->lf_cin;
    case LF_NEG:  return a != 0;
    case LF_SHL:  return (int)(((uint64_t)a << c->lf_b) >> (c->lf_sz * 8)) & 1;
    case LF_SHR:  return (int)(a >> (c->lf_b - 1)) & 1;
    case LF_SAR: {
        int32_t sa = c->lf_sz == 4 ? (int32_t)a : c->lf_sz == 2 ? (int16_t)a : (int8_t)a;
        return (int)((sa >> (c->lf_b > 31 ? 31 : c->lf_b - 1)) & 1);
    }
    case LF_MUL:  return (int)c->lf_b;
    }
    return 0;
}
static inline int f_zf(const CPU *c) {
    return c->lf_op == LF_NONE ? !!(c->eflags & FL_ZF) : (c->lf_res & sz_mask(c->lf_sz)) == 0;
}
static inline int f_sf(const CPU *c) {
    return c->lf_op == LF_NONE ? !!(c->eflags & FL_SF) : !!(c->lf_res & sz_sign(c->lf_sz));
}
static inline int f_pf(const CPU *c) {
    return c->lf_op == LF_NONE ? !!(c->eflags & FL_PF) : !__builtin_parity(c->lf_res & 0xFFu);
}
static inline int f_of(const CPU *c) {
    uint32_t s = sz_sign(c->lf_sz), m = sz_mask(c->lf_sz);
    uint32_t a = c->lf_a, b = c->lf_b, r = c->lf_res;
    switch (c->lf_op) {
    case LF_NONE: return !!(c->eflags & FL_OF);
    case LF_ADD: case LF_ADC: return !!(((a ^ r) & (b ^ r)) & s);
    case LF_SUB: case LF_SBB: return !!(((a ^ b) & (a ^ r)) & s);
    case LF_LOGIC: return 0;
    case LF_INC: return (r & m) == s;
    case LF_DEC: return (r & m) == s - 1;
    case LF_NEG: return (r & m) == s;
    case LF_SHL: return (!!(r & s)) ^ f_cf(c);
    case LF_SHR: return c->lf_b == 1 ? !!(a & s) : 0;
    case LF_SAR: return 0;
    case LF_MUL: return (int)c->lf_b;
    }
    return 0;
}
static inline uint32_t flags_get(const CPU *c) {
    uint32_t f = 0x2;
    if (c->lf_op == LF_NONE) f = (c->eflags & ~FL_DF) | 0x2;
    else f |= (f_cf(c) ? FL_CF : 0) | (f_pf(c) ? FL_PF : 0) | (f_zf(c) ? FL_ZF : 0) |
              (f_sf(c) ? FL_SF : 0) | (f_of(c) ? FL_OF : 0);
    return f | (c->df ? FL_DF : 0);
}
static inline void flags_set(CPU *c, uint32_t f) {
    c->eflags = f & ~FL_DF; c->df = !!(f & FL_DF); c->lf_op = LF_NONE;
}

/* condition codes */
#define CC_O(c)  f_of(c)
#define CC_NO(c) (!f_of(c))
#define CC_B(c)  f_cf(c)
#define CC_AE(c) (!f_cf(c))
#define CC_E(c)  f_zf(c)
#define CC_NE(c) (!f_zf(c))
#define CC_BE(c) (f_cf(c) || f_zf(c))
#define CC_A(c)  (!f_cf(c) && !f_zf(c))
#define CC_S(c)  f_sf(c)
#define CC_NS(c) (!f_sf(c))
#define CC_P(c)  f_pf(c)
#define CC_NP(c) (!f_pf(c))
#define CC_L(c)  (f_sf(c) != f_of(c))
#define CC_GE(c) (f_sf(c) == f_of(c))
#define CC_LE(c) (f_zf(c) || f_sf(c) != f_of(c))
#define CC_G(c)  (!f_zf(c) && f_sf(c) == f_of(c))

/* -------------------------------------------------------------------- x87 */
#define ST(c, i) ((c)->st[((c)->ftop + (i)) & 7])
static inline void fpush(CPU *c, double v) { c->ftop = (c->ftop - 1) & 7; c->st[c->ftop] = v; }
static inline void fpop(CPU *c) { c->ftop = (c->ftop + 1) & 7; }
static inline uint16_t fpu_sw(const CPU *c) { return (uint16_t)((c->fsw & ~0x3800) | ((c->ftop & 7) << 11)); }
static inline void fcompare(CPU *c, double a, double b) {
    uint16_t cc;
    if (a != a || b != b) cc = 0x4500;      /* unordered: C3 C2 C0 */
    else if (a > b) cc = 0;
    else if (a < b) cc = 0x0100;           /* C0 */
    else cc = 0x4000;                      /* C3 */
    c->fsw = (uint16_t)((c->fsw & ~0x4700) | cc);
}
static inline void fcomi(CPU *c, double a, double b) {
    uint32_t f = flags_get(c) & ~(FL_ZF | FL_PF | FL_CF | FL_OF | FL_SF | FL_AF);
    if (a != a || b != b) f |= FL_ZF | FL_PF | FL_CF;
    else if (a < b) f |= FL_CF;
    else if (a == b) f |= FL_ZF;
    flags_set(c, f);
}
static inline double fround(const CPU *c, double v) {
    switch ((c->fcw >> 10) & 3) {
    case 0: return nearbyint(v);
    case 1: return floor(v);
    case 2: return ceil(v);
    default: return trunc(v);
    }
}
static inline int32_t fist32(const CPU *c, double v) {
    double r = fround(c, v);
    return (r >= -2147483648.0 && r <= 2147483647.0) ? (int32_t)r : INT32_MIN;
}
static inline int16_t fist16(const CPU *c, double v) {
    double r = fround(c, v);
    return (r >= -32768.0 && r <= 32767.0) ? (int16_t)r : INT16_MIN;
}
static inline int64_t fist64(const CPU *c, double v) {
    double r = fround(c, v);
    return (r >= -9.2233720368547758e18 && r < 9.2233720368547758e18) ? (int64_t)r : INT64_MIN;
}
static inline void fxam(CPU *c) {
    double v = ST(c, 0);
    uint16_t cc = signbit(v) ? 0x0200 : 0; /* C1 = sign */
    if (isnan(v)) cc |= 0x0100;
    else if (isinf(v)) cc |= 0x0500;
    else if (v == 0) cc |= 0x4000;
    else if (fpclassify(v) == FP_SUBNORMAL) cc |= 0x4400;
    else cc |= 0x0400;
    c->fsw = (uint16_t)((c->fsw & ~0x4700) | cc);
}

/* ------------------------------------------------------- string helpers */
static inline int32_t str_step(const CPU *c, int sz) { return c->df ? -sz : sz; }

/* ------------------------------------------------------- runtime hooks */
void call_indirect(CPU *c, uint32_t target);           /* return address already pushed */
void recomp_trap(CPU *c, uint32_t eip, const char *what) __attribute__((noreturn));
void recomp_unimpl(CPU *c, uint32_t eip, const char *insn) __attribute__((noreturn));
void guest_int3(CPU *c, uint32_t eip);
void guest_port_io(CPU *c, uint32_t eip, int is_out, uint32_t port, int size);
uint32_t guest_rdtsc_lo(CPU *c, uint32_t *hi);
void guest_cpuid(CPU *c);

#define UNIMPL(c, eip, s) recomp_unimpl((c), (eip), (s))

#include "mmx.h"

/* Emitted on loop back-edges: lets other guest threads (and the kernel timer
 * thread) run even when the game busy-waits, like the Xbox's preemptive
 * scheduler would. */
void guest_preempt(CPU *c);
#define SAFEPOINT(c) do { if (__builtin_expect(--(c)->budget < 0, 0)) guest_preempt(c); } while (0)

#ifdef __cplusplus
}
#endif
