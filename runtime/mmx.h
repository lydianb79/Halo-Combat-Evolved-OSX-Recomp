/* mmx.h — MMX helpers for recompiled code (64-bit packed integer ops).
 * Lanes are processed in plain C; compilers vectorize these well enough. */
#pragma once
#include <stdint.h>

#define MMX_LANES(T, N, a, b, body)                               \
    do {                                                          \
        T x_[N], y_[N], r_[N];                                    \
        memcpy(x_, &(a), 8); memcpy(y_, &(b), 8);                 \
        for (int i_ = 0; i_ < (N); i_++) { body; }                \
        uint64_t o_; memcpy(&o_, r_, 8); return o_;               \
    } while (0)

static inline int32_t mmx_sat(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : v > hi ? hi : v; }

/* add/sub */
static inline uint64_t mmx_paddb(uint64_t a, uint64_t b) { MMX_LANES(uint8_t, 8, a, b, r_[i_] = (uint8_t)(x_[i_] + y_[i_])); }
static inline uint64_t mmx_paddw(uint64_t a, uint64_t b) { MMX_LANES(uint16_t, 4, a, b, r_[i_] = (uint16_t)(x_[i_] + y_[i_])); }
static inline uint64_t mmx_paddd(uint64_t a, uint64_t b) { MMX_LANES(uint32_t, 2, a, b, r_[i_] = x_[i_] + y_[i_]); }
static inline uint64_t mmx_paddsb(uint64_t a, uint64_t b) { MMX_LANES(int8_t, 8, a, b, r_[i_] = (int8_t)mmx_sat(x_[i_] + y_[i_], -128, 127)); }
static inline uint64_t mmx_paddsw(uint64_t a, uint64_t b) { MMX_LANES(int16_t, 4, a, b, r_[i_] = (int16_t)mmx_sat(x_[i_] + y_[i_], -32768, 32767)); }
static inline uint64_t mmx_paddusb(uint64_t a, uint64_t b) { MMX_LANES(uint8_t, 8, a, b, r_[i_] = (uint8_t)mmx_sat(x_[i_] + y_[i_], 0, 255)); }
static inline uint64_t mmx_paddusw(uint64_t a, uint64_t b) { MMX_LANES(uint16_t, 4, a, b, r_[i_] = (uint16_t)mmx_sat(x_[i_] + y_[i_], 0, 65535)); }
static inline uint64_t mmx_psubb(uint64_t a, uint64_t b) { MMX_LANES(uint8_t, 8, a, b, r_[i_] = (uint8_t)(x_[i_] - y_[i_])); }
static inline uint64_t mmx_psubw(uint64_t a, uint64_t b) { MMX_LANES(uint16_t, 4, a, b, r_[i_] = (uint16_t)(x_[i_] - y_[i_])); }
static inline uint64_t mmx_psubd(uint64_t a, uint64_t b) { MMX_LANES(uint32_t, 2, a, b, r_[i_] = x_[i_] - y_[i_]); }
static inline uint64_t mmx_psubsb(uint64_t a, uint64_t b) { MMX_LANES(int8_t, 8, a, b, r_[i_] = (int8_t)mmx_sat(x_[i_] - y_[i_], -128, 127)); }
static inline uint64_t mmx_psubsw(uint64_t a, uint64_t b) { MMX_LANES(int16_t, 4, a, b, r_[i_] = (int16_t)mmx_sat(x_[i_] - y_[i_], -32768, 32767)); }
static inline uint64_t mmx_psubusb(uint64_t a, uint64_t b) { MMX_LANES(uint8_t, 8, a, b, r_[i_] = (uint8_t)mmx_sat((int32_t)x_[i_] - y_[i_], 0, 255)); }
static inline uint64_t mmx_psubusw(uint64_t a, uint64_t b) { MMX_LANES(uint16_t, 4, a, b, r_[i_] = (uint16_t)mmx_sat((int32_t)x_[i_] - y_[i_], 0, 65535)); }

/* multiply */
static inline uint64_t mmx_pmullw(uint64_t a, uint64_t b) { MMX_LANES(int16_t, 4, a, b, r_[i_] = (int16_t)(x_[i_] * y_[i_])); }
static inline uint64_t mmx_pmulhw(uint64_t a, uint64_t b) { MMX_LANES(int16_t, 4, a, b, r_[i_] = (int16_t)((x_[i_] * y_[i_]) >> 16)); }
static inline uint64_t mmx_pmulhuw(uint64_t a, uint64_t b) { MMX_LANES(uint16_t, 4, a, b, r_[i_] = (uint16_t)(((uint32_t)x_[i_] * y_[i_]) >> 16)); }
static inline uint64_t mmx_pmaddwd(uint64_t a, uint64_t b) {
    int16_t x[4], y[4]; int32_t r[2];
    memcpy(x, &a, 8); memcpy(y, &b, 8);
    r[0] = x[0] * y[0] + x[1] * y[1];
    r[1] = x[2] * y[2] + x[3] * y[3];
    uint64_t o; memcpy(&o, r, 8); return o;
}

/* compare */
static inline uint64_t mmx_pcmpeqb(uint64_t a, uint64_t b) { MMX_LANES(uint8_t, 8, a, b, r_[i_] = x_[i_] == y_[i_] ? 0xFF : 0); }
static inline uint64_t mmx_pcmpeqw(uint64_t a, uint64_t b) { MMX_LANES(uint16_t, 4, a, b, r_[i_] = x_[i_] == y_[i_] ? 0xFFFF : 0); }
static inline uint64_t mmx_pcmpeqd(uint64_t a, uint64_t b) { MMX_LANES(uint32_t, 2, a, b, r_[i_] = x_[i_] == y_[i_] ? 0xFFFFFFFFu : 0); }
static inline uint64_t mmx_pcmpgtb(uint64_t a, uint64_t b) { MMX_LANES(int8_t, 8, a, b, r_[i_] = x_[i_] > y_[i_] ? -1 : 0); }
static inline uint64_t mmx_pcmpgtw(uint64_t a, uint64_t b) { MMX_LANES(int16_t, 4, a, b, r_[i_] = x_[i_] > y_[i_] ? -1 : 0); }
static inline uint64_t mmx_pcmpgtd(uint64_t a, uint64_t b) { MMX_LANES(int32_t, 2, a, b, r_[i_] = x_[i_] > y_[i_] ? -1 : 0); }

/* SSE integer extensions on MMX registers (Pentium III) */
static inline uint64_t mmx_pavgb(uint64_t a, uint64_t b) { MMX_LANES(uint8_t, 8, a, b, r_[i_] = (uint8_t)((x_[i_] + y_[i_] + 1) >> 1)); }
static inline uint64_t mmx_pavgw(uint64_t a, uint64_t b) { MMX_LANES(uint16_t, 4, a, b, r_[i_] = (uint16_t)((x_[i_] + y_[i_] + 1) >> 1)); }
static inline uint64_t mmx_pmaxub(uint64_t a, uint64_t b) { MMX_LANES(uint8_t, 8, a, b, r_[i_] = x_[i_] > y_[i_] ? x_[i_] : y_[i_]); }
static inline uint64_t mmx_pminub(uint64_t a, uint64_t b) { MMX_LANES(uint8_t, 8, a, b, r_[i_] = x_[i_] < y_[i_] ? x_[i_] : y_[i_]); }
static inline uint64_t mmx_pmaxsw(uint64_t a, uint64_t b) { MMX_LANES(int16_t, 4, a, b, r_[i_] = x_[i_] > y_[i_] ? x_[i_] : y_[i_]); }
static inline uint64_t mmx_pminsw(uint64_t a, uint64_t b) { MMX_LANES(int16_t, 4, a, b, r_[i_] = x_[i_] < y_[i_] ? x_[i_] : y_[i_]); }

/* pack */
static inline uint64_t mmx_packuswb(uint64_t a, uint64_t b) {
    int16_t x[4], y[4]; uint8_t r[8];
    memcpy(x, &a, 8); memcpy(y, &b, 8);
    for (int i = 0; i < 4; i++) { r[i] = (uint8_t)mmx_sat(x[i], 0, 255); r[i + 4] = (uint8_t)mmx_sat(y[i], 0, 255); }
    uint64_t o; memcpy(&o, r, 8); return o;
}
static inline uint64_t mmx_packsswb(uint64_t a, uint64_t b) {
    int16_t x[4], y[4]; int8_t r[8];
    memcpy(x, &a, 8); memcpy(y, &b, 8);
    for (int i = 0; i < 4; i++) { r[i] = (int8_t)mmx_sat(x[i], -128, 127); r[i + 4] = (int8_t)mmx_sat(y[i], -128, 127); }
    uint64_t o; memcpy(&o, r, 8); return o;
}
static inline uint64_t mmx_packssdw(uint64_t a, uint64_t b) {
    int32_t x[2], y[2]; int16_t r[4];
    memcpy(x, &a, 8); memcpy(y, &b, 8);
    r[0] = (int16_t)mmx_sat(x[0], -32768, 32767); r[1] = (int16_t)mmx_sat(x[1], -32768, 32767);
    r[2] = (int16_t)mmx_sat(y[0], -32768, 32767); r[3] = (int16_t)mmx_sat(y[1], -32768, 32767);
    uint64_t o; memcpy(&o, r, 8); return o;
}

/* unpack (interleave) */
static inline uint64_t mmx_unpack(uint64_t a, uint64_t b, int elem, int high) {
    uint8_t x[8], y[8], r[8];
    memcpy(x, &a, 8); memcpy(y, &b, 8);
    int half = high ? 4 : 0, n = 4 / elem;
    for (int i = 0; i < n; i++) {
        memcpy(r + (2 * i) * elem, x + half + i * elem, (size_t)elem);
        memcpy(r + (2 * i + 1) * elem, y + half + i * elem, (size_t)elem);
    }
    uint64_t o; memcpy(&o, r, 8); return o;
}
#define mmx_punpcklbw(a, b) mmx_unpack(a, b, 1, 0)
#define mmx_punpcklwd(a, b) mmx_unpack(a, b, 2, 0)
#define mmx_punpckldq(a, b) mmx_unpack(a, b, 4, 0)
#define mmx_punpckhbw(a, b) mmx_unpack(a, b, 1, 1)
#define mmx_punpckhwd(a, b) mmx_unpack(a, b, 2, 1)
#define mmx_punpckhdq(a, b) mmx_unpack(a, b, 4, 1)

/* shifts (count is the full 64-bit operand) */
static inline uint64_t mmx_psllw(uint64_t a, uint64_t n) { if (n > 15) return 0; MMX_LANES(uint16_t, 4, a, a, r_[i_] = (uint16_t)(x_[i_] << n)); }
static inline uint64_t mmx_pslld(uint64_t a, uint64_t n) { if (n > 31) return 0; MMX_LANES(uint32_t, 2, a, a, r_[i_] = x_[i_] << n); }
static inline uint64_t mmx_psllq(uint64_t a, uint64_t n) { return n > 63 ? 0 : a << n; }
static inline uint64_t mmx_psrlw(uint64_t a, uint64_t n) { if (n > 15) return 0; MMX_LANES(uint16_t, 4, a, a, r_[i_] = (uint16_t)(x_[i_] >> n)); }
static inline uint64_t mmx_psrld(uint64_t a, uint64_t n) { if (n > 31) return 0; MMX_LANES(uint32_t, 2, a, a, r_[i_] = x_[i_] >> n); }
static inline uint64_t mmx_psrlq(uint64_t a, uint64_t n) { return n > 63 ? 0 : a >> n; }
static inline uint64_t mmx_psraw(uint64_t a, uint64_t n) { if (n > 15) n = 15; MMX_LANES(int16_t, 4, a, a, r_[i_] = (int16_t)(x_[i_] >> n)); }
static inline uint64_t mmx_psrad(uint64_t a, uint64_t n) { if (n > 31) n = 31; MMX_LANES(int32_t, 2, a, a, r_[i_] = x_[i_] >> n); }

/* bitwise */
static inline uint64_t mmx_pand(uint64_t a, uint64_t b) { return a & b; }
static inline uint64_t mmx_pandn(uint64_t a, uint64_t b) { return ~a & b; }
static inline uint64_t mmx_por(uint64_t a, uint64_t b) { return a | b; }
static inline uint64_t mmx_pxor(uint64_t a, uint64_t b) { return a ^ b; }
