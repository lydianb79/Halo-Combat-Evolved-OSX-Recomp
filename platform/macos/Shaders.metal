// Shaders.metal — presentation pass for the Halo recompilation: places the
// 640x480 game picture in the window (scaling, aspect ratio, filtering) and
// applies the optional scanlines, brightness and gamma from Graphics Settings.
#include <metal_stdlib>
using namespace metal;

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

// Layout must match PresentParams in graphics.h.
struct PresentParams {
    float2 src_size;      // game picture, texels
    float2 rect_origin;   // where it goes in the drawable, pixels (top-left origin)
    float2 rect_size;
    int filter;           // 0 nearest, 1 bilinear, 2 sharp bilinear, 3 bicubic, 4 Lanczos
    float scanlines;      // 0..1
    float brightness;
    float gamma;
};

// One oversized triangle covering the drawable; no vertex buffer needed.
vertex VSOut fullscreen_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.position = float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = p;
    return o;
}

// Upload of guest-memory pixels into a render target drawn at a higher
// resolution (Graphics → Render resolution): the source is stretched over the viewport.
fragment float4 copy_fs(VSOut in [[stage_in]], texture2d<float> src [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    return src.sample(s, in.uv);
}

// Catmull-Rom weights for the four taps around a sample point.
static float4 cubic_weights(float t) {
    float t2 = t * t, t3 = t2 * t;
    return float4(-0.5 * t3 + t2 - 0.5 * t,
                  1.5 * t3 - 2.5 * t2 + 1.0,
                  -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                  0.5 * t3 - 0.5 * t2);
}

// Lanczos-2 weights for the four taps at distances 1+t, t, 1-t, 2-t.
static float lanczos2(float x) {
    x = fabs(x);
    if (x < 1e-4) return 1.0;
    if (x >= 2.0) return 0.0;
    float px = M_PI_F * x;
    return 2.0 * sin(px) * sin(px * 0.5) / (px * px);
}
static float4 lanczos_weights(float t) {
    float4 w = float4(lanczos2(1.0 + t), lanczos2(t), lanczos2(1.0 - t), lanczos2(2.0 - t));
    return w / (w.x + w.y + w.z + w.w);
}

fragment float4 present_fs(VSOut in [[stage_in]],
                           texture2d<float> frame [[texture(0)]],
                           sampler nearest_s [[sampler(0)]],
                           sampler linear_s [[sampler(1)]],
                           constant PresentParams &p [[buffer(0)]]) {
    float2 uv = (in.position.xy - p.rect_origin) / p.rect_size;
    if (any(uv < 0.0) || any(uv > 1.0)) return float4(0, 0, 0, 1);
    float2 texel = uv * p.src_size;
    float3 c;
    if (p.filter == 0) {
        c = frame.sample(nearest_s, uv).rgb;
    } else if (p.filter == 2) {
        // Sharp bilinear: nearest-neighbour inside each texel, a one-pixel
        // linear blend at texel edges (no shimmer at non-integer scales).
        float2 scale = max(p.rect_size / p.src_size, float2(1.0));
        float2 base = floor(texel);
        float2 d = fract(texel) - 0.5;
        float2 range = 0.5 - 0.5 / scale;
        float2 f = (d - clamp(d, -range, range)) * scale + 0.5;
        c = frame.sample(linear_s, (base + f) / p.src_size).rgb;
    } else if (p.filter == 3 || p.filter == 4) {
        float2 t = texel - 0.5;
        float2 f = fract(t);
        int2 base = int2(floor(t));
        float4 wx = p.filter == 3 ? cubic_weights(f.x) : lanczos_weights(f.x);
        float4 wy = p.filter == 3 ? cubic_weights(f.y) : lanczos_weights(f.y);
        int2 hi = int2(p.src_size) - 1;
        float3 acc = float3(0);
        for (int j = 0; j < 4; j++) {
            float3 row = float3(0);
            for (int i = 0; i < 4; i++) {
                int2 q = clamp(base + int2(i - 1, j - 1), int2(0), hi);
                row += frame.read(uint2(q)).rgb * wx[i];
            }
            acc += row * wy[j];
        }
        c = saturate(acc);
    } else {
        c = frame.sample(linear_s, uv).rgb;
    }
    if (p.scanlines > 0.0) {
        // Bright at the centre of each game line, dark between lines.
        float line = 0.5 - 0.5 * cos(texel.y * 2.0 * M_PI_F);
        c *= 1.0 - p.scanlines * (1.0 - line);
    }
    c = saturate(c * p.brightness);
    if (p.gamma != 1.0) c = pow(c, float3(1.0 / p.gamma));
    return float4(c, 1.0);
}

// =====================================================================
// NV2A pixel pipeline on the GPU (metal_backend.m). Vertices arrive
// transformed, clipped and culled by the CPU; this does texturing, the
// register combiners and alpha test; depth test and blending are Metal
// fixed-function state. Mirrors the CPU reference in runtime/nv2a_raster.c.
// =====================================================================

struct NVVertex {           // layout must match GVtx in metal_backend.m
    float4 pos;             // clip space (already multiplied by w)
    float4 d0, d1;          // diffuse, specular
    float4 t0, t1, t2, t3;  // texture coordinates
    float4 fog;             // x: fog coordinate
};

struct NVVaryings {
    float4 pos [[position, invariant]];   // identical depth in every pass (Halo draws surfaces again with depth EQUAL)
    float4 d0, d1;
    float4 t0, t1, t2, t3;
    float fog;
};

// Layout must match FragUniforms in metal_backend.m (544 bytes).
struct NVFragUniforms {
    float4 factor0[8];
    float4 factor1[8];
    float4 fog;
    float4 tex_scale[4];    // xy: 1 for normalized coordinates, 1/size for texel coordinates
    float4 specfog[2];      // final combiner constants C0/C1
    uint color_icw[8], color_ocw[8], alpha_icw[8], alpha_ocw[8];
    uint comb_stages, comb_control, final0, final1;
    uint alpha_test, alpha_func, tex_mask, debug_view;   // tex_mask bits 0-3: 2D stages, 4-7: cube stages; debug_view: diagnosis
    float alpha_ref, fog_bias, fog_mult, fog_code;      // fog: see nv_fog
};

vertex NVVaryings nv_vs(uint vid [[vertex_id]], const device NVVertex *v [[buffer(0)]]) {
    NVVertex in = v[vid];
    NVVaryings o;
    o.pos = in.pos;
    o.d0 = in.d0; o.d1 = in.d1;
    o.t0 = in.t0; o.t1 = in.t1; o.t2 = in.t2; o.t3 = in.t3;
    o.fog = in.fog.x;
    return o;
}

// Register combiner input mapping (input byte bits 7:5).
static float nv_map(float x, uint m) {
    switch (m) {
    case 0: return max(x, 0.0);                     // unsigned identity
    case 1: return 1.0 - saturate(x);               // unsigned invert
    case 2: return 2.0 * max(x, 0.0) - 1.0;         // expand normal
    case 3: return -2.0 * max(x, 0.0) + 1.0;        // expand negate
    case 4: return max(x, 0.0) - 0.5;               // half-bias normal
    case 5: return -max(x, 0.0) + 0.5;              // half-bias negate
    case 6: return x;                               // signed identity
    default: return -x;                             // signed negate
    }
}

static float3 nv_in_rgb(thread const float4 *reg, uint sel) {
    float4 r = reg[sel & 15];
    uint m = (sel >> 5) & 7;
    if (sel & 0x10) { float a = nv_map(r.w, m); return float3(a, a, a); }
    return float3(nv_map(r.x, m), nv_map(r.y, m), nv_map(r.z, m));
}

static float nv_in_a(thread const float4 *reg, uint sel) {
    float4 r = reg[sel & 15];
    return nv_map((sel & 0x10) ? r.w : r.z, (sel >> 5) & 7);
}

static float nv_out(float x, uint op) {
    switch (op) {
    case 1: x = x - 0.5; break;
    case 2: x = x * 2.0; break;
    case 3: x = (x - 0.5) * 2.0; break;
    case 4: x = x * 4.0; break;
    case 6: x = x * 0.5; break;
    default: break;
    }
    return clamp(x, -1.0, 1.0);
}

static float3 nv_fin(thread const float4 *reg, uint sel) {
    float4 r = reg[sel & 15];
    float3 k = (sel & 0x10) ? float3(r.w, r.w, r.w) : r.xyz;
    k = saturate(k);
    if (sel & 0x20) k = 1.0 - k;
    return k;
}

static float4 nv_combiners(thread float4 *reg, constant NVFragUniforms &u) {
    uint n = min(u.comb_stages, 8u);
    bool msb_mux = ((u.comb_control >> 8) & 1) != 0;
    for (uint i = 0; i < n; i++) {
        reg[1] = u.factor0[((u.comb_control >> 12) & 1) ? i : 0];
        reg[2] = u.factor1[((u.comb_control >> 16) & 1) ? i : 0];
        uint icw = u.color_icw[i], ocw = u.color_ocw[i];
        float3 A = nv_in_rgb(reg, icw >> 24), B = nv_in_rgb(reg, (icw >> 16) & 255);
        float3 C = nv_in_rgb(reg, (icw >> 8) & 255), D = nv_in_rgb(reg, icw & 255);
        float3 ab = (ocw & (1u << 13)) ? float3(dot(A, B)) : A * B;
        float3 cd = (ocw & (1u << 12)) ? float3(dot(C, D)) : C * D;
        bool mux_sel = msb_mux ? reg[12].w >= 0.5 : ((int(reg[12].w * 255.0) & 1) != 0);
        float3 sum = (ocw & (1u << 14)) ? (mux_sel ? cd : ab) : ab + cd;
        uint op = (ocw >> 15) & 7;
        uint aicw = u.alpha_icw[i], aocw = u.alpha_ocw[i];
        float aA = nv_in_a(reg, aicw >> 24), aB = nv_in_a(reg, (aicw >> 16) & 255);
        float aC = nv_in_a(reg, (aicw >> 8) & 255), aD = nv_in_a(reg, aicw & 255);
        float aab = aA * aB, acd = aC * aD;
        float asum = (aocw & (1u << 14)) ? (mux_sel ? acd : aab) : aab + acd;
        uint aop = (aocw >> 15) & 7;
        uint d_cd = ocw & 15, d_ab = (ocw >> 4) & 15, d_sum = (ocw >> 8) & 15;
        if (d_ab) {
            reg[d_ab].xyz = float3(nv_out(ab.x, op), nv_out(ab.y, op), nv_out(ab.z, op));
            if (ocw & (1u << 19)) reg[d_ab].w = reg[d_ab].z;
        }
        if (d_cd) {
            reg[d_cd].xyz = float3(nv_out(cd.x, op), nv_out(cd.y, op), nv_out(cd.z, op));
            if (ocw & (1u << 18)) reg[d_cd].w = reg[d_cd].z;
        }
        if (d_sum) reg[d_sum].xyz = float3(nv_out(sum.x, op), nv_out(sum.y, op), nv_out(sum.z, op));
        uint ad_cd = aocw & 15, ad_ab = (aocw >> 4) & 15, ad_sum = (aocw >> 8) & 15;
        if (ad_ab) reg[ad_ab].w = nv_out(aab, aop);
        if (ad_cd) reg[ad_cd].w = nv_out(acd, aop);
        if (ad_sum) reg[ad_sum].w = nv_out(asum, aop);
    }
    // final combiner: rgb = A*B + (1-A)*C + D, alpha = G; extra inputs E*F (15) and V1+R0 (14);
    // its constants C0/C1 are the specular/fog factors
    reg[1] = u.specfog[0];
    reg[2] = u.specfog[1];
    float3 s = reg[5].xyz + reg[12].xyz;
    if (u.final1 & 0x80) s = saturate(s);
    reg[14] = float4(s, 0.0);
    float3 E = nv_fin(reg, u.final1 >> 24), F = nv_fin(reg, (u.final1 >> 16) & 255);
    reg[15] = float4(E * F, 0.0);
    float3 A = nv_fin(reg, u.final0 >> 24), B = nv_fin(reg, (u.final0 >> 16) & 255);
    float3 C = nv_fin(reg, (u.final0 >> 8) & 255), D = nv_fin(reg, u.final0 & 255);
    uint g = (u.final1 >> 8) & 255;
    float ga = saturate((g & 0x10) ? reg[g & 15].w : reg[g & 15].z);
    if (g & 0x20) ga = 1.0 - ga;
    return float4(saturate(A * B + (1.0 - A) * C + D), ga);
}

// NV2A fog factor from the interpolated fog coordinate (oFog.x); 1 = no fog.
// code: 1 linear, 2 exp, 3 exp2, +4 absolute coordinate; bias/multiplier = FOG_PARAMS 0/1
static float nv_fog(float d, constant NVFragUniforms &u) {
    int code = int(u.fog_code);
    if (code & 4) d = abs(d);
    if (isinf(d)) d = 0.0;
    float f;
    switch (code & 3) {
    case 1: f = u.fog_bias + d * u.fog_mult - 1.0; break;
    case 2: f = u.fog_bias + exp2(d * u.fog_mult * 16.0) - 1.5; break;
    default: f = u.fog_bias + exp2(-d * d * u.fog_mult * u.fog_mult * 32.0) - 1.5; break;
    }
    return saturate(f);
}

static bool nv_compare(uint func, float a, float b) {
    switch (func & 7) {
    case 0: return false;
    case 1: return a < b;
    case 2: return a == b;
    case 3: return a <= b;
    case 4: return a > b;
    case 5: return a != b;
    case 6: return a >= b;
    default: return true;
    }
}

// The alpha test drops a pixel through its coverage mask instead of
// discard_fragment().
//
// IMPORTANT: nv_fs must keep a (possibly never executed) write to device memory
// (the `dbg` buffer below). A fragment function with side effects makes Apple
// GPUs give up their early, hidden-surface-removal depth handling for it. With
// that handling, alpha-tested draws that also write depth (Halo's menu ring)
// came out with texel-sized holes even though no pixel failed the test; the
// shadow comparison against the CPU renderer went from ~26000 differing pixels
// to ~15 once the write was present, without the write ever running.
struct NVFragOut {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

fragment NVFragOut nv_fs(NVVaryings in [[stage_in]],
                      constant NVFragUniforms &u [[buffer(0)]],
                      texture2d<float> tex0 [[texture(0)]], texture2d<float> tex1 [[texture(1)]],
                      texture2d<float> tex2 [[texture(2)]], texture2d<float> tex3 [[texture(3)]],
                      texturecube<float> cube0 [[texture(4)]], texturecube<float> cube1 [[texture(5)]],
                      texturecube<float> cube2 [[texture(6)]], texturecube<float> cube3 [[texture(7)]],
                      sampler s0 [[sampler(0)]], sampler s1 [[sampler(1)]],
                      sampler s2 [[sampler(2)]], sampler s3 [[sampler(3)]],
                      device atomic_uint *dbg [[buffer(1)]]) {
    float4 reg[16];
    for (int i = 0; i < 16; i++) reg[i] = float4(0.0);
    reg[3] = u.fog;
    if (u.fog_code != 0.0) reg[3].w = nv_fog(in.fog, u);
    reg[4] = in.d0;
    reg[5] = in.d1;
    if (u.tex_mask & 1) reg[8] = tex0.sample(s0, in.t0.xy * u.tex_scale[0].xy);
    if (u.tex_mask & 2) reg[9] = tex1.sample(s1, in.t1.xy * u.tex_scale[1].xy);
    if (u.tex_mask & 4) reg[10] = tex2.sample(s2, in.t2.xy * u.tex_scale[2].xy);
    if (u.tex_mask & 8) reg[11] = tex3.sample(s3, in.t3.xy * u.tex_scale[3].xy);
    // cube maps: direction (s, t, r)
    if (u.tex_mask & 16) reg[8] = cube0.sample(s0, in.t0.xyz);
    if (u.tex_mask & 32) reg[9] = cube1.sample(s1, in.t1.xyz);
    if (u.tex_mask & 64) reg[10] = cube2.sample(s2, in.t2.xyz);
    if (u.tex_mask & 128) reg[11] = cube3.sample(s3, in.t3.xyz);
    if (u.debug_view == 9 && (u.tex_mask & 1)) reg[8] = tex0.sample(s0, in.t0.xy * u.tex_scale[0].xy, level(0.0));  // diagnosis
    reg[12].w = reg[8].w;                                    // r0.a starts as t0.a
    float t0a_before = reg[8].w;
    float4 col;
    if (u.comb_stages == 0) {
        col = in.d0;
        if (u.tex_mask & 17) col *= reg[8];
    } else {
        col = nv_combiners(reg, u);
    }
    if (u.debug_view == 7) col.w = t0a_before;               // diagnosis: alpha straight from texture 0
    if (u.debug_view == 8) col.w = 1.0;                      // diagnosis: alpha forced opaque
    NVFragOut o;
    o.color = col;
    o.mask = 0xFFFFFFFFu;
    bool keep = !u.alpha_test || nv_compare(u.alpha_func, col.w * 255.0, u.alpha_ref);
    if (u.debug_view == 4) { if (!keep) o.color = float4(1.0, 0.0, 0.0, 1.0); return o; }  // diagnosis: red where dropped
    if (u.debug_view == 6) { if (!keep) discard_fragment(); return o; }                    // diagnosis: old discard path
    if (!keep) o.mask = 0;
    if (u.debug_view == 10) {                                // diagnosis: statistics about texture 0 in this draw
        float2 c = in.t0.xy * u.tex_scale[0].xy;
        float l0a = (u.tex_mask & 1) ? tex0.sample(s0, c, level(0.0)).w : -1.0;
        atomic_fetch_add_explicit(&dbg[0], 1u, memory_order_relaxed);
        if (t0a_before < 0.5) atomic_fetch_add_explicit(&dbg[1], 1u, memory_order_relaxed);
        if (any(c < 0.0) || any(c > 1.0)) atomic_fetch_add_explicit(&dbg[2], 1u, memory_order_relaxed);
        if (!all(isfinite(in.t0))) atomic_fetch_add_explicit(&dbg[3], 1u, memory_order_relaxed);
        atomic_store_explicit(&dbg[4], u.tex_mask, memory_order_relaxed);
        atomic_store_explicit(&dbg[5], tex0.get_width(), memory_order_relaxed);
        if (l0a < 0.5) atomic_fetch_add_explicit(&dbg[6], 1u, memory_order_relaxed);
        if (!keep) atomic_fetch_add_explicit(&dbg[8], 1u, memory_order_relaxed);
        if (col.w < 0.5) atomic_fetch_add_explicit(&dbg[9], 1u, memory_order_relaxed);
        if (t0a_before < 0.5) {
            uint k = atomic_fetch_add_explicit(&dbg[7], 1u, memory_order_relaxed);
            if (k < 16) {
                device atomic_uint *r = dbg + 16 + k * 12;
                float vals[12] = {in.pos.x, in.pos.y, in.t0.x, in.t0.y, in.t0.z, in.t0.w,
                                  reg[8].x, reg[8].y, reg[8].z, t0a_before, l0a, col.w};
                for (int i = 0; i < 12; i++) atomic_store_explicit(&r[i], as_type<uint>(vals[i]), memory_order_relaxed);
            }
        }
    }
    return o;
}

// The same shading without the alpha test and without the device-memory
// write: used for every draw that has no alpha test (most of a frame), so the
// GPU keeps its hidden-surface removal and skips pixels that end up behind
// closer geometry. nv_fs (with the write) stays for alpha-tested draws.
fragment float4 nv_fs_fast(NVVaryings in [[stage_in]],
                           constant NVFragUniforms &u [[buffer(0)]],
                           texture2d<float> tex0 [[texture(0)]], texture2d<float> tex1 [[texture(1)]],
                           texture2d<float> tex2 [[texture(2)]], texture2d<float> tex3 [[texture(3)]],
                           texturecube<float> cube0 [[texture(4)]], texturecube<float> cube1 [[texture(5)]],
                           texturecube<float> cube2 [[texture(6)]], texturecube<float> cube3 [[texture(7)]],
                           sampler s0 [[sampler(0)]], sampler s1 [[sampler(1)]],
                           sampler s2 [[sampler(2)]], sampler s3 [[sampler(3)]]) {
    float4 reg[16];
    for (int i = 0; i < 16; i++) reg[i] = float4(0.0);
    reg[3] = u.fog;
    if (u.fog_code != 0.0) reg[3].w = nv_fog(in.fog, u);
    reg[4] = in.d0;
    reg[5] = in.d1;
    if (u.tex_mask & 1) reg[8] = tex0.sample(s0, in.t0.xy * u.tex_scale[0].xy);
    if (u.tex_mask & 2) reg[9] = tex1.sample(s1, in.t1.xy * u.tex_scale[1].xy);
    if (u.tex_mask & 4) reg[10] = tex2.sample(s2, in.t2.xy * u.tex_scale[2].xy);
    if (u.tex_mask & 8) reg[11] = tex3.sample(s3, in.t3.xy * u.tex_scale[3].xy);
    if (u.tex_mask & 16) reg[8] = cube0.sample(s0, in.t0.xyz);
    if (u.tex_mask & 32) reg[9] = cube1.sample(s1, in.t1.xyz);
    if (u.tex_mask & 64) reg[10] = cube2.sample(s2, in.t2.xyz);
    if (u.tex_mask & 128) reg[11] = cube3.sample(s3, in.t3.xyz);
    reg[12].w = reg[8].w;                                    // r0.a starts as t0.a
    if (u.comb_stages == 0) {
        float4 col = in.d0;
        if (u.tex_mask & 17) col *= reg[8];
        return col;
    }
    return nv_combiners(reg, u);
}

// Clears: one triangle over the whole target; the scissor rectangle limits it.
struct NVClearOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

struct NVClearUniforms {
    float4 color;
    float depth, pad0, pad1, pad2;
};

vertex NVVaryings nv_clear_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    NVVaryings o;
    o.pos = float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
    o.d0 = float4(0); o.d1 = float4(0);
    o.t0 = float4(0); o.t1 = float4(0); o.t2 = float4(0); o.t3 = float4(0);
    o.fog = 0;
    return o;
}

// With a depth buffer: colour and depth.
fragment NVClearOut nv_clear_depth_fs(NVVaryings in [[stage_in]], constant NVClearUniforms &u [[buffer(0)]]) {
    NVClearOut o;
    o.color = u.color;
    o.depth = u.depth;
    return o;
}

// Without a depth buffer: colour only.
fragment float4 nv_clear_fs(NVVaryings in [[stage_in]], constant NVClearUniforms &u [[buffer(0)]]) {
    return u.color;
}
