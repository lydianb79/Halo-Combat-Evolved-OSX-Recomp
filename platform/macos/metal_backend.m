// metal_backend.m — NV2A pixel work on the Mac's GPU.
//
// The NV2A model hands over transformed, clipped and culled triangles with the
// full pixel state (runtime/gpu_backend.h); this file turns them into Metal
// draws:
//   - render targets: one BGRA8 texture per colour-surface address (filled
//     from guest memory when first used) and one Depth32Float texture per
//     depth-surface address; the display shows the colour texture directly;
//   - textures: decoded from guest memory with the runtime's own decoder,
//     uploaded with all mip levels and cached; re-checked (hashed) once per
//     frame so that video frames and other updates are picked up; a texture
//     whose address is a render target is sampled from that render target;
//   - register combiners and alpha test: nv_fs in Shaders.metal;
//   - depth test, blending and colour masks: Metal pipeline/depth state, cached.
// Everything runs on the game thread except metal_backend_surface (main thread).
#import <Metal/Metal.h>
#include <os/lock.h>
#include <stdlib.h>
#include <string.h>

#import "metal_backend.h"
#include "../../runtime/gpu_backend.h"
#include "../../runtime/nv2a.h"
#include "../../runtime/runtime.h"

// ------------------------------------------------------------ GPU layouts
typedef struct {            // NVVertex in Shaders.metal
    float pos[4], d0[4], d1[4], t[4][4];
    float fog[4];           // x: fog coordinate
} GVtx;

typedef struct {            // NVFragUniforms in Shaders.metal (544 bytes)
    float factor0[8][4], factor1[8][4], fog[4], tex_scale[4][4], specfog[2][4];
    uint32_t color_icw[8], color_ocw[8], alpha_icw[8], alpha_ocw[8];
    uint32_t comb_stages, comb_control, final0, final1;
    uint32_t alpha_test, alpha_func, tex_mask, debug_view;
    float alpha_ref, fog_bias, fog_mult, fog_code;
} FragUniforms;
_Static_assert(sizeof(FragUniforms) == 544, "FragUniforms layout");

typedef struct { float color[4]; float depth, pad[3]; } ClearUniforms;

// ------------------------------------------------ specialized combiner shaders
// The generic nv_fs interprets the register combiners per pixel: a register
// array indexed at run time (which the GPU keeps in slow memory) and a branch
// per input mapping. For each distinct combiner setup a straight-line shader
// is generated and compiled in the background (like xemu does); until it is
// ready the generic shader draws. HALO_NO_SPEC_SHADER=1 turns this off.
static NSMutableDictionary<NSNumber *, id> *gSpecFns;          // key -> MTLFunction, or NSNull while compiling / @NO on error
static NSMutableDictionary<NSNumber *, id> *gSpecPipes;        // key+state -> MTLRenderPipelineState, or NSNull while compiling
// GPU vertex programs (NvVsDraw): program hash -> MTLFunction / NSNull / @NO; pipelines by combined key
static NSMutableDictionary<NSNumber *, id> *gVsFns, *gVsPipes;
static unsigned gVsCompiled, gVsFailed, gStatVsDraws;
static os_unfair_lock gSpecLock = OS_UNFAIR_LOCK_INIT;
static BOOL gDisableSpec;
static unsigned gStatSpecDraws, gSpecCompiled, gSpecFailed;

// z: the zero of the operand's type ("0.0" or "float3(0.0)"), so no scalar/vector mixing in max/clamp
static NSString *spec_map(NSString *x, uint32_t m, NSString *z) {
    switch (m & 7) {
    case 0: return [NSString stringWithFormat:@"max(%@, %@)", x, z];
    case 1: return [NSString stringWithFormat:@"(1.0 - saturate(%@))", x];
    case 2: return [NSString stringWithFormat:@"(2.0 * max(%@, %@) - 1.0)", x, z];
    case 3: return [NSString stringWithFormat:@"(-2.0 * max(%@, %@) + 1.0)", x, z];
    case 4: return [NSString stringWithFormat:@"(max(%@, %@) - 0.5)", x, z];
    case 5: return [NSString stringWithFormat:@"(-max(%@, %@) + 0.5)", x, z];
    case 6: return [NSString stringWithFormat:@"(%@)", x];
    default: return [NSString stringWithFormat:@"(-(%@))", x];
    }
}
static NSString *spec_rgb(uint32_t sel) {
    NSString *r = [NSString stringWithFormat:@"R%u.%@", sel & 15, (sel & 0x10) ? @"www" : @"xyz"];
    return spec_map(r, sel >> 5, @"float3(0.0)");
}
static NSString *spec_a(uint32_t sel) {
    NSString *r = [NSString stringWithFormat:@"R%u.%@", sel & 15, (sel & 0x10) ? @"w" : @"z"];
    return spec_map(r, sel >> 5, @"0.0");
}
static NSString *spec_out(NSString *x, uint32_t op, BOOL vec) {
    NSString *e;
    switch (op & 7) {
    case 1: e = [NSString stringWithFormat:@"(%@ - 0.5)", x]; break;
    case 2: e = [NSString stringWithFormat:@"(%@ * 2.0)", x]; break;
    case 3: e = [NSString stringWithFormat:@"((%@ - 0.5) * 2.0)", x]; break;
    case 4: e = [NSString stringWithFormat:@"(%@ * 4.0)", x]; break;
    case 6: e = [NSString stringWithFormat:@"(%@ * 0.5)", x]; break;
    default: e = x; break;
    }
    return vec ? [NSString stringWithFormat:@"clamp(%@, float3(-1.0), float3(1.0))", e]
               : [NSString stringWithFormat:@"clamp(%@, -1.0, 1.0)", e];
}
static NSString *spec_fin(uint32_t sel) {
    NSString *k = [NSString stringWithFormat:@"saturate(R%u.%@)", sel & 15, (sel & 0x10) ? @"www" : @"xyz"];
    return (sel & 0x20) ? [NSString stringWithFormat:@"(1.0 - %@)", k] : k;
}

static inline uint64_t mix64(uint64_t h, uint64_t v) { h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); return h * 0xFF51AFD7ED558CCDull; }
static uint64_t spec_key(const FragUniforms *u, BOOL alphaTest) {
    uint32_t n = u->comb_stages > 8 ? 8 : u->comb_stages;
    uint64_t h = mix64(mix64(mix64(0x5EC0ull, n), u->comb_control & 0x11100), ((uint64_t)u->final0 << 32) | u->final1);
    h = mix64(h, (uint64_t)(alphaTest ? 1 : 0));
    for (uint32_t i = 0; i < n; i++) {
        h = mix64(h, ((uint64_t)u->color_icw[i] << 32) | u->color_ocw[i]);
        h = mix64(h, ((uint64_t)u->alpha_icw[i] << 32) | u->alpha_ocw[i]);
    }
    return h;
}

static NSString *spec_source(const FragUniforms *u, BOOL alphaTest) {
    NSMutableString *s = [NSMutableString string];
    // must match NVVaryings / NVFragUniforms in Shaders.metal
    [s appendString:@"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct NVVaryings { float4 pos [[position, invariant]]; float4 d0, d1; float4 t0, t1, t2, t3; float fog; };\n"
        "struct NVFragUniforms { float4 factor0[8]; float4 factor1[8]; float4 fog; float4 tex_scale[4]; float4 specfog[2];\n"
        "  uint color_icw[8], color_ocw[8], alpha_icw[8], alpha_ocw[8]; uint comb_stages, comb_control, final0, final1;\n"
        "  uint alpha_test, alpha_func, tex_mask, debug_view; float alpha_ref, fog_bias, fog_mult, fog_code; };\n"
        "static float nv_fog(float d, constant NVFragUniforms &u) { int code = int(u.fog_code); if (code & 4) d = abs(d); if (isinf(d)) d = 0.0;\n"
        "  float f; switch (code & 3) { case 1: f = u.fog_bias + d * u.fog_mult - 1.0; break;\n"
        "  case 2: f = u.fog_bias + exp2(d * u.fog_mult * 16.0) - 1.5; break;\n"
        "  default: f = u.fog_bias + exp2(-d * d * u.fog_mult * u.fog_mult * 32.0) - 1.5; break; } return saturate(f); }\n"
        "struct NVFragOut { float4 color [[color(0)]]; uint mask [[sample_mask]]; };\n"
        "static bool nv_compare(uint func, float a, float b) { switch (func & 7) { case 0: return false; case 1: return a < b;\n"
        "  case 2: return a == b; case 3: return a <= b; case 4: return a > b; case 5: return a != b; case 6: return a >= b; default: return true; } }\n"];
    [s appendFormat:@"fragment %@ nv_fs_spec(NVVaryings in [[stage_in]], constant NVFragUniforms &u [[buffer(0)]],\n"
        "  texture2d<float> tex0 [[texture(0)]], texture2d<float> tex1 [[texture(1)]], texture2d<float> tex2 [[texture(2)]], texture2d<float> tex3 [[texture(3)]],\n"
        "  texturecube<float> cube0 [[texture(4)]], texturecube<float> cube1 [[texture(5)]], texturecube<float> cube2 [[texture(6)]], texturecube<float> cube3 [[texture(7)]],\n"
        "  sampler s0 [[sampler(0)]], sampler s1 [[sampler(1)]], sampler s2 [[sampler(2)]], sampler s3 [[sampler(3)]]%@) {\n",
        alphaTest ? @"NVFragOut" : @"float4", alphaTest ? @",\n  device atomic_uint *dbg [[buffer(1)]]" : @""];
    [s appendString:@"  const float4 Z = float4(0.0);\n"
        "  float4 R0 = Z, R1 = Z, R2 = Z, R3 = u.fog, R4 = in.d0, R5 = in.d1, R6 = Z, R7 = Z;\n"
        "  float4 R8 = Z, R9 = Z, R10 = Z, R11 = Z, R12 = Z, R13 = Z, R14 = Z, R15 = Z;\n"
        "  if (u.fog_code != 0.0) R3.w = nv_fog(in.fog, u);\n"
        "  if (u.tex_mask & 1) R8 = tex0.sample(s0, in.t0.xy * u.tex_scale[0].xy);\n"
        "  if (u.tex_mask & 2) R9 = tex1.sample(s1, in.t1.xy * u.tex_scale[1].xy);\n"
        "  if (u.tex_mask & 4) R10 = tex2.sample(s2, in.t2.xy * u.tex_scale[2].xy);\n"
        "  if (u.tex_mask & 8) R11 = tex3.sample(s3, in.t3.xy * u.tex_scale[3].xy);\n"
        "  if (u.tex_mask & 16) R8 = cube0.sample(s0, in.t0.xyz);\n"
        "  if (u.tex_mask & 32) R9 = cube1.sample(s1, in.t1.xyz);\n"
        "  if (u.tex_mask & 64) R10 = cube2.sample(s2, in.t2.xyz);\n"
        "  if (u.tex_mask & 128) R11 = cube3.sample(s3, in.t3.xyz);\n"
        "  R12.w = R8.w;\n"];
    uint32_t n = u->comb_stages > 8 ? 8 : u->comb_stages;
    BOOL msb = ((u->comb_control >> 8) & 1) != 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t icw = u->color_icw[i], ocw = u->color_ocw[i], aicw = u->alpha_icw[i], aocw = u->alpha_ocw[i];
        uint32_t f0 = ((u->comb_control >> 12) & 1) ? i : 0, f1 = ((u->comb_control >> 16) & 1) ? i : 0;
        [s appendFormat:@"  { // stage %u\n    R1 = u.factor0[%u]; R2 = u.factor1[%u];\n", i, f0, f1];
        [s appendFormat:@"    float3 A = %@, B = %@, C = %@, D = %@;\n", spec_rgb(icw >> 24), spec_rgb((icw >> 16) & 255),
                         spec_rgb((icw >> 8) & 255), spec_rgb(icw & 255)];
        [s appendFormat:@"    float3 ab = %@, cd = %@;\n", (ocw & (1u << 13)) ? @"float3(dot(A, B))" : @"A * B",
                         (ocw & (1u << 12)) ? @"float3(dot(C, D))" : @"C * D"];
        [s appendFormat:@"    bool mux = %@;\n", msb ? @"R12.w >= 0.5" : @"(int(R12.w * 255.0) & 1) != 0"];
        [s appendFormat:@"    float3 sum = %@;\n", (ocw & (1u << 14)) ? @"mux ? cd : ab" : @"ab + cd"];
        [s appendFormat:@"    float aA = %@, aB = %@, aC = %@, aD = %@;\n", spec_a(aicw >> 24), spec_a((aicw >> 16) & 255),
                         spec_a((aicw >> 8) & 255), spec_a(aicw & 255)];
        [s appendFormat:@"    float aab = aA * aB, acd = aC * aD, asum = %@;\n", (aocw & (1u << 14)) ? @"mux ? acd : aab" : @"aab + acd"];
        uint32_t op = (ocw >> 15) & 7, aop = (aocw >> 15) & 7;
        uint32_t d_cd = ocw & 15, d_ab = (ocw >> 4) & 15, d_sum = (ocw >> 8) & 15;
        if (d_ab) {
            [s appendFormat:@"    R%u.xyz = %@;\n", d_ab, spec_out(@"ab", op, YES)];
            if (ocw & (1u << 19)) [s appendFormat:@"    R%u.w = R%u.z;\n", d_ab, d_ab];
        }
        if (d_cd) {
            [s appendFormat:@"    R%u.xyz = %@;\n", d_cd, spec_out(@"cd", op, YES)];
            if (ocw & (1u << 18)) [s appendFormat:@"    R%u.w = R%u.z;\n", d_cd, d_cd];
        }
        if (d_sum) [s appendFormat:@"    R%u.xyz = %@;\n", d_sum, spec_out(@"sum", op, YES)];
        uint32_t ad_cd = aocw & 15, ad_ab = (aocw >> 4) & 15, ad_sum = (aocw >> 8) & 15;
        if (ad_ab) [s appendFormat:@"    R%u.w = %@;\n", ad_ab, spec_out(@"aab", aop, NO)];
        if (ad_cd) [s appendFormat:@"    R%u.w = %@;\n", ad_cd, spec_out(@"acd", aop, NO)];
        if (ad_sum) [s appendFormat:@"    R%u.w = %@;\n", ad_sum, spec_out(@"asum", aop, NO)];
        [s appendString:@"  }\n"];
    }
    uint32_t f0 = u->final0, f1 = u->final1, g = (f1 >> 8) & 255;
    [s appendFormat:@"  R1 = u.specfog[0]; R2 = u.specfog[1];\n  float3 vs = R5.xyz + R12.xyz;%@\n  R14 = float4(vs, 0.0);\n",
                     (f1 & 0x80) ? @" vs = saturate(vs);" : @""];
    [s appendFormat:@"  float3 E = %@, F = %@;\n  R15 = float4(E * F, 0.0);\n", spec_fin(f1 >> 24), spec_fin((f1 >> 16) & 255)];
    [s appendFormat:@"  float3 FA = %@, FB = %@, FC = %@, FD = %@;\n", spec_fin(f0 >> 24), spec_fin((f0 >> 16) & 255),
                     spec_fin((f0 >> 8) & 255), spec_fin(f0 & 255)];
    [s appendFormat:@"  float ga = saturate(R%u.%@);%@\n", g & 15, (g & 0x10) ? @"w" : @"z", (g & 0x20) ? @" ga = 1.0 - ga;" : @""];
    [s appendString:@"  float4 col = float4(saturate(FA * FB + (1.0 - FA) * FC + FD), ga);\n"];
    if (alphaTest)
        [s appendString:@"  NVFragOut o; o.color = col; o.mask = 0xFFFFFFFFu;\n"
            "  if (u.alpha_test != 0 && !nv_compare(u.alpha_func, col.w * 255.0, u.alpha_ref)) o.mask = 0;\n"
            "  if (u.debug_view == 10) atomic_fetch_add_explicit(&dbg[0], 1u, memory_order_relaxed);   // see nv_fs: keeps HSR off\n"
            "  return o;\n}\n"];
    else
        [s appendString:@"  return col;\n}\n"];
    return s;
}

static void argb_to_float4(uint32_t c, float *out) {
    out[0] = (float)(c >> 16 & 255) / 255.0f;
    out[1] = (float)(c >> 8 & 255) / 255.0f;
    out[2] = (float)(c & 255) / 255.0f;
    out[3] = (float)(c >> 24) / 255.0f;
}

// ---------------------------------------------------------------- caches
@interface NVTexEntry : NSObject
@property(nonatomic, strong) id<MTLTexture> texture;
@property(nonatomic) uint64_t contentHash;
@property(nonatomic) uint64_t sparseHash;
@property(nonatomic) uint32_t fmt, w, h, pitch, levels, palette, bytes;
@property(nonatomic) uint64_t checkedFrame;
@end
@implementation NVTexEntry
@end

@interface NVMetal : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLTexture>> *colorTargets, *depthTargets;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NVTexEntry *> *textures;
// Render targets: frame of the last draw (under gTargetsLock), and the guest
// memory hash when the target was created (to notice the CPU reusing it).
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSNumber *> *rtLastDraw, *rtGuestHash, *rtBytes, *rtChecked;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *pipelines;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLDepthStencilState>> *depthStates;
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLSamplerState>> *samplers;
@property(nonatomic, strong) id<MTLTexture> dummyTexture, dummyCube;
@property(nonatomic, strong) id<MTLCommandBuffer> frameCmd;
@property(nonatomic, strong) id<MTLRenderCommandEncoder> enc;
@property(nonatomic) uint32_t encColor, encZeta;
@property(nonatomic, strong) id<MTLBuffer> vbuf;
@property(nonatomic) NSUInteger voff;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *freeBuffers;
@property(nonatomic) uint64_t frame;
@property(nonatomic) uint32_t rtLastFlushColor;
@end

static NVMetal *gMetal;

// ---------------------------------------------------------------- diagnosis
// If build/gpu_debug.txt exists next to the app, the renderer cycles through
// test settings, holding each for DBG_HOLD flips, and saves the displayed frame
// of each to build/gpu_debug/<n>_<name>.ppm. Used to find rendering bugs.
// Currently: skips one category of draws per setting to find which one paints over the ring.
enum { DBG_NORMAL, DBG_SKIP_ALPHA_BLENDED, DBG_SKIP_7STAGE, DBG_SKIP_SPRITES_01AE, DBG_SKIP_ADDITIVE, DBG_NORMAL2, DBG_COUNT };
static const char *kDbgNames[DBG_COUNT] = {"normal", "skip_alpha_blended", "skip_7stage_nebula", "skip_sprites_01ae0000",
                                           "skip_additive_302_1", "normal_again"};
#define DBG_HOLD 90
static int gDbgOn, gDbgMode;
// The screen shows the frame the game finished last (at its buffer flip), not
// the buffer named by the display-start register: the game sets that register
// from its vertical-blank interrupt, and when our GPU work runs behind the
// game's CPU side, it names a buffer whose next frame is already being drawn.
static uint32_t gDrawingAddr;      // full-size surface the current frame is drawn into
static uint32_t gCompletedAddr;    // surface of the last complete frame
static uint64_t gDbgFlips;
static NSString *gDbgDir;
// Shadow comparison: for one frame the CPU rasterizer draws too; after every
// draw to the main picture, the GPU result is read back and compared with the
// CPU result in guest memory.
#define SHADOW_FLIP 600
static id<MTLBuffer> gShadowBuf;
static id<MTLBuffer> gStatBuf;     // nv_fs side-effect buffer (see Shaders.metal); statistics for debug_view 10
static BOOL gStatUsed;
static NSString *gShadowDesc;
static BOOL gShadowPending, gShadowStarted;
static int gShadowOps;
static long gShadowPrevDiff;
static uint32_t gShadowTarget;
static int gShadowVariant = -1, gShadowIndex;   // which change is applied to the ring draws (8 combiner stages) in this comparison
static const char *kShadowVariant[4] = {"none", "ring: texture 0 statistics", "ring: alpha forced 1", "ring: alpha = texture 0 alpha before the combiners"};
static FILE *gDbgLog;
#define DBGLOG(...) do { if (gDbgLog) { fprintf(gDbgLog, __VA_ARGS__); fflush(gDbgLog); } } while (0)
static uint64_t hash_bytes(const uint8_t *p, size_t n, uint64_t h);
#define HASH_SEED 0xCBF29CE484222325ull
static os_unfair_lock gTargetsLock = OS_UNFAIR_LOCK_INIT;
// Render resolution: render targets are drawn at gScale times the guest size;
// everything the game sees (coordinates, blits, pixel counts) stays in guest pixels.
static uint32_t gScale = 1;
static volatile uint32_t gPendingScale;          // set from the Settings window, applied at the next flip
// Presentation: at each flip the finished frame is copied (on the game's queue,
// right after its drawing) into one of three present textures; when the copy
// has completed, that texture becomes the one the window shows. The window
// draws on its own queue, so it never waits behind the game's GPU work.
static id<MTLTexture> gPresentTex[3];
static int gPresentReady = -1, gPresentBusy[3];
// GPU load statistics, logged as [gpu] every 150 flips
static uint64_t gGpuBusyNs, gGpuStatT0;
static int gGpuInFlight, gGpuInFlightMax;
static unsigned gGpuPublishSkipped;
static unsigned gStatDraws, gStatFastDraws, gStatPasses, gStatUploads;
// Visibility tests: every render pass gets a visibility-result buffer (one per
// command buffer). While the game counts (SET_ZPASS_PIXEL_COUNT_ENABLE), draws
// count passing samples into a slot per query and encoder; GET_REPORT sums the
// query's slots when the command buffer has completed and writes the report.
#define VIS_SLOTS 2048
static id<MTLBuffer> gVisBuf;
static id<MTLCommandBuffer> gVisCmd;
static NSUInteger gVisNext;
static BOOL gZpass;
static NSMutableArray *gQuery;
static long gEncVisSlot = -1, gEncVisOff = -1;
static unsigned gStatQueries;
static BOOL gDisableFastShader;      // set unless HALO_HSR=1: every draw keeps the side-effect write (no HSR)
#define VBUF_SIZE (8u << 20)

@implementation NVMetal

- (instancetype)initWithDevice:(id<MTLDevice>)device queue:(id<MTLCommandQueue>)queue library:(id<MTLLibrary>)lib {
    if ((self = [super init])) {
        _device = device;
        _queue = queue;
        _library = lib;
        _colorTargets = [NSMutableDictionary dictionary];
        _depthTargets = [NSMutableDictionary dictionary];
        _textures = [NSMutableDictionary dictionary];
        _rtLastDraw = [NSMutableDictionary dictionary];
        _rtGuestHash = [NSMutableDictionary dictionary];
        _rtBytes = [NSMutableDictionary dictionary];
        _rtChecked = [NSMutableDictionary dictionary];
        _pipelines = [NSMutableDictionary dictionary];
        _depthStates = [NSMutableDictionary dictionary];
        _samplers = [NSMutableDictionary dictionary];
        _freeBuffers = [NSMutableArray array];
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                      width:1 height:1 mipmapped:NO];
        td.storageMode = MTLStorageModeShared;
        _dummyTexture = [device newTextureWithDescriptor:td];
        uint32_t white = 0xFFFFFFFFu;
        [_dummyTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&white bytesPerRow:4];
        MTLTextureDescriptor *cd = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                         size:1 mipmapped:NO];
        cd.storageMode = MTLStorageModeShared;
        _dummyCube = [device newTextureWithDescriptor:cd];
        for (NSUInteger f = 0; f < 6; f++)
            [_dummyCube replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:f withBytes:&white bytesPerRow:4 bytesPerImage:4];
    }
    return self;
}

// ------------------------------------------------------- command buffers
- (id<MTLCommandBuffer>)commandBuffer {
    if (!_frameCmd) {
        _frameCmd = [_queue commandBuffer];
        _frameCmd.label = @"NV2A frame";
    }
    return _frameCmd;
}

- (void)endEncoder {
    if (_enc) { [_enc endEncoding]; _enc = nil; }
    gEncVisSlot = gEncVisOff = -1;
    _encColor = _encZeta = 0;
}

- (void)captureTarget:(uint32_t)addr name:(NSString *)name {
    id<MTLTexture> t = _colorTargets[@(addr)];
    if (!t) return;
    NSUInteger w = t.width, h = t.height;
    id<MTLBuffer> buf = [_device newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [[self commandBuffer] blitCommandEncoder];
    [blit copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                 toBuffer:buf destinationOffset:0 destinationBytesPerRow:w * 4 destinationBytesPerImage:w * h * 4];
    [blit endEncoding];
    NSString *path = [gDbgDir stringByAppendingPathComponent:name];
    [[self commandBuffer] addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        (void)cb;
        FILE *f = fopen(path.fileSystemRepresentation, "wb");
        if (!f) return;
        fprintf(f, "P6\n%u %u\n255\n", (unsigned)w, (unsigned)h);
        const uint8_t *p = buf.contents;
        for (NSUInteger i = 0; i < w * h; i++) { uint8_t rgb[3] = {p[i * 4 + 2], p[i * 4 + 1], p[i * 4]}; fwrite(rgb, 1, 3, f); }
        fclose(f);
    }];
}

#define SHADOW_MAX_OPS 2500
// Shadow comparison, one operation at a time (constant memory): after a draw or
// clear to the main picture, its GPU result is read back into one buffer (the
// command buffer is submitted and waited for); at the start of the next
// operation the CPU rasterizer has drawn the same thing into guest memory, and
// the two are compared and logged.
- (BOOL)shadowing:(const NvDrawState *)st {
    if (!g_gpu_cpu_shadow || gScale != 1) return NO;
    if (gShadowOps >= SHADOW_MAX_OPS) { g_gpu_cpu_shadow = 0; return NO; }
    return st->color_addr && st->width >= 640;
}

// Compare the pending GPU readback with what the CPU has drawn by now.
- (void)shadowCompare {
    if (!gShadowPending || !gShadowBuf) return;
    gShadowPending = NO;
    const uint8_t *g = (const uint8_t *)gShadowBuf.contents, *c = g_mem + gShadowTarget;
    long diff = 0;
    for (NSUInteger p = 0; p < 640 * 480; p++) {
        int d0 = abs(g[p * 4] - c[p * 4]), d1 = abs(g[p * 4 + 1] - c[p * 4 + 1]), d2 = abs(g[p * 4 + 2] - c[p * 4 + 2]);
        if (d0 > 40 || d1 > 40 || d2 > 40) diff++;
    }
    DBGLOG("op %3d: %7ld differ (%+ld) | %s\n", gShadowOps, diff, diff - gShadowPrevDiff, gShadowDesc.UTF8String);
    gShadowPrevDiff = diff;
    gShadowOps++;
}

- (void)shadowBefore:(const NvDrawState *)st desc:(NSString *)desc {
    if (![self shadowing:st]) return;
    [self shadowCompare];
    gShadowTarget = st->color_addr;
    gShadowDesc = desc;
}

- (void)shadowAfter:(const NvDrawState *)st {
    if (![self shadowing:st]) return;
    id<MTLTexture> t = _colorTargets[@(st->color_addr)];
    if (!t || t.width < 640 || t.height < 480) return;
    [self endEncoder];
    if (!gShadowBuf) gShadowBuf = [_device newBufferWithLength:640 * 480 * 4 options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [[self commandBuffer] blitCommandEncoder];
    [blit copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(640, 480, 1)
                 toBuffer:gShadowBuf destinationOffset:0 destinationBytesPerRow:640 * 4 destinationBytesPerImage:640 * 480 * 4];
    [blit endEncoding];
    [_frameCmd commit];                                  // the vertex buffer stays with us: the GPU is done with it below
    [_frameCmd waitUntilCompleted];
    _frameCmd = nil;
    gShadowPending = YES;
    if (gStatUsed) {
        gStatUsed = NO;
        const uint32_t *d = gStatBuf.contents;
        DBGLOG("  stats: %u fragments, t0.a<0.5 %u, coords outside 0..1 %u, non-finite coords %u, tex_mask %x, t0 width %u, "
               "level-0 t0.a<0.5 %u, alpha test dropped %u, final alpha<0.5 %u\n",
               d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[8], d[9]);
        for (uint32_t k = 0; k < MIN(d[7], 16u); k++) {
            const float *r = (const float *)(d + 16 + k * 12);
            DBGLOG("    pixel %.1f,%.1f t0 %g %g %g %g | sample %.3f %.3f %.3f a %.3f | level0 a %.3f | final a %.3f\n",
                   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);
        }
        memset(gStatBuf.contents, 0, 4096);
    }
}

- (void)shadowFinish {
    [self shadowCompare];
    DBGLOG("shadow comparison done: %d operations on %08x\n", gShadowOps, gShadowTarget);
    // the final CPU and GPU images
    NSString *dir = gDbgDir;
    void (^save)(const uint8_t *, NSString *) = ^(const uint8_t *px, NSString *name) {
        FILE *f = fopen([dir stringByAppendingPathComponent:name].fileSystemRepresentation, "wb");
        if (!f) return;
        fprintf(f, "P6\n640 480\n255\n");
        for (NSUInteger p = 0; p < 640 * 480; p++) { uint8_t rgb[3] = {px[p * 4 + 2], px[p * 4 + 1], px[p * 4]}; fwrite(rgb, 1, 3, f); }
        fclose(f);
    };
    if (gShadowBuf) save((const uint8_t *)gShadowBuf.contents, [NSString stringWithFormat:@"shadow%d_gpu.ppm", gShadowIndex]);
    save(g_mem + gShadowTarget, [NSString stringWithFormat:@"shadow%d_cpu.ppm", gShadowIndex]);
    // guest memory of the render targets now holds the CPU image: re-baseline their hashes
    for (NSNumber *k in _colorTargets.allKeys)
        if (_rtBytes[k]) _rtGuestHash[k] = @(hash_bytes(g_mem + k.unsignedIntValue, _rtBytes[k].unsignedIntValue, HASH_SEED));
}

- (void)flush {
    static uint32_t lastFlips;
    uint32_t flips = nv2a_stat_flips();                 // the emulated GPU's own frame count
    if (gDbgOn && flips != lastFlips) {
        lastFlips = flips;
        // a frame of the main picture is complete
        gDbgFlips++;
        // six comparisons, 900 frames (30 s at 30 fps) apart, from frame 600 on
        uint64_t rel = gDbgFlips >= SHADOW_FLIP - 1 ? gDbgFlips - (SHADOW_FLIP - 1) : 1;
        if (rel % 900 == 0 && rel / 900 < 6 && !g_gpu_cpu_shadow) {
            gShadowIndex = (int)(rel / 900);
            gShadowVariant = 0;
            gShadowOps = 0; gShadowPrevDiff = 0; gShadowPending = NO;
            DBGLOG("=== comparison %d (frame %llu)\n", gShadowIndex, (unsigned long long)gDbgFlips);
            gShadowStarted = YES;
            g_gpu_cpu_shadow = 1;
        } else if (rel % 900 == 1 && gShadowStarted) {
            gShadowStarted = NO;
            g_gpu_cpu_shadow = 0;
            [self endEncoder];
            [self shadowFinish];
            gShadowVariant = -1;
        }
        if (gDbgFlips % DBG_HOLD == DBG_HOLD - 1) {
            [self endEncoder];
            [self captureTarget:_rtLastFlushColor name:[NSString stringWithFormat:@"%d_%s.ppm", gDbgMode, kDbgNames[gDbgMode]]];
        }
        if (gDbgFlips % DBG_HOLD == 0) gDbgMode = (int)((gDbgFlips / DBG_HOLD) % DBG_COUNT);
    }
    if (_encColor) _rtLastFlushColor = _encColor;
    [self endEncoder];
    if (_frameCmd) {
        id<MTLBuffer> used = _vbuf;
        NSMutableArray *pool = _freeBuffers;
        if (used) {
            [_frameCmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                (void)cb;
                @synchronized(pool) { [pool addObject:used]; }
            }];
        }
        __atomic_add_fetch(&gGpuInFlight, 1, __ATOMIC_RELAXED);
        {
            int n = __atomic_load_n(&gGpuInFlight, __ATOMIC_RELAXED);
            if (n > gGpuInFlightMax) gGpuInFlightMax = n;
        }
        [_frameCmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            double t = cb.GPUEndTime - cb.GPUStartTime;
            if (t > 0 && t < 1.0) __atomic_add_fetch(&gGpuBusyNs, (uint64_t)(t * 1e9), __ATOMIC_RELAXED);
            __atomic_sub_fetch(&gGpuInFlight, 1, __ATOMIC_RELAXED);
        }];
        [_frameCmd commit];
        _frameCmd = nil;
        _vbuf = nil;
    }
    _frame++;
}

// Space for n vertices in the current vertex buffer; returns the buffer and offset.
- (id<MTLBuffer>)vertexSpace:(NSUInteger)bytes offset:(NSUInteger *)offset {
    if (bytes > VBUF_SIZE) return nil;
    if (_vbuf && _voff + bytes > VBUF_SIZE) {
        // full: keep it alive until the GPU is done, then return it to the pool
        id<MTLBuffer> used = _vbuf;
        NSMutableArray *pool = _freeBuffers;
        [[self commandBuffer] addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            (void)cb;
            @synchronized(pool) { [pool addObject:used]; }
        }];
        _vbuf = nil;
    }
    if (!_vbuf) {
        @synchronized(_freeBuffers) {
            _vbuf = _freeBuffers.lastObject;
            if (_vbuf) [_freeBuffers removeLastObject];
        }
        if (!_vbuf) _vbuf = [_device newBufferWithLength:VBUF_SIZE options:MTLResourceStorageModeShared];
        _voff = 0;
    }
    *offset = _voff;
    _voff += (bytes + 255) & ~(NSUInteger)255;
    return _vbuf;
}

// ---------------------------------------------------------- render targets
- (id<MTLTexture>)colorTarget:(const NvDrawState *)st {
    NSNumber *key = @(st->color_addr);
    id<MTLTexture> t = _colorTargets[key];
    if (t && t.width >= st->width * gScale && t.height >= st->height * gScale) return t;
    NSUInteger gw = MAX(st->width, t ? t.width / gScale : 0), gh = MAX(st->height, t ? t.height / gScale : 0);   // guest pixels
    NSUInteger w = gw * gScale, h = gh * gScale;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                  width:w height:h mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    t = [_device newTextureWithDescriptor:td];
    // start from what the guest has in memory there
    [self endEncoder];
    NSUInteger pitch = st->pitch ? st->pitch : gw * 4;
    [self uploadGuest:g_mem + st->color_addr pitch:pitch w:gw h:gh to:t x:0 y:0];
    os_unfair_lock_lock(&gTargetsLock);
    _colorTargets[key] = t;
    os_unfair_lock_unlock(&gTargetsLock);
    _rtGuestHash[key] = @(hash_bytes(g_mem + st->color_addr, pitch * gh, HASH_SEED));
    _rtBytes[key] = @(pitch * gh);
    [_rtChecked removeObjectForKey:key];
    return t;
}

- (id<MTLTexture>)depthTarget:(const NvDrawState *)st size:(id<MTLTexture>)color {
    NSNumber *key = @(st->zeta_addr);
    id<MTLTexture> t = _depthTargets[key];
    if (t && t.width == color.width && t.height == color.height) return t;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                                                  width:color.width height:color.height
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
    t = [_device newTextureWithDescriptor:td];
    _depthTargets[key] = t;
    // new depth buffers start cleared (far)
    [self endEncoder];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.depthAttachment.texture = t;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    rp.stencilAttachment.texture = t;
    rp.stencilAttachment.loadAction = MTLLoadActionClear;
    rp.stencilAttachment.clearStencil = 0;
    rp.stencilAttachment.storeAction = MTLStoreActionStore;
    [[[self commandBuffer] renderCommandEncoderWithDescriptor:rp] endEncoding];
    return t;
}

- (BOOL)beginPass:(const NvDrawState *)st {
    if (!st->color_addr || !st->width || !st->height) return NO;
    uint32_t zeta = st->zeta_fmt ? st->zeta_addr : 0;
    if (_enc && _encColor == st->color_addr && _encZeta == zeta) return YES;
    [self endEncoder];
    id<MTLTexture> color = [self colorTarget:st];
    id<MTLTexture> depth = zeta ? [self depthTarget:st size:color] : nil;
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = color;
    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (depth) {
        rp.depthAttachment.texture = depth;
        rp.depthAttachment.loadAction = MTLLoadActionLoad;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        rp.stencilAttachment.texture = depth;
        rp.stencilAttachment.loadAction = MTLLoadActionLoad;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    id<MTLCommandBuffer> passCmd = [self commandBuffer];
    if (gVisCmd != passCmd) {
        gVisCmd = passCmd;
        gVisBuf = [_device newBufferWithLength:VIS_SLOTS * 8 options:MTLResourceStorageModeShared];
        gVisNext = 0;
    }
    rp.visibilityResultBuffer = gVisBuf;
    _enc = [passCmd renderCommandEncoderWithDescriptor:rp];
    gEncVisSlot = gEncVisOff = -1;
    gStatPasses++;
    _encColor = st->color_addr;
    _encZeta = zeta;
    [_enc setViewport:(MTLViewport){0, 0, (double)color.width, (double)color.height, 0, 1}];
    [_enc setDepthClipMode:MTLDepthClipModeClamp];          // NV2A clamps depth, it does not clip on it
    return YES;
}

// Copy w x h guest pixels (32-bit, `pitch` bytes per row) to (x, y) in guest
// pixels of a render target; at higher render resolutions they are stretched.
- (void)uploadGuest:(const uint8_t *)src pitch:(NSUInteger)pitch w:(NSUInteger)w h:(NSUInteger)h
                 to:(id<MTLTexture>)dst x:(NSUInteger)x y:(NSUInteger)y {
    if (!w || !h) return;
    id<MTLBuffer> buf = [_device newBufferWithBytes:src length:pitch * (h - 1) + w * 4 options:MTLResourceStorageModeShared];
    if (gScale == 1) {
        id<MTLBlitCommandEncoder> b = [[self commandBuffer] blitCommandEncoder];
        [b copyFromBuffer:buf sourceOffset:0 sourceBytesPerRow:pitch sourceBytesPerImage:pitch * h sourceSize:MTLSizeMake(w, h, 1)
                toTexture:dst destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(x, y, 0)];
        [b endEncoding];
        return;
    }
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> tmp = [_device newTextureWithDescriptor:td];
    id<MTLBlitCommandEncoder> b = [[self commandBuffer] blitCommandEncoder];
    [b copyFromBuffer:buf sourceOffset:0 sourceBytesPerRow:pitch sourceBytesPerImage:pitch * h sourceSize:MTLSizeMake(w, h, 1)
            toTexture:tmp destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b endEncoding];
    static id<MTLRenderPipelineState> copyPipe;
    if (!copyPipe) {
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [_library newFunctionWithName:@"fullscreen_vs"];
        pd.fragmentFunction = [_library newFunctionWithName:@"copy_fs"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        NSError *err = nil;
        copyPipe = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!copyPipe) { NSLog(@"HaloRecomp: copy pipeline: %@", err); return; }
    }
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [[self commandBuffer] renderCommandEncoderWithDescriptor:rp];
    [e setViewport:(MTLViewport){(double)(x * gScale), (double)(y * gScale), (double)(w * gScale), (double)(h * gScale), 0, 1}];
    [e setRenderPipelineState:copyPipe];
    [e setFragmentTexture:tmp atIndex:0];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [e endEncoding];
}

// Render resolution changed: drop every render target; they are made again at the new size.
- (void)applyScale:(uint32_t)scale {
    [self endEncoder];
    os_unfair_lock_lock(&gTargetsLock);
    [_colorTargets removeAllObjects];
    [_rtLastDraw removeAllObjects];
    gScale = scale;
    os_unfair_lock_unlock(&gTargetsLock);
    [_depthTargets removeAllObjects];
    [_rtGuestHash removeAllObjects];
    [_rtBytes removeAllObjects];
    [_rtChecked removeAllObjects];
    NSLog(@"HaloRecomp: render resolution %ux", scale);
}

- (void)setScissorX:(uint32_t)x y:(uint32_t)y w:(uint32_t)w h:(uint32_t)h {
    x *= gScale; y *= gScale; w *= gScale; h *= gScale;          // guest pixels -> render target pixels
    id<MTLTexture> t = _colorTargets[@(_encColor)];
    NSUInteger tw = t.width, th = t.height;
    if (x >= tw || y >= th) { x = 0; y = 0; w = 0; h = 0; }
    if (x + w > tw) w = (uint32_t)(tw - x);
    if (y + h > th) h = (uint32_t)(th - y);
    [_enc setScissorRect:(MTLScissorRect){x, y, w ? w : 1, h ? h : 1}];
}

// --------------------------------------------------------- pipeline state
static MTLBlendFactor blend_factor(uint32_t f) {
    switch (f) {
    case 0: return MTLBlendFactorZero;
    case 1: return MTLBlendFactorOne;
    case 0x300: return MTLBlendFactorSourceColor;
    case 0x301: return MTLBlendFactorOneMinusSourceColor;
    case 0x302: return MTLBlendFactorSourceAlpha;
    case 0x303: return MTLBlendFactorOneMinusSourceAlpha;
    case 0x304: return MTLBlendFactorDestinationAlpha;
    case 0x305: return MTLBlendFactorOneMinusDestinationAlpha;
    case 0x306: return MTLBlendFactorDestinationColor;
    case 0x307: return MTLBlendFactorOneMinusDestinationColor;
    case 0x308: return MTLBlendFactorSourceAlphaSaturated;
    case 0x8001: return MTLBlendFactorBlendColor;
    case 0x8002: return MTLBlendFactorOneMinusBlendColor;
    case 0x8003: return MTLBlendFactorBlendAlpha;
    case 0x8004: return MTLBlendFactorOneMinusBlendAlpha;
    default: return MTLBlendFactorOne;
    }
}

static MTLBlendOperation blend_op(uint32_t eq) {
    switch (eq) {
    case 0x800A: return MTLBlendOperationSubtract;
    case 0x800B: return MTLBlendOperationReverseSubtract;
    case 0x8007: return MTLBlendOperationMin;
    case 0x8008: return MTLBlendOperationMax;
    default: return MTLBlendOperationAdd;
    }
}

// colour mask bytes (B, G, R, A in memory order) -> Metal write mask
static MTLColorWriteMask write_mask(uint32_t m) {
    MTLColorWriteMask w = MTLColorWriteMaskNone;
    if (m & 0x000000FF) w |= MTLColorWriteMaskBlue;
    if (m & 0x0000FF00) w |= MTLColorWriteMaskGreen;
    if (m & 0x00FF0000) w |= MTLColorWriteMaskRed;
    if (m & 0xFF000000) w |= MTLColorWriteMaskAlpha;
    return w;
}

- (id<MTLRenderPipelineState>)pipelineClear:(BOOL)clear blend:(BOOL)blend src:(uint32_t)src dst:(uint32_t)dst
                                         eq:(uint32_t)eq mask:(MTLColorWriteMask)mask depth:(BOOL)depth {
    return [self pipelineClear:clear blend:blend src:src dst:dst eq:eq mask:mask depth:depth fast:NO];
}
- (id<MTLRenderPipelineState>)pipelineClear:(BOOL)clear blend:(BOOL)blend src:(uint32_t)src dst:(uint32_t)dst
                                         eq:(uint32_t)eq mask:(MTLColorWriteMask)mask depth:(BOOL)depth fast:(BOOL)fast {
    if (clear) fast = NO;
    uint64_t key = (uint64_t)clear | (uint64_t)blend << 1 | (uint64_t)depth << 2 | (uint64_t)mask << 3 | (uint64_t)fast << 7 |
                   (uint64_t)(eq & 0xFFFF) << 8 | (uint64_t)(src & 0xFFFF) << 24 | (uint64_t)(dst & 0xFFFF) << 40;
    if (!blend) key &= 0xFF;
    NSNumber *k = @(key);
    id<MTLRenderPipelineState> p = _pipelines[k];
    if (p) return p;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [_library newFunctionWithName:clear ? @"nv_clear_vs" : @"nv_vs"];
    pd.fragmentFunction = [_library newFunctionWithName:clear ? (depth ? @"nv_clear_depth_fs" : @"nv_clear_fs")
                                                              : fast ? @"nv_fs_fast" : @"nv_fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.colorAttachments[0].writeMask = mask;
    if (blend) {
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = blend_factor(src);
        pd.colorAttachments[0].sourceAlphaBlendFactor = blend_factor(src);
        pd.colorAttachments[0].destinationRGBBlendFactor = blend_factor(dst);
        pd.colorAttachments[0].destinationAlphaBlendFactor = blend_factor(dst);
        pd.colorAttachments[0].rgbBlendOperation = blend_op(eq);
        pd.colorAttachments[0].alphaBlendOperation = blend_op(eq);
    }
    if (depth) { pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8; pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8; }
    NSError *err = nil;
    p = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p) NSLog(@"HaloRecomp: Metal pipeline error: %@", err);
    else _pipelines[k] = p;
    return p;
}

// Specialized pipeline for this combiner setup, or nil while it is being compiled.
- (id<MTLRenderPipelineState>)specPipeline:(const FragUniforms *)u alphaTest:(BOOL)alphaTest blend:(BOOL)blend
                                       src:(uint32_t)src dst:(uint32_t)dst eq:(uint32_t)eq
                                      mask:(MTLColorWriteMask)mask depth:(BOOL)depth {
    if (gDisableSpec || u->comb_stages == 0) return nil;
    NSNumber *fkey = @(spec_key(u, alphaTest));
    uint64_t bkey = (uint64_t)blend | (uint64_t)depth << 1 | (uint64_t)mask << 2;
    if (blend) bkey |= (uint64_t)(eq & 0xFFFF) << 8 | (uint64_t)(src & 0xFFFF) << 24 | (uint64_t)(dst & 0xFFFF) << 40;
    NSNumber *pkey = @(mix64(fkey.unsignedLongLongValue, bkey));
    os_unfair_lock_lock(&gSpecLock);
    if (!gSpecFns) { gSpecFns = [NSMutableDictionary dictionary]; gSpecPipes = [NSMutableDictionary dictionary]; }
    id p = gSpecPipes[pkey];
    id fn = gSpecFns[fkey];
    if (!fn) gSpecFns[fkey] = [NSNull null];
    BOOL makePipe = !p && fn && [fn conformsToProtocol:@protocol(MTLFunction)];
    if (makePipe) gSpecPipes[pkey] = [NSNull null];
    os_unfair_lock_unlock(&gSpecLock);
    if (p && p != [NSNull null]) return p;
    if (!fn) {                                   // compile the function in the background
        NSString *src = spec_source(u, alphaTest);
        MTLCompileOptions *opts = [MTLCompileOptions new];
        if (@available(macOS 11.0, *)) opts.preserveInvariance = YES;
        [_device newLibraryWithSource:src options:opts completionHandler:^(id<MTLLibrary> lib, NSError *err) {
            id<MTLFunction> f = lib ? [lib newFunctionWithName:@"nv_fs_spec"] : nil;
            os_unfair_lock_lock(&gSpecLock);
            gSpecFns[fkey] = f ? (id)f : (id)@NO;
            if (f) gSpecCompiled++; else gSpecFailed++;
            os_unfair_lock_unlock(&gSpecLock);
            if (!f) {
                static int logged;
                if (logged++ < 3) fprintf(stderr, "[gpu] specialized shader failed: %s\n%s\n",
                                          err.localizedDescription.UTF8String, src.UTF8String);
            }
        }];
        return nil;
    }
    if (makePipe) {
        static id<MTLFunction> vs;
        if (!vs) vs = [_library newFunctionWithName:@"nv_vs"];
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = vs;
        pd.fragmentFunction = fn;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        pd.colorAttachments[0].writeMask = mask;
        if (blend) {
            pd.colorAttachments[0].blendingEnabled = YES;
            pd.colorAttachments[0].sourceRGBBlendFactor = blend_factor(src);
            pd.colorAttachments[0].sourceAlphaBlendFactor = blend_factor(src);
            pd.colorAttachments[0].destinationRGBBlendFactor = blend_factor(dst);
            pd.colorAttachments[0].destinationAlphaBlendFactor = blend_factor(dst);
            pd.colorAttachments[0].rgbBlendOperation = blend_op(eq);
            pd.colorAttachments[0].alphaBlendOperation = blend_op(eq);
        }
        if (depth) { pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8; pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8; }
        [_device newRenderPipelineStateWithDescriptor:pd completionHandler:^(id<MTLRenderPipelineState> ps, NSError *err) {
            (void)err;
            os_unfair_lock_lock(&gSpecLock);
            if (ps) gSpecPipes[pkey] = ps;          // on failure the NSNull stays: the generic shader keeps drawing
            os_unfair_lock_unlock(&gSpecLock);
        }];
    }
    return nil;
}

// ------------------------------------------------- GPU vertex programs
// The NV2A program translated to Metal (runtime/nv2a_raster.c vs_generate),
// compiled in the background; nil until it is ready.
- (id<MTLFunction>)vsFunction:(const NvVsDraw *)d {
    NSNumber *k = @(d->hash);
    os_unfair_lock_lock(&gSpecLock);
    if (!gVsFns) { gVsFns = [NSMutableDictionary dictionary]; gVsPipes = [NSMutableDictionary dictionary]; }
    id fn = gVsFns[k];
    if (!fn) gVsFns[k] = [NSNull null];
    os_unfair_lock_unlock(&gSpecLock);
    if (fn) return [fn conformsToProtocol:@protocol(MTLFunction)] ? fn : nil;
    const char *csrc = d->source ? d->source() : NULL;
    if (!csrc) {
        os_unfair_lock_lock(&gSpecLock); gVsFns[k] = @NO; os_unfair_lock_unlock(&gSpecLock);
        return nil;
    }
    NSString *src = [NSString stringWithUTF8String:csrc];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    if (@available(macOS 11.0, *)) opts.preserveInvariance = YES;
    [_device newLibraryWithSource:src options:opts completionHandler:^(id<MTLLibrary> lib, NSError *err) {
        id<MTLFunction> f = lib ? [lib newFunctionWithName:@"nv_vs_prog"] : nil;
        os_unfair_lock_lock(&gSpecLock);
        gVsFns[k] = f ? (id)f : (id)@NO;
        if (f) gVsCompiled++; else gVsFailed++;
        os_unfair_lock_unlock(&gSpecLock);
        if (!f) {
            static int logged;
            if (logged++ < 3) fprintf(stderr, "[gpu] vertex program failed to compile: %s\n%s\n", err.localizedDescription.UTF8String, src.UTF8String);
        }
    }];
    return nil;
}

// Pipeline for a translated vertex program and a fragment function; nil until it is built.
- (id<MTLRenderPipelineState>)vsPipeline:(id<MTLFunction>)vs vsKey:(uint64_t)vkey fs:(id<MTLFunction>)fs fsKey:(uint64_t)fkey
                                   blend:(BOOL)blend src:(uint32_t)src dst:(uint32_t)dst eq:(uint32_t)eq
                                    mask:(MTLColorWriteMask)mask depth:(BOOL)depth {
    uint64_t bkey = (uint64_t)blend | (uint64_t)depth << 1 | (uint64_t)mask << 2;
    if (blend) bkey |= (uint64_t)(eq & 0xFFFF) << 8 | (uint64_t)(src & 0xFFFF) << 24 | (uint64_t)(dst & 0xFFFF) << 40;
    NSNumber *pkey = @(mix64(mix64(vkey, fkey), bkey));
    os_unfair_lock_lock(&gSpecLock);
    id p = gVsPipes[pkey];
    if (!p) gVsPipes[pkey] = [NSNull null];
    os_unfair_lock_unlock(&gSpecLock);
    if (p) return p != [NSNull null] ? p : nil;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.colorAttachments[0].writeMask = mask;
    if (blend) {
        pd.colorAttachments[0].blendingEnabled = YES;
        pd.colorAttachments[0].sourceRGBBlendFactor = blend_factor(src);
        pd.colorAttachments[0].sourceAlphaBlendFactor = blend_factor(src);
        pd.colorAttachments[0].destinationRGBBlendFactor = blend_factor(dst);
        pd.colorAttachments[0].destinationAlphaBlendFactor = blend_factor(dst);
        pd.colorAttachments[0].rgbBlendOperation = blend_op(eq);
        pd.colorAttachments[0].alphaBlendOperation = blend_op(eq);
    }
    if (depth) { pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8; pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8; }
    [_device newRenderPipelineStateWithDescriptor:pd completionHandler:^(id<MTLRenderPipelineState> ps, NSError *err) {
        if (!ps) {
            static int logged;
            if (logged++ < 3) fprintf(stderr, "[gpu] vertex program pipeline failed: %s\n", err.localizedDescription.UTF8String);
            return;                                   // the NSNull stays: the CPU path keeps drawing
        }
        os_unfair_lock_lock(&gSpecLock);
        gVsPipes[pkey] = ps;
        os_unfair_lock_unlock(&gSpecLock);
    }];
    return nil;
}

- (id<MTLDepthStencilState>)depthTest:(BOOL)test func:(uint32_t)func write:(BOOL)write {
    return [self depthTest:test func:func write:write stencil:NULL];
}
static MTLStencilOperation stencil_op(uint32_t op) {
    switch (op) {
    case 0x0000: return MTLStencilOperationZero;
    case 0x1E01: return MTLStencilOperationReplace;
    case 0x1E02: return MTLStencilOperationIncrementClamp;
    case 0x1E03: return MTLStencilOperationDecrementClamp;
    case 0x150A: return MTLStencilOperationInvert;
    case 0x8507: return MTLStencilOperationIncrementWrap;
    case 0x8508: return MTLStencilOperationDecrementWrap;
    default: return MTLStencilOperationKeep;                          // 0x1E00 KEEP
    }
}
static uint32_t stencil_op_code(uint32_t op) { return (uint32_t)stencil_op(op); }   // 0-7
// st: stencil settings (NULL or disabled: no stencil test)
- (id<MTLDepthStencilState>)depthTest:(BOOL)test func:(uint32_t)func write:(BOOL)write stencil:(const NvDrawState *)st {
    uint64_t key = (test ? 1u : 0u) | (write ? 2u : 0u) | (func & 7) << 2;
    if (st && st->stencil_enable)
        key |= 1ull << 5 | (uint64_t)(st->stencil_func & 7) << 6 | (uint64_t)(st->stencil_mask & 0xFF) << 9 |
               (uint64_t)(st->stencil_func_mask & 0xFF) << 17 | (uint64_t)stencil_op_code(st->stencil_op_fail) << 25 |
               (uint64_t)stencil_op_code(st->stencil_op_zfail) << 28 | (uint64_t)stencil_op_code(st->stencil_op_zpass) << 31;
    NSNumber *k = @(key);
    id<MTLDepthStencilState> s = _depthStates[k];
    if (s) return s;
    static const MTLCompareFunction funcs[8] = {
        MTLCompareFunctionNever, MTLCompareFunctionLess, MTLCompareFunctionEqual, MTLCompareFunctionLessEqual,
        MTLCompareFunctionGreater, MTLCompareFunctionNotEqual, MTLCompareFunctionGreaterEqual, MTLCompareFunctionAlways};
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = test ? funcs[func & 7] : MTLCompareFunctionAlways;
    dd.depthWriteEnabled = write;
    if (st && st->stencil_enable) {
        MTLStencilDescriptor *sd = [MTLStencilDescriptor new];
        sd.stencilCompareFunction = funcs[st->stencil_func & 7];
        sd.stencilFailureOperation = stencil_op(st->stencil_op_fail);
        sd.depthFailureOperation = stencil_op(st->stencil_op_zfail);
        sd.depthStencilPassOperation = stencil_op(st->stencil_op_zpass);
        sd.readMask = st->stencil_func_mask & 0xFF;
        sd.writeMask = st->stencil_mask & 0xFF;
        dd.frontFaceStencil = sd;
        dd.backFaceStencil = sd;
    }
    s = [_device newDepthStencilStateWithDescriptor:dd];
    _depthStates[k] = s;
    return s;
}

- (id<MTLSamplerState>)samplerFor:(const NvTexture *)t {
    uint32_t key = (t->addr_u & 15) | (t->addr_v & 15) << 4 | (t->linear ? 1u : 0u) << 8 | (uint32_t)t->mag_linear << 9 |
                   (uint32_t)t->min_linear << 10 | (uint32_t)(t->mip & 3) << 11 | (uint32_t)(t->max_aniso & 31) << 13 | (uint32_t)(t->cube ? 1 : 0) << 18;
    NSNumber *k = @(key);
    id<MTLSamplerState> s = _samplers[k];
    if (s) return s;
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    MTLSamplerAddressMode (^mode)(uint32_t) = ^MTLSamplerAddressMode(uint32_t m) {
        if (t->linear || t->cube) return MTLSamplerAddressModeClampToEdge;
        return m == 1 ? MTLSamplerAddressModeRepeat : m == 2 ? MTLSamplerAddressModeMirrorRepeat
                                                              : MTLSamplerAddressModeClampToEdge;
    };
    sd.sAddressMode = mode(t->addr_u);
    sd.tAddressMode = mode(t->addr_v);
    sd.magFilter = t->mag_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.minFilter = t->min_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.mipFilter = t->mip == 2 ? MTLSamplerMipFilterLinear : t->mip == 1 ? MTLSamplerMipFilterNearest
                                                                         : MTLSamplerMipFilterNotMipmapped;
    sd.maxAnisotropy = (NSUInteger)MAX(1, MIN(16, t->max_aniso));
    s = [_device newSamplerStateWithDescriptor:sd];
    _samplers[k] = s;
    return s;
}

// ---------------------------------------------------------------- textures
static uint64_t hash_bytes(const uint8_t *p, size_t n, uint64_t h) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        h = (h ^ v) * 0x100000001B3ull;
        h ^= h >> 29;
    }
    for (; i < n; i++) h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

// A cheap fingerprint: the first and last 256 bytes and 16 bytes at 256 spots.
static uint64_t hash_sparse(const uint8_t *p, size_t n) {
    if (n <= 8192) return hash_bytes(p, n, HASH_SEED);
    uint64_t h = hash_bytes(p, 256, HASH_SEED);
    h = hash_bytes(p + n - 256, 256, h);
    size_t step = (n - 16) / 256;
    for (size_t i = 0; i < 256; i++) h = hash_bytes(p + i * step, 16, h);
    return h;
}

- (id<MTLTexture>)textureFor:(const NvTexture *)t {
    // Render-to-texture: sample the GPU render target directly, if the game has
    // drawn into it and the CPU has not written new data there since.
    NSNumber *rtKey = @(t->addr);
    id<MTLTexture> rt = _colorTargets[rtKey];
    if (rt && !t->cube) {
        os_unfair_lock_lock(&gTargetsLock);
        BOOL drawn = _rtLastDraw[rtKey] != nil;
        os_unfair_lock_unlock(&gTargetsLock);
        if (drawn && !g_gpu_cpu_shadow && _rtChecked[rtKey].unsignedLongLongValue != _frame + 1) {
            uint64_t gh = hash_bytes(g_mem + t->addr, _rtBytes[rtKey].unsignedIntValue, HASH_SEED);
            _rtChecked[rtKey] = @(_frame + 1);
            if (gh != _rtGuestHash[rtKey].unsignedLongLongValue) drawn = NO;    // the CPU reused this memory
        }
        if (drawn) return rt;
        if (_rtLastDraw[rtKey] != nil || _rtChecked[rtKey] != nil) {
            // guest memory is authoritative again: forget the GPU copy
            { static int n; if (n++ < 100) NSLog(@"HaloRecomp: render target %08x sampled after the CPU changed it: using guest memory", t->addr); }
            if (_encColor == t->addr) [self endEncoder];
            os_unfair_lock_lock(&gTargetsLock);
            [_colorTargets removeObjectForKey:rtKey];
            [_rtLastDraw removeObjectForKey:rtKey];
            os_unfair_lock_unlock(&gTargetsLock);
            [_rtGuestHash removeObjectForKey:rtKey];
            [_rtBytes removeObjectForKey:rtKey];
            [_rtChecked removeObjectForKey:rtKey];
        }
    }
    if (!t->addr || t->addr + (uint64_t)t->bytes > GUEST_RAM_MIRROR1 + GUEST_RAM_SIZE) return t->cube ? _dummyCube : _dummyTexture;
    NSNumber *key = @(t->addr);
    NVTexEntry *e = _textures[key];
    BOOL same = e && e.fmt == t->fmt && e.w == t->w && e.h == t->h && e.pitch == t->pitch &&
                e.levels == (uint32_t)t->levels && e.palette == t->palette_addr && e.bytes == t->bytes &&
                e.texture.textureType == (t->cube ? MTLTextureTypeCube : MTLTextureType2D);
    if (same && e.checkedFrame == _frame) return e.texture;
    // Has the game changed the texture's memory? Hashing every texture in full
    // every frame costs gigabytes per second in a level, so each frame checks
    // a sparse sample (start, end and 256 spots in between) and only every 64th
    // frame (staggered per texture) the whole thing.
    BOOL full = !same || ((_frame + (t->addr >> 12)) & 63) == 0;
    uint64_t h = full ? hash_bytes(g_mem + t->addr, t->bytes, HASH_SEED) : hash_sparse(g_mem + t->addr, t->bytes);
    if (t->palette_addr) h = hash_bytes(g_mem + t->palette_addr, 1024, h);
    if (same && (full ? e.contentHash : e.sparseHash) == h) { e.checkedFrame = _frame; return e.texture; }
    if (same && !full) {                                   // sample changed: confirm with the full hash
        uint64_t hf = hash_bytes(g_mem + t->addr, t->bytes, HASH_SEED);
        if (t->palette_addr) hf = hash_bytes(g_mem + t->palette_addr, 1024, hf);
        if (hf == e.contentHash) { e.sparseHash = h; e.checkedFrame = _frame; return e.texture; }
    }
    if (!full) {                                           // recompute both for the new contents
        h = hash_bytes(g_mem + t->addr, t->bytes, HASH_SEED);
        if (t->palette_addr) h = hash_bytes(g_mem + t->palette_addr, 1024, h);
    }
    uint64_t sparse = hash_sparse(g_mem + t->addr, t->bytes);
    if (t->palette_addr) sparse = hash_bytes(g_mem + t->palette_addr, 1024, sparse);

    // (re)upload: always into a new texture, the old one may still be in use by the GPU
    int levels = t->linear ? 1 : MAX(1, t->levels);
    MTLTextureDescriptor *td = t->cube
        ? [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm size:t->w mipmapped:levels > 1]
        : [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:t->w height:t->h
                                                         mipmapped:levels > 1];
    td.mipmapLevelCount = (NSUInteger)levels;
    td.usage = MTLTextureUsageShaderRead;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    td.storageMode = _device.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;   // Managed: Intel Macs
#pragma clang diagnostic pop
    id<MTLTexture> tex = [_device newTextureWithDescriptor:td];
    gStatUploads++;
    uint32_t *buf = malloc((size_t)t->w * t->h * 4);
    if (!tex || !buf) { free(buf); return t->cube ? _dummyCube : _dummyTexture; }
    for (int f = 0; f < (t->cube ? 6 : 1); f++)
        for (int l = 0; l < levels; l++) {
            uint32_t w = 0, hgt = 0;
            nv2a_decode_texture_level(t, f, l, buf, &w, &hgt);
            if (gDbgLog && l == 0 && f == 0) {
                uint32_t transparent = 0;
                for (uint32_t i = 0; i < w * hgt; i++) transparent += (buf[i] >> 24) < 128;
                DBGLOG("frame %llu upload %08x fmt %02x %ux%u levels %d cube %d bytes %u hash %016llx: %.1f%% texels alpha<0.5%s\n",
                       (unsigned long long)_frame, t->addr, t->fmt, w, hgt, levels, t->cube, t->bytes, (unsigned long long)h,
                       100.0 * transparent / (w * hgt), e ? " (re-upload)" : "");
            }
            [tex replaceRegion:MTLRegionMake2D(0, 0, w, hgt) mipmapLevel:(NSUInteger)l slice:(NSUInteger)f
                     withBytes:buf bytesPerRow:w * 4 bytesPerImage:w * hgt * 4];
        }
    free(buf);
    if (!e) { e = [NVTexEntry new]; _textures[key] = e; }
    e.texture = tex;
    e.contentHash = h; e.sparseHash = sparse; e.fmt = t->fmt; e.w = t->w; e.h = t->h; e.pitch = t->pitch;
    e.levels = (uint32_t)t->levels; e.palette = t->palette_addr; e.bytes = t->bytes;
    e.checkedFrame = _frame;
    return tex;
}

// ------------------------------------------------------------------- draws
- (void)draw:(const NvDrawState *)st vertices:(const OutVtx *)v count:(int)count {
    if (0) {
        static int idx;
        DBGLOG("draw %d flip %llu -> %08x n %d | comb %u final %08x %08x | atest %u | blend %u %x %x | depth %u %x %u | cmask %08x | "
               "tex %08x/%02x %08x/%02x %08x/%02x %08x/%02x | v0 %.1f %.1f %.1f w %.4f\n",
               idx++, (unsigned long long)gDbgFlips, st->color_addr, count, st->comb_control & 0xFF, st->final0, st->final1,
               st->alpha_test_enable, st->blend_enable, st->blend_src, st->blend_dst, st->depth_test, st->depth_func,
               st->depth_mask, st->color_mask,
               st->tex[0].enabled ? st->tex[0].addr : 0, st->tex[0].fmt, st->tex[1].enabled ? st->tex[1].addr : 0, st->tex[1].fmt,
               st->tex[2].enabled ? st->tex[2].addr : 0, st->tex[2].fmt, st->tex[3].enabled ? st->tex[3].addr : 0, st->tex[3].fmt,
               v[0].x, v[0].y, v[0].z, v[0].w);
    }
    if (gDbgOn) {
        BOOL alphaBlend = st->blend_enable && st->blend_src == 0x302 && st->blend_dst == 0x303;
        (void)alphaBlend;

    }
    if (count <= 0) return;
    if ([self shadowing:st]) [self shadowBefore:st desc:[NSString stringWithFormat:@"draw n %d comb %u final %08x %08x atest %u blend %u %x %x depth %u %x %u cmask %08x tex %08x/%02x %08x/%02x %08x/%02x %08x/%02x",
        count, st->comb_control & 0xFF, st->final0, st->final1, st->alpha_test_enable, st->blend_enable, st->blend_src, st->blend_dst,
        st->depth_test, st->depth_func, st->depth_mask, st->color_mask, st->tex[0].enabled ? st->tex[0].addr : 0, st->tex[0].fmt,
        st->tex[1].enabled ? st->tex[1].addr : 0, st->tex[1].fmt, st->tex[2].enabled ? st->tex[2].addr : 0, st->tex[2].fmt,
        st->tex[3].enabled ? st->tex[3].addr : 0, st->tex[3].fmt]];
    [self drawInner:st vertices:v count:count];
    [self shadowAfter:st];
}

- (void)drawInner:(const NvDrawState *)st vertices:(const OutVtx *)v count:(int)count {
    [self drawInner:st vertices:v count:count vs:NULL];
}

// With vsd, the vertex program runs on the GPU; returns NO (drawing nothing)
// while its shader or pipeline is not ready, and the caller draws on the CPU path.
- (BOOL)drawInner:(const NvDrawState *)st vertices:(const OutVtx *)v count:(int)count vs:(const NvVsDraw *)vsd {
    id<MTLFunction> vsf = nil;
    if (vsd) {
        vsf = [self vsFunction:vsd];
        if (!vsf) return NO;
        if (vsd->cull_enable && vsd->cull_face == 0x408) return YES;     // both faces culled: nothing to draw
    }
    if (![self beginPass:st]) return vsd ? YES : NO;
    // textures first: an upload may need to end the pass (render target creation)
    id<MTLTexture> texs[4], cubes[4];
    id<MTLSamplerState> samps[4];
    FragUniforms u;
    memset(&u, 0, sizeof u);
    for (int i = 0; i < 4; i++) {
        NvTexture tdbg = st->tex[i];
        BOOL ringDraw = g_gpu_cpu_shadow && (st->comb_control & 0xFF) == 8;
        (void)ringDraw;
        const NvTexture *t = &tdbg;
        cubes[i] = _dummyCube;
        if (t->enabled && t->cube) {
            cubes[i] = [self textureFor:t];
            texs[i] = _dummyTexture;
            samps[i] = [self samplerFor:t];
            u.tex_mask |= 16u << i;                       // bits 4-7: cube map on this stage
        } else if (t->enabled) {
            texs[i] = [self textureFor:t];
            samps[i] = [self samplerFor:t];
            u.tex_mask |= 1u << i;
            u.tex_scale[i][0] = t->linear ? 1.0f / (float)t->w : 1.0f;
            u.tex_scale[i][1] = t->linear ? 1.0f / (float)t->h : 1.0f;
        } else {
            texs[i] = _dummyTexture;
            samps[i] = nil;
        }
    }
    if (![self beginPass:st]) return YES;
    os_unfair_lock_lock(&gTargetsLock);
    _rtLastDraw[@(st->color_addr)] = @(_frame);
    if (st->width >= 320) gDrawingAddr = st->color_addr;
    os_unfair_lock_unlock(&gTargetsLock);

    // vertices: screen space -> clip space (x w, so Metal interpolates with perspective)
    float W = (float)(_colorTargets[@(st->color_addr)].width / gScale), H = (float)(_colorTargets[@(st->color_addr)].height / gScale);
    float zmax = st->zeta_fmt == 1 ? 65535.0f : 16777215.0f;
    NSUInteger off = 0, ioff = 0, bytes = vsd ? (NSUInteger)vsd->nverts * 16 * 16 : (NSUInteger)count * sizeof(GVtx);
    id<MTLBuffer> vb = [self vertexSpace:bytes offset:&off], ib = nil;
    if (!vb) return vsd ? NO : YES;
    if (vsd) {
        memcpy((uint8_t *)vb.contents + off, vsd->inputs, bytes);
        ib = [self vertexSpace:(NSUInteger)vsd->nindices * 4 offset:&ioff];
        if (!ib) return NO;

        memcpy((uint8_t *)ib.contents + ioff, vsd->indices, (NSUInteger)vsd->nindices * 4);
        count = 0;
    }
    GVtx *g = (GVtx *)((uint8_t *)vb.contents + off);
    for (int i = 0; i < count; i++) {
        const OutVtx *s = &v[i];
        // Homogeneous position: the CPU does not clip for the GPU path, so w may be
        // negative (behind the camera); the GPU clips against w > 0. Depth is not
        // clamped per vertex; the pass clamps it per pixel (like the CPU renderer).
        float w = s->w;
        float z = s->z / zmax;
        g[i].pos[0] = (s->x / W * 2.0f - 1.0f) * w;
        g[i].pos[1] = (1.0f - s->y / H * 2.0f) * w;
        g[i].pos[2] = z * w;
        g[i].pos[3] = w;
        memcpy(g[i].d0, &s->d0, 16);
        memcpy(g[i].d1, &s->d1, 16);
        memcpy(g[i].t, s->t, 64);
        g[i].fog[0] = s->fog; g[i].fog[1] = g[i].fog[2] = g[i].fog[3] = 0;
    }

    // pixel state
    u.comb_stages = st->comb_control & 0xFF;
    u.comb_control = st->comb_control;
    u.final0 = st->final0;
    u.final1 = st->final1;
    for (int i = 0; i < 8; i++) {
        argb_to_float4(st->factor0[i], u.factor0[i]);
        argb_to_float4(st->factor1[i], u.factor1[i]);
    }
    argb_to_float4(st->fog_color, u.fog);
    argb_to_float4(st->specfog[0], u.specfog[0]);
    argb_to_float4(st->specfog[1], u.specfog[1]);
    memcpy(u.color_icw, st->color_icw, sizeof u.color_icw);
    memcpy(u.color_ocw, st->color_ocw, sizeof u.color_ocw);
    memcpy(u.alpha_icw, st->alpha_icw, sizeof u.alpha_icw);
    memcpy(u.alpha_ocw, st->alpha_ocw, sizeof u.alpha_ocw);
    u.alpha_test = st->alpha_test_enable ? 1 : 0;
    BOOL ringVariant = g_gpu_cpu_shadow && (st->comb_control & 0xFF) == 8;
    if (ringVariant && gShadowVariant >= 1) u.debug_view = gShadowVariant == 1 ? 10 : gShadowVariant == 2 ? 8 : 7;   // Shaders.metal
    if (!gStatBuf) {
        gStatBuf = [_device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        memset(gStatBuf.contents, 0, 4096);
    }
    if (u.debug_view == 10) gStatUsed = YES;
    if (gDbgLog && st->alpha_test_enable) {
        static int n;
        if (n++ < 60)
            DBGLOG("frame %llu alpha-tested draw -> %08x: func %x ref %u comb %u final1 %08x | t0 %08x fmt %02x %ux%u (gpu %lux%lu, %lu levels, %s) | t1 %08x t2 %08x t3 %08x cube %d\n",
                   (unsigned long long)_frame, st->color_addr, st->alpha_func, st->alpha_ref, st->comb_control & 0xFF, st->final1,
                   st->tex[0].addr, st->tex[0].fmt, st->tex[0].w, st->tex[0].h, (unsigned long)texs[0].width,
                   (unsigned long)texs[0].height, (unsigned long)texs[0].mipmapLevelCount,
                   texs[0] == _dummyTexture ? "dummy" : _colorTargets[@(st->tex[0].addr)] == texs[0] ? "render target" : "guest texture",
                   st->tex[1].addr, st->tex[2].addr, st->tex[3].addr, st->tex[3].cube);
    }
    u.alpha_func = st->alpha_func;
    u.alpha_ref = (float)(st->alpha_ref & 255);
    u.fog_code = (float)st->fog_code;
    u.fog_bias = st->fog_bias;
    u.fog_mult = st->fog_mult;

    BOOL hasDepth = _encZeta != 0;
    // no alpha test and no diagnosis: the lean shader, which keeps the GPU's hidden-surface removal
    BOOL fast = !u.alpha_test && u.debug_view == 0 && !g_gpu_cpu_shadow && !gDisableFastShader;
    id<MTLRenderPipelineState> p = nil;
    if (u.debug_view == 0 && !g_gpu_cpu_shadow)
        p = [self specPipeline:&u alphaTest:u.alpha_test != 0 || gDisableFastShader blend:st->blend_enable != 0 src:st->blend_src dst:st->blend_dst
                            eq:st->blend_eq mask:write_mask(st->color_mask) depth:hasDepth];
    if (vsd) {
        // same fragment choice as below, combined with the translated vertex program
        static id<MTLFunction> fsGeneric, fsFast;
        if (!fsGeneric) { fsGeneric = [_library newFunctionWithName:@"nv_fs"]; fsFast = [_library newFunctionWithName:@"nv_fs_fast"]; }
        id<MTLFunction> fs = fast ? fsFast : fsGeneric;
        uint64_t fkey = fast ? 1 : 2;
        if (p) {                                          // a specialized fragment function is ready
            BOOL at = u.alpha_test != 0 || gDisableFastShader;
            NSNumber *k = @(spec_key(&u, at));
            os_unfair_lock_lock(&gSpecLock);
            id f = gSpecFns[k];
            os_unfair_lock_unlock(&gSpecLock);
            if ([f conformsToProtocol:@protocol(MTLFunction)]) { fs = f; fkey = k.unsignedLongLongValue; }
        }
        p = [self vsPipeline:vsf vsKey:vsd->hash fs:fs fsKey:fkey blend:st->blend_enable != 0 src:st->blend_src dst:st->blend_dst
                          eq:st->blend_eq mask:write_mask(st->color_mask) depth:hasDepth];
        if (!p) return NO;
        gStatVsDraws++;
        if (fkey > 2) gStatSpecDraws++;
    } else if (p) gStatSpecDraws++;
    else p = [self pipelineClear:NO blend:st->blend_enable != 0 src:st->blend_src dst:st->blend_dst
                              eq:st->blend_eq mask:write_mask(st->color_mask) depth:hasDepth fast:fast];
    gStatDraws++;
    if (fast) gStatFastDraws++;
    if (!p) return YES;
    [_enc setRenderPipelineState:p];
    if (hasDepth) {
        BOOL dbgNoDepth = NO;
        [_enc setDepthStencilState:[self depthTest:st->depth_test != 0 && !dbgNoDepth func:st->depth_func
                                             write:st->depth_test && st->depth_mask && !dbgNoDepth stencil:st]];
        if (st->stencil_enable) [_enc setStencilReferenceValue:st->stencil_ref & 0xFF];
    }
    if (st->blend_enable) {
        float bc[4];
        argb_to_float4(st->blend_color, bc);
        [_enc setBlendColorRed:bc[0] green:bc[1] blue:bc[2] alpha:bc[3]];
    }
    if (vsd && vsd->cull_enable) {
        // the CPU path's rule (gpu_triangle): clockwise on screen is front when front_face is CW (0x900)
        static int invert = -1;
        if (invert < 0) invert = getenv("HALO_CULL_INVERT") ? 1 : 0;
        BOOL cwFront = (vsd->front_face == 0x900) ^ (invert != 0);
        [_enc setFrontFacingWinding:cwFront ? MTLWindingClockwise : MTLWindingCounterClockwise];
        [_enc setCullMode:vsd->cull_face == 0x404 ? MTLCullModeFront : MTLCullModeBack];
    } else [_enc setCullMode:MTLCullModeNone];               // the CPU path culled already
    {   // polygon offset (decals drawn on top of walls and floors). The NV2A bias is in
        // depth-buffer units, like Metal's for a float depth buffer near the far end.
        float bias = st->poly_offset_fill ? st->poly_bias : 0.0f, scale = st->poly_offset_fill ? st->poly_scale : 0.0f;
        if (st->zeta_fmt == 1) bias *= 256.0f;                // Z16: one unit is 1/65535, 256 float units near 1.0
        static int logged;
        if (st->poly_offset_fill && logged < 4) { logged++; fprintf(stderr, "[gpu] polygon offset: scale %g bias %g\n", scale, bias); }
        [_enc setDepthBias:bias slopeScale:scale clamp:0.0f];
    }
    [self setScissorX:st->clip_x y:st->clip_y w:st->clip_w h:st->clip_h];
    [_enc setVertexBuffer:vb offset:off atIndex:0];
    if (vsd) {
        float vu[4] = {W, H, zmax, 0};
        [_enc setVertexBytes:vsd->consts length:192 * 16 atIndex:1];
        [_enc setVertexBytes:vu length:sizeof vu atIndex:2];
    }
    [_enc setFragmentBytes:&u length:sizeof u atIndex:0];
    [_enc setFragmentBuffer:gStatBuf offset:0 atIndex:1];
    for (int i = 0; i < 4; i++) {
        [_enc setFragmentTexture:texs[i] atIndex:(NSUInteger)i];
        [_enc setFragmentTexture:cubes[i] atIndex:(NSUInteger)(4 + i)];
        if (!samps[i]) samps[i] = [self samplerFor:&st->tex[i]];
        [_enc setFragmentSamplerState:samps[i] atIndex:(NSUInteger)i];
    }
    [self applyVisibility];
    if (vsd)
        [_enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:(NSUInteger)vsd->nindices indexType:MTLIndexTypeUInt32
                        indexBuffer:ib indexBufferOffset:ioff];
    else [_enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:(NSUInteger)count];
    return YES;
}

// Visibility counting for the next draw on the current encoder.
- (void)applyVisibility {
    if (gZpass && gVisBuf) {
        if (gEncVisSlot < 0 && gVisNext < VIS_SLOTS) {
            gEncVisSlot = (long)gVisNext++;
            ((uint64_t *)gVisBuf.contents)[gEncVisSlot] = 0;
            if (!gQuery) gQuery = [NSMutableArray array];
            [gQuery addObject:@[gVisBuf, @(gEncVisSlot)]];
        }
        if (gEncVisSlot >= 0 && gEncVisOff != gEncVisSlot * 8) {
            [_enc setVisibilityResultMode:MTLVisibilityResultModeCounting offset:(NSUInteger)gEncVisSlot * 8];
            gEncVisOff = gEncVisSlot * 8;
        }
    } else if (gEncVisOff >= 0) {
        [_enc setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        gEncVisOff = -1;
    }
}

- (void)report:(int)op arg:(uint32_t)arg {
    if (op == 0) { gZpass = arg != 0; return; }
    if (op == 1) { gQuery = [NSMutableArray array]; gEncVisSlot = -1; return; }
    NSArray *slots = gQuery ? [gQuery copy] : @[];
    uint32_t dst = arg, area = gScale * gScale;            // counts in guest pixels
    gStatQueries++;
    [[self commandBuffer] addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        (void)cb;
        uint64_t n = 0;
        for (NSArray *e in slots) n += ((const uint64_t *)((id<MTLBuffer>)e[0]).contents)[[e[1] unsignedIntegerValue]];
        n /= area;
        uint32_t *p = (uint32_t *)(g_mem + dst);
        __atomic_store_n(&p[2], n > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)n, __ATOMIC_RELAXED);
        __atomic_store_n(&p[3], 0u, __ATOMIC_RELEASE);          // status: done
    }];
}

- (BOOL)drawVS:(const NvDrawState *)st vs:(const NvVsDraw *)d {
    if (g_gpu_cpu_shadow) return NO;
    return [self drawInner:st vertices:NULL count:0 vs:d];
}

- (void)clear:(const NvDrawState *)st flags:(uint32_t)flags color:(uint32_t)color zs:(uint32_t)zs
           x0:(uint32_t)x0 y0:(uint32_t)y0 x1:(uint32_t)x1 y1:(uint32_t)y1 {
    if (0)
        DBGLOG("clear flip %llu -> %08x zeta %08x fmt %u flags %x color %08x zs %08x rect %u,%u-%u,%u\n", (unsigned long long)gDbgFlips,
               st->color_addr, st->zeta_addr, st->zeta_fmt, flags, color, zs, x0, y0, x1, y1);
    if ([self shadowing:st]) [self shadowBefore:st desc:[NSString stringWithFormat:@"clear flags %x color %08x", flags, color]];
    [self clearInner:st flags:flags color:color zs:zs x0:x0 y0:y0 x1:x1 y1:y1];
    [self shadowAfter:st];
}

- (void)clearInner:(const NvDrawState *)st flags:(uint32_t)flags color:(uint32_t)color zs:(uint32_t)zs
                x0:(uint32_t)x0 y0:(uint32_t)y0 x1:(uint32_t)x1 y1:(uint32_t)y1 {
    if (![self beginPass:st]) return;
    BOOL hasDepth = _encZeta != 0;
    MTLColorWriteMask mask = MTLColorWriteMaskNone;
    if (flags & 0x10) mask |= MTLColorWriteMaskRed;
    if (flags & 0x20) mask |= MTLColorWriteMaskGreen;
    if (flags & 0x40) mask |= MTLColorWriteMaskBlue;
    if (flags & 0x80) mask |= MTLColorWriteMaskAlpha;
    BOOL clearDepth = hasDepth && (flags & 1);
    BOOL clearStencil = hasDepth && (flags & 2) && st->zeta_fmt == 2;
    if (mask == MTLColorWriteMaskNone && !clearDepth && !clearStencil) return;
    ClearUniforms cu;
    memset(&cu, 0, sizeof cu);
    argb_to_float4(color, cu.color);
    cu.depth = st->zeta_fmt == 1 ? (float)(zs & 0xFFFF) / 65535.0f : (float)(zs >> 8) / 16777215.0f;
    id<MTLRenderPipelineState> p = [self pipelineClear:YES blend:NO src:0 dst:0 eq:0 mask:mask depth:hasDepth];
    if (!p) return;
    [_enc setRenderPipelineState:p];
    if (hasDepth) {
        NvDrawState sst;                                         // stencil: always replace with the clear value
        memset(&sst, 0, sizeof sst);
        sst.stencil_enable = clearStencil;
        sst.stencil_func = 0x207; sst.stencil_mask = 0xFF; sst.stencil_func_mask = 0xFF;
        sst.stencil_op_fail = sst.stencil_op_zfail = sst.stencil_op_zpass = 0x1E01;
        [_enc setDepthStencilState:[self depthTest:NO func:7 write:clearDepth stencil:&sst]];
        if (clearStencil) [_enc setStencilReferenceValue:zs & 0xFF];
    }
    [self setScissorX:x0 y:y0 w:x1 - x0 + 1 h:y1 - y0 + 1];
    [_enc setFragmentBytes:&cu length:sizeof cu atIndex:0];
    [_enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}

// The game copied a rectangle between surfaces (NV09F image blit), typically
// the frame it just drew into a texture for a screen effect. Guest memory has
// been copied already; mirror the copy on the GPU, where the real pixels are.
- (void)blitFrom:(uint32_t)src pitch:(uint32_t)sp to:(uint32_t)dst pitch:(uint32_t)dp bpp:(uint32_t)bpp
              sx:(uint32_t)sx sy:(uint32_t)sy dx:(uint32_t)dx dy:(uint32_t)dy w:(uint32_t)w h:(uint32_t)h {
    os_unfair_lock_lock(&gTargetsLock);
    id<MTLTexture> s = _rtLastDraw[@(src)] ? _colorTargets[@(src)] : nil;
    id<MTLTexture> d = _colorTargets[@(dst)];
    os_unfair_lock_unlock(&gTargetsLock);
    if (bpp != 4) return;                                   // render targets here are 32-bit
    if (!s && !d) return;                                   // both only in guest memory: nothing to do
    [self endEncoder];
    if (!d || d.width < dp / 4 * gScale || d.height < (dy + h) * gScale) {
        // the destination becomes a GPU surface the game can sample
        NvDrawState st;
        memset(&st, 0, sizeof st);
        st.color_addr = dst;
        st.width = dp / 4;
        st.height = MAX(dy + h, d ? (uint32_t)(d.height / gScale) : 0);
        st.pitch = dp;
        d = [self colorTarget:&st];
    }
    uint32_t dw = (uint32_t)(d.width / gScale), dh = (uint32_t)(d.height / gScale);     // guest pixels
    w = MIN(w, dw > dx ? dw - dx : 0);
    h = MIN(h, dh > dy ? dh - dy : 0);
    if (s) {
        uint32_t sw = (uint32_t)(s.width / gScale), sh = (uint32_t)(s.height / gScale);
        w = MIN(w, sw > sx ? sw - sx : 0);
        h = MIN(h, sh > sy ? sh - sy : 0);
    }
    if (!w || !h) return;
    if (s) {
        id<MTLBlitCommandEncoder> b = [[self commandBuffer] blitCommandEncoder];
        [b copyFromTexture:s sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(sx * gScale, sy * gScale, 0)
                sourceSize:MTLSizeMake(w * gScale, h * gScale, 1)
                 toTexture:d destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(dx * gScale, dy * gScale, 0)];
        [b endEncoding];
    } else {
        // source only in guest memory: upload the rectangle into the GPU surface
        [self uploadGuest:g_mem + src + sy * sp + sx * 4 pitch:sp w:w h:h to:d x:dx y:dy];
    }
    NSNumber *key = @(dst);
    os_unfair_lock_lock(&gTargetsLock);
    _rtLastDraw[key] = @(_frame);
    os_unfair_lock_unlock(&gTargetsLock);
    // guest memory was copied too: it is the new baseline for "has the CPU changed it?"
    if (_rtBytes[key]) _rtGuestHash[key] = @(hash_bytes(g_mem + dst, _rtBytes[key].unsignedIntValue, HASH_SEED));
    [_rtChecked removeObjectForKey:key];
}


@end

// ------------------------------------------------------------- C interface
static void mb_blit(uint32_t src, uint32_t sp, uint32_t dst, uint32_t dp, uint32_t bpp, uint32_t sx, uint32_t sy,
                    uint32_t dx, uint32_t dy, uint32_t w, uint32_t h) {
    @autoreleasepool { [gMetal blitFrom:src pitch:sp to:dst pitch:dp bpp:bpp sx:sx sy:sy dx:dx dy:dy w:w h:h]; }
}
static void mb_draw(const NvDrawState *st, const OutVtx *v, int count) {
    @autoreleasepool { [gMetal draw:st vertices:v count:count]; }
}
static void mb_clear(const NvDrawState *st, uint32_t flags, uint32_t color, uint32_t zs, uint32_t x0, uint32_t y0,
                     uint32_t x1, uint32_t y1) {
    @autoreleasepool { [gMetal clear:st flags:flags color:color zs:zs x0:x0 y0:y0 x1:x1 y1:y1]; }
}
static int mb_draw_vs(const NvDrawState *st, const NvVsDraw *d) {
    @autoreleasepool { return [gMetal drawVS:st vs:d] ? 1 : 0; }
}
static void mb_report(int op, uint32_t arg) {
    @autoreleasepool { [gMetal report:op arg:arg]; }
}
static void mb_flush(void) {
    @autoreleasepool { [gMetal flush]; }
}
// End of a frame (FLIP_INCREMENT_WRITE): submit it; everything drawn so far is complete.
static void publish_frame(id<MTLTexture> src) {
    int ready = __atomic_load_n(&gPresentReady, __ATOMIC_ACQUIRE), k = -1;
    for (int i = 0; i < 3; i++)
        if (i != ready && !__atomic_load_n(&gPresentBusy[i], __ATOMIC_ACQUIRE)) { k = i; break; }
    if (k < 0) { gGpuPublishSkipped++; return; }          // the GPU is far behind: keep showing the last one
    id<MTLTexture> dst = gPresentTex[k];
    if (!dst || dst.width != src.width || dst.height != src.height || dst.pixelFormat != src.pixelFormat) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat
                                                                                     width:src.width height:src.height mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModePrivate;
        dst = [gMetal.device newTextureWithDescriptor:d];
        gPresentTex[k] = dst;
    }
    __atomic_store_n(&gPresentBusy[k], 1, __ATOMIC_RELEASE);
    id<MTLCommandBuffer> cb = [gMetal.queue commandBuffer];
    cb.label = @"present copy";
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
            sourceSize:MTLSizeMake(src.width, src.height, 1)
             toTexture:dst destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b endEncoding];
    int kk = k;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        (void)c;
        __atomic_store_n(&gPresentReady, kk, __ATOMIC_RELEASE);
        __atomic_store_n(&gPresentBusy[kk], 0, __ATOMIC_RELEASE);
    }];
    [cb commit];
}

static void gpu_stats(void) {
    static unsigned flips;
    uint64_t now = host_time_ns();
    if (!gGpuStatT0) { gGpuStatT0 = now; return; }
    if (++flips < 150) return;
    double wall = (double)(now - gGpuStatT0);
    uint64_t busy = __atomic_exchange_n(&gGpuBusyNs, 0, __ATOMIC_RELAXED);
    fprintf(stderr, "[gpu] busy %.0f%% of the time (%.1f ms per frame), command buffers in flight max %d, frames not shown (GPU behind) %u | per frame: %u draws (%u lean shader, %u specialized, %u vertex program on GPU), %u render passes, %u texture uploads | specialized shaders %u (%u failed), vertex programs %u (%u failed) | visibility tests %u\n",
            100.0 * (double)busy / wall, (double)busy / 1e6 / flips, gGpuInFlightMax, gGpuPublishSkipped,
            gStatDraws / flips, gStatFastDraws / flips, gStatSpecDraws / flips, gStatVsDraws / flips, gStatPasses / flips, gStatUploads / flips,
            gSpecCompiled, gSpecFailed, gVsCompiled, gVsFailed, gStatQueries / flips);
    gStatDraws = gStatFastDraws = gStatPasses = gStatUploads = gStatSpecDraws = gStatVsDraws = gStatQueries = 0;
    flips = 0; gGpuStatT0 = now; gGpuInFlightMax = 0; gGpuPublishSkipped = 0;
}

static void mb_flip(void) {
    @autoreleasepool {
        [gMetal flush];
        uint32_t ps = gPendingScale;
        if (ps && ps != gScale) [gMetal applyScale:ps];
        os_unfair_lock_lock(&gTargetsLock);
        if (gDrawingAddr) gCompletedAddr = gDrawingAddr;
        id<MTLTexture> src = gCompletedAddr ? gMetal.colorTargets[@(gCompletedAddr)] : nil;
        os_unfair_lock_unlock(&gTargetsLock);
        if (src) publish_frame(src);
        gpu_stats();
    }
}

static GpuBackend gBackend = {mb_draw, mb_clear, mb_flush, mb_blit, mb_flip, mb_draw_vs, mb_report};

void metal_backend_install(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    if (!device || !queue || !library || ![library newFunctionWithName:@"nv_fs"]) {
        NSLog(@"HaloRecomp: Metal renderer unavailable, using the CPU renderer");
        return;
    }
    gMetal = [[NVMetal alloc] initWithDevice:device queue:queue library:library];
    /* Every shader keeps the device-memory write that turns off the GPU's hidden-surface
     * removal: with it on, surfaces drawn in several passes got single-pixel dots
     * (2026-11-08; the same quirk as the menu ring's holes). HALO_HSR=1 turns it back on
     * for draws without alpha test, to re-test on newer macOS versions. */
    gDisableFastShader = !(getenv("HALO_HSR") && *getenv("HALO_HSR") == '1');
    gDisableSpec = getenv("HALO_NO_SPEC_SHADER") && *getenv("HALO_NO_SPEC_SHADER") == '1';
    NSString *buildDir = [NSBundle.mainBundle.bundlePath stringByDeletingLastPathComponent];
    if ([NSFileManager.defaultManager fileExistsAtPath:[buildDir stringByAppendingPathComponent:@"gpu_debug.txt"]]) {
        gDbgDir = [buildDir stringByAppendingPathComponent:@"gpu_debug"];
        [NSFileManager.defaultManager createDirectoryAtPath:gDbgDir withIntermediateDirectories:YES attributes:nil error:nil];
        gDbgOn = 1;
        gDbgLog = fopen([gDbgDir stringByAppendingPathComponent:@"log.txt"].fileSystemRepresentation, "w");
        NSLog(@"HaloRecomp: GPU diagnosis on, frames go to %@", gDbgDir);
    }
    g_gpu_backend = &gBackend;
    NSLog(@"HaloRecomp: drawing with Metal on %@", device.name);
}

void metal_backend_set_render_scale(int scale) {
    gPendingScale = (uint32_t)(scale < 1 ? 1 : scale > 4 ? 4 : scale);
}

id<MTLTexture> metal_backend_surface(uint32_t addr) {
    if (!gMetal) return nil;
    // Only show a GPU render target the game has drawn into; one that was only
    // cleared (or never used) is shown from guest memory.
    os_unfair_lock_lock(&gTargetsLock);
    id<MTLTexture> t;
    int how;
    int ready = __atomic_load_n(&gPresentReady, __ATOMIC_ACQUIRE);
    if (gMetal.rtLastDraw[@(addr)] && ready >= 0 && gPresentTex[ready]) {
        t = gPresentTex[ready];                          // the game shows GPU-drawn frames: the last complete one
        how = gCompletedAddr == addr ? 1 : 2;
    } else if (gMetal.rtLastDraw[@(addr)] && gCompletedAddr && gMetal.colorTargets[@(gCompletedAddr)]) {
        t = gMetal.colorTargets[@(gCompletedAddr)];
        how = gCompletedAddr == addr ? 1 : 2;
    } else {
        t = gMetal.rtLastDraw[@(addr)] ? gMetal.colorTargets[@(addr)] : nil;
        how = t ? 1 : 0;
    }
    os_unfair_lock_unlock(&gTargetsLock);
    {   // flicker diagnosis: log changes of what is shown
        static int logged; static uint32_t lastAddr; static int lastHow = -1; static void *lastTex;
        if ((addr != lastAddr || how != lastHow || (__bridge void *)t != lastTex) && logged < 600) {
            logged++;
            fprintf(stderr, "[present] scanout %08x: %s (texture %p)\n", addr,
                    how == 2 ? "GPU, last complete frame from the other buffer" : how == 1 ? "GPU" : "guest memory",
                    (__bridge void *)t);
        }
        lastAddr = addr; lastHow = how; lastTex = (__bridge void *)t;
    }
    return t;
}
