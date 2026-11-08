// main.m — native macOS shell for the Halo recompilation.
//
// AppKit window + MTKView. The recompiled game runs on its own thread; the
// HLE Direct3D layer talks to this file through runtime/gfx.h. For now the
// renderer presents a 640x480 BGRA framebuffer (clears + any software-drawn
// content) scaled to the window with a Metal fragment shader; the full
// NV2A/D3D8 -> Metal translation plugs in behind the same interface.
#import <Cocoa/Cocoa.h>
#import <GameController/GameController.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <pthread.h>
#include <stdatomic.h>

#include "../../runtime/gfx.h"
#include "../../runtime/runtime.h"
#include "../../runtime/nv2a.h"
#import "input.h"
#import "graphics.h"
#import "settings.h"
#import "metal_backend.h"
#import "audio.h"
#include <unistd.h>

// ------------------------------------------------------------ shared state
static uint32_t gFBWidth = 640, gFBHeight = 480;
static uint32_t *gFramebuffer;                 // guest-visible "back buffer" (BGRA)
static uint32_t *gFrontBuffer;                 // last presented frame
static pthread_mutex_t gFBLock = PTHREAD_MUTEX_INITIALIZER;
static dispatch_semaphore_t gVBlank;
static volatile uint64_t gPresentCount;
static volatile uint32_t gScanoutAddr;          // guest address of the displayed framebuffer

// ---------------------------------------------------------- gfx.h backend
void gfx_create_device(uint32_t w, uint32_t h) {
    pthread_mutex_lock(&gFBLock);
    gFBWidth = w ? w : 640;
    gFBHeight = h ? h : 480;
    free(gFramebuffer);
    free(gFrontBuffer);
    gFramebuffer = calloc((size_t)gFBWidth * gFBHeight, 4);
    gFrontBuffer = calloc((size_t)gFBWidth * gFBHeight, 4);
    pthread_mutex_unlock(&gFBLock);
}

void gfx_clear(uint32_t argb, uint32_t flags) {
    if (!(flags & 0x000000F0) && flags) return;   // D3DCLEAR_TARGET is 0x000000F0 on Xbox
    pthread_mutex_lock(&gFBLock);
    if (gFramebuffer) {
        uint32_t bgra = argb;                     // D3DCOLOR is already ARGB == BGRA in memory
        size_t n = (size_t)gFBWidth * gFBHeight;
        for (size_t i = 0; i < n; i++) gFramebuffer[i] = bgra;
    }
    pthread_mutex_unlock(&gFBLock);
}

void gfx_present(void) {
    pthread_mutex_lock(&gFBLock);
    if (gFramebuffer) memcpy(gFrontBuffer, gFramebuffer, (size_t)gFBWidth * gFBHeight * 4);
    gPresentCount++;
    pthread_mutex_unlock(&gFBLock);
}

void gfx_present_frame(const uint32_t *px, uint32_t w, uint32_t h, uint32_t pitch) {
    pthread_mutex_lock(&gFBLock);
    if (w != gFBWidth || h != gFBHeight || !gFrontBuffer) {
        free(gFramebuffer); free(gFrontBuffer);
        gFBWidth = w; gFBHeight = h;
        gFramebuffer = calloc((size_t)w * h, 4);
        gFrontBuffer = calloc((size_t)w * h, 4);
    }
    gScanoutAddr = (uint32_t)((const uint8_t *)px - g_mem);
    for (uint32_t y = 0; y < h; y++)
        memcpy(gFrontBuffer + (size_t)y * w, (const uint8_t *)px + (size_t)y * pitch, (size_t)w * 4);
    gPresentCount++;
    pthread_mutex_unlock(&gFBLock);
}

void gfx_wait_vblank(void) {
    // Let other guest threads run while we wait for the display.
    guest_unlock();
    dispatch_semaphore_wait(gVBlank, dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC));
    guest_lock();
}

// Input (keyboard, mouse, game controllers, Input Settings window): input.m

// --------------------------------------------------------------- renderer
@interface Renderer : NSObject <MTKViewDelegate>
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLCommandQueue> presentQueue;   // the window's own queue: never waits behind the game's GPU work
@property(nonatomic, strong) id<MTLRenderPipelineState> pipeline;
@property(nonatomic, strong) id<MTLTexture> frameTexture;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLSamplerState> nearestSampler, linearSampler;
@end

@implementation Renderer
- (instancetype)initWithView:(MTKView *)view {
    if ((self = [super init])) {
        _device = view.device;
        _queue = [_device newCommandQueue];
        _presentQueue = [_device newCommandQueue];
        NSError *err = nil;
        NSString *libPath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        id<MTLLibrary> lib = libPath ? [_device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err] : nil;
        if (!lib) {   // Command Line Tools only: compile the shipped shader source at runtime
            NSString *src = [NSString stringWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"Shaders" ofType:@"metal"]
                                                      encoding:NSUTF8StringEncoding error:nil];
            MTLCompileOptions *opts = [MTLCompileOptions new];
            if (@available(macOS 11.0, *)) opts.preserveInvariance = YES;   // honour [[invariant]] (multi-pass depth EQUAL)
            if (src) lib = [_device newLibraryWithSource:src options:opts error:&err];
        }
        if (err) {   // compiler messages next to the app: build/shader_errors.txt
            NSString *out = [[[NSBundle mainBundle].bundlePath stringByDeletingLastPathComponent]
                stringByAppendingPathComponent:@"shader_errors.txt"];
            [err.description writeToFile:out atomically:YES encoding:NSUTF8StringEncoding error:nil];
        }
        if (!lib) { NSLog(@"Metal library missing: %@", err); return nil; }
        _library = lib;
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"fullscreen_vs"];
        pd.fragmentFunction = [lib newFunctionWithName:@"present_fs"];
        pd.colorAttachments[0].pixelFormat = view.colorPixelFormat;
        _pipeline = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!_pipeline) NSLog(@"pipeline error: %@", err);
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        _nearestSampler = [_device newSamplerStateWithDescriptor:sd];
        sd.minFilter = MTLSamplerMinMagFilterLinear;
        sd.magFilter = MTLSamplerMinMagFilterLinear;
        _linearSampler = [_device newSamplerStateWithDescriptor:sd];
    }
    return self;
}

- (void)ensureTexture {
    if (_frameTexture && _frameTexture.width == gFBWidth && _frameTexture.height == gFBHeight) return;
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                  width:gFBWidth
                                                                                 height:gFBHeight
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    _frameTexture = [_device newTextureWithDescriptor:td];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {}

- (void)drawInMTKView:(MTKView *)view {
    nv2a_host_vblank();
    dispatch_semaphore_signal(gVBlank);
    // With the Metal renderer the displayed framebuffer is a GPU texture;
    // otherwise (CPU renderer, or nothing drawn there yet) upload guest memory.
    id<MTLTexture> shown = metal_backend_surface(gScanoutAddr);
    if (!shown) {
        [self ensureTexture];
        pthread_mutex_lock(&gFBLock);
        if (gFrontBuffer)
            [_frameTexture replaceRegion:MTLRegionMake2D(0, 0, gFBWidth, gFBHeight)
                             mipmapLevel:0
                               withBytes:gFrontBuffer
                             bytesPerRow:gFBWidth * 4];
        pthread_mutex_unlock(&gFBLock);
        shown = _frameTexture;
    }

    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    if (!rp) return;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    id<MTLCommandBuffer> cb = [_presentQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    // Graphics settings: where the picture goes, how it is filtered (Shaders.metal).
    CGSize ds = view.drawableSize;
    CGRect r = graphics_output_rect(ds, gFBWidth, gFBHeight);
    PresentParams pp = {
        .src_size = {(float)shown.width, (float)shown.height},
        .rect_origin = {(float)r.origin.x, (float)r.origin.y},
        .rect_size = {(float)r.size.width, (float)r.size.height},
        .filter = g_gfx.filter, .scanlines = g_gfx.scanlines,
        .brightness = g_gfx.brightness, .gamma = g_gfx.gamma};
    [enc setViewport:(MTLViewport){0, 0, ds.width, ds.height, 0, 1}];
    [enc setRenderPipelineState:_pipeline];
    [enc setFragmentTexture:shown atIndex:0];
    [enc setFragmentSamplerState:_nearestSampler atIndex:0];
    [enc setFragmentSamplerState:_linearSampler atIndex:1];
    [enc setFragmentBytes:&pp length:sizeof pp atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb presentDrawable:view.currentDrawable];
    [cb commit];
}
@end

// -------------------------------------------------------------- app shell
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) MTKView *view;
@property(nonatomic, strong) Renderer *renderer;
@end

static RuntimeConfig gConfig;
static NSString *gXbePath, *gIsoPath;

static void *game_thread(void *arg) {
    (void)arg;
    if (runtime_init(&gConfig)) {
        dispatch_async(dispatch_get_main_queue(), ^{
            NSAlert *a = [NSAlert new];
            a.messageText = @"Could not start the game";
            a.informativeText = [NSString stringWithFormat:
                @"Looked for:\n• %@\n• disc image: %@\n\nRun `make extract` in the HaloRecomp folder, or set HALO_XBE / HALO_ISO. "
                @"Details are in the log (run the app from Terminal to see it).",
                gXbePath, gIsoPath ?: @"none found"];
            [a runModal];
            [NSApp terminate:nil];
        });
        return NULL;
    }
    runtime_run();
    return NULL;
}

static NSString *resolvePath(NSString *p, NSString *base) {
    if (!p.length) return nil;
    if ([p hasPrefix:@"/"]) return p;
    return [base stringByAppendingPathComponent:p];
}

@implementation AppDelegate
- (void)applicationDidFinishLaunching:(NSNotification *)note {
    gVBlank = dispatch_semaphore_create(0);
    graphics_load();
    gfx_create_device(640, 480);
    g_host.poll_input = input_poll;

    NSRect frame = NSMakeRect(0, 0, 1280, 960);
    _window = [[NSWindow alloc] initWithContentRect:frame
                                          styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                                    NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"Halo Recomp";
    _window.delegate = self;
    _window.collectionBehavior = NSWindowCollectionBehaviorFullScreenPrimary;
    _view = [[MTKView alloc] initWithFrame:frame device:MTLCreateSystemDefaultDevice()];
    _view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    _view.preferredFramesPerSecond = 60;
    _renderer = [[Renderer alloc] initWithView:_view];
    // Pixel work on the GPU unless the CPU renderer is chosen (Settings → Graphics, or HALO_RENDERER=cpu).
    NSString *rendererEnv = NSProcessInfo.processInfo.environment[@"HALO_RENDERER"];
    BOOL useCPU = rendererEnv ? [rendererEnv.lowercaseString isEqualToString:@"cpu"] : g_gfx.renderer == GFX_RENDERER_CPU;
    if (!useCPU && _renderer) metal_backend_install(_renderer.device, _renderer.queue, _renderer.library);
    _view.delegate = _renderer;
    _window.contentView = _view;
    [_window center];
    input_install(_window);
    graphics_install(_window, _view);
    audio_install();
    [_window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];

    // Paths: find the HaloRecomp folder by walking up from the .app until
    // extracted/default.xbe appears (the app normally sits in HaloRecomp/build/).
    // Overrides: HALO_XBE / HALO_ISO / HALO_SAVES (absolute or relative to that folder).
    NSFileManager *fm = [NSFileManager defaultManager];
    NSDictionary *env = NSProcessInfo.processInfo.environment;
    NSString *base = [[NSBundle mainBundle] bundlePath];
    for (int i = 0; i < 4; i++) {
        base = [base stringByDeletingLastPathComponent];
        if ([fm fileExistsAtPath:[base stringByAppendingPathComponent:@"extracted/default.xbe"]]) break;
    }
    NSString *xbe = resolvePath(env[@"HALO_XBE"] ?: @"extracted/default.xbe", base);
    NSString *iso = resolvePath(env[@"HALO_ISO"] ?: @"", base);
    if (!iso) {
        // default: the first .iso in the HaloRecomp folder or the folder above it
        for (NSString *dir in @[base, [base stringByDeletingLastPathComponent]]) {
            for (NSString *f in [fm contentsOfDirectoryAtPath:dir error:nil])
                if ([f.pathExtension.lowercaseString isEqualToString:@"iso"]) { iso = [dir stringByAppendingPathComponent:f]; break; }
            if (iso) break;
        }
    }
    NSString *saves = resolvePath(env[@"HALO_SAVES"] ?: @"xbox_hdd", base);
    gXbePath = xbe;
    gIsoPath = iso;
    NSLog(@"HaloRecomp: xbe=%@ iso=%@ saves=%@", xbe, iso ?: @"(none)", saves);
    gConfig.xbe_path = strdup(xbe.fileSystemRepresentation);
    gConfig.iso_path = iso ? strdup(iso.fileSystemRepresentation) : NULL;
    gConfig.save_root = strdup(saves.fileSystemRepresentation);
    gConfig.trace_kernel = env[@"HALO_TRACE_KERNEL"] != nil;
    gConfig.trace_calls = env[@"HALO_TRACE_CALLS"] != nil;

    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 64u << 20);   // deep recompiled call chains
    pthread_t th;
    pthread_create(&th, &at, game_thread, NULL);
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)app { return YES; }
@end

static void build_menu(void) {
    NSMenu *bar = [NSMenu new];
    NSString *name = @"Halo Recomp";

    NSMenuItem *appItem = [NSMenuItem new];
    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:name];
    [appMenu addItemWithTitle:[@"About " stringByAppendingString:name]
                       action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:[@"Hide " stringByAppendingString:name] action:@selector(hide:) keyEquivalent:@"h"];
    NSMenuItem *hideOthers = [appMenu addItemWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
    hideOthers.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagOption;
    [appMenu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:[@"Quit " stringByAppendingString:name] action:@selector(terminate:) keyEquivalent:@"q"];
    appItem.submenu = appMenu;
    [bar addItem:appItem];

    settings_add_menu(bar);   // Settings… (Input, Graphics, Audio pages), Full Screen

    NSMenuItem *winItem = [NSMenuItem new];
    NSMenu *winMenu = [[NSMenu alloc] initWithTitle:@"Window"];
    [winMenu addItemWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"];
    [winMenu addItem:[NSMenuItem separatorItem]];
    [winMenu addItemWithTitle:@"Close" action:@selector(performClose:) keyEquivalent:@"w"];
    winItem.submenu = winMenu;
    [bar addItem:winItem];

    NSApp.mainMenu = bar;
    NSApp.windowsMenu = winMenu;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        // The runtime logs to stderr; when started from Finder or `open`, keep
        // that in halo.log next to the app (build/halo.log), previous run in halo.prev.log.
        if (!isatty(2)) {
            NSString *dir = [NSBundle.mainBundle.bundlePath stringByDeletingLastPathComponent];
            NSString *log = [dir stringByAppendingPathComponent:@"halo.log"];
            NSString *prev = [dir stringByAppendingPathComponent:@"halo.prev.log"];
            [NSFileManager.defaultManager removeItemAtPath:prev error:nil];
            [NSFileManager.defaultManager moveItemAtPath:log toPath:prev error:nil];
            freopen(log.fileSystemRepresentation, "w", stderr);
        }
        {   // diagnosis: build/script_funcs.txt present -> log Halo script function calls (runtime.c)
            NSString *dir = [NSBundle.mainBundle.bundlePath stringByDeletingLastPathComponent];
            NSString *sf = [dir stringByAppendingPathComponent:@"script_funcs.txt"];
            if ([NSFileManager.defaultManager fileExistsAtPath:sf]) setenv("HALO_TRACE_SCRIPT", sf.fileSystemRepresentation, 1);
        }
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        build_menu();
        AppDelegate *d = [AppDelegate new];
        app.delegate = d;
        [app run];
    }
    return 0;
}
