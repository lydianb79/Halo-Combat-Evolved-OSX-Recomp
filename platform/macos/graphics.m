// graphics.m — display options for the macOS shell: full screen, window size,
// aspect ratio, scaling, filtering, scanlines, brightness/gamma, performance
// overlay and render threads. Shown as the "Graphics" page of the Settings
// window (settings.m). Settings are stored in the user defaults.
#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>
#include <math.h>

#import "graphics.h"
#import "metal_backend.h"
#include "../../runtime/nv2a.h"
#include "../../runtime/nv2a_raster.h"

GraphicsSettings g_gfx;

static NSArray<NSString *> *filter_names(void) {
    return @[@"Nearest (sharp pixels)", @"Bilinear (smooth)", @"Sharp bilinear", @"Bicubic", @"Lanczos (sharp, smooth)"];
}
static NSArray<NSString *> *texture_names(void) {
    return @[@"As the game sets it", @"Fastest (nearest, no mipmaps)", @"Nearest", @"Bilinear", @"Trilinear",
             @"Anisotropic 2×", @"Anisotropic 4×", @"Anisotropic 8×", @"Anisotropic 16×"];
}
static NSArray<NSString *> *scale_names(void) {
    return @[@"Fit to window", @"Integer scale (pixel-perfect)", @"Fill window (crop)"];
}
static NSArray<NSString *> *aspect_names(void) {
    return @[@"4:3 (original)", @"16:9 (stretched)", @"Stretch to window"];
}
static const int kThreadChoices[] = {0, 1, 2, 4, 6, 8, 12, 16};
#define NTHREADCHOICES ((int)(sizeof kThreadChoices / sizeof kThreadChoices[0]))

static NSString *const kDefaults = @"HaloGraphics.v1";

static __weak NSWindow *gGameWindow;
static __weak MTKView *gGameView;
static NSTextField *gStatsLabel;
static NSTimer *gStatsTimer;
static BOOL gCursorHidden;

// ------------------------------------------------------------- settings
static void reset_defaults(void) {
    g_gfx = (GraphicsSettings){
        .filter = GFX_FILTER_SHARP, .scaling = GFX_SCALE_FIT, .aspect = GFX_ASPECT_4_3, .texFilter = TEX_GAME,
        .escExitsFullScreen = YES,
        .scanlines = 0, .brightness = 1, .gamma = 1,
        .showStats = NO, .startFullScreen = NO, .hideCursor = YES, .renderThreads = 0, .renderScale = 1};
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

static void save_settings(void) {
    NSDictionary *d = @{
        @"filter": @(g_gfx.filter), @"scaling": @(g_gfx.scaling), @"aspect": @(g_gfx.aspect),
        @"scanlines": @(g_gfx.scanlines), @"brightness": @(g_gfx.brightness), @"gamma": @(g_gfx.gamma),
        @"showStats": @(g_gfx.showStats), @"startFullScreen": @(g_gfx.startFullScreen),
        @"hideCursor": @(g_gfx.hideCursor), @"renderThreads": @(g_gfx.renderThreads),
        @"texFilter": @(g_gfx.texFilter), @"renderer": @(g_gfx.renderer), @"escExitsFullScreen": @(g_gfx.escExitsFullScreen),
        @"renderScale": @(g_gfx.renderScale), @"filterMigrated": @YES};
    [NSUserDefaults.standardUserDefaults setObject:d forKey:kDefaults];
}

void graphics_load(void) {
    reset_defaults();
    NSDictionary *d = [NSUserDefaults.standardUserDefaults dictionaryForKey:kDefaults];
    #define NUM(k) ([d[k] isKindOfClass:NSNumber.class] ? (NSNumber *)d[k] : nil)
    if (NUM(@"filter")) g_gfx.filter = clampi(NUM(@"filter").intValue, 0, GFX_FILTER_COUNT - 1);
    if (NUM(@"scaling")) g_gfx.scaling = clampi(NUM(@"scaling").intValue, 0, GFX_SCALE_COUNT - 1);
    if (NUM(@"aspect")) g_gfx.aspect = clampi(NUM(@"aspect").intValue, 0, GFX_ASPECT_COUNT - 1);
    if (NUM(@"scanlines")) g_gfx.scanlines = clampf(NUM(@"scanlines").floatValue, 0, 1);
    if (NUM(@"brightness")) g_gfx.brightness = clampf(NUM(@"brightness").floatValue, 0.5f, 1.5f);
    if (NUM(@"gamma")) g_gfx.gamma = clampf(NUM(@"gamma").floatValue, 0.6f, 1.6f);
    if (NUM(@"showStats")) g_gfx.showStats = NUM(@"showStats").boolValue;
    if (NUM(@"startFullScreen")) g_gfx.startFullScreen = NUM(@"startFullScreen").boolValue;
    if (NUM(@"hideCursor")) g_gfx.hideCursor = NUM(@"hideCursor").boolValue;
    if (NUM(@"renderThreads")) g_gfx.renderThreads = clampi(NUM(@"renderThreads").intValue, 0, 64);
    if (NUM(@"renderer")) g_gfx.renderer = clampi(NUM(@"renderer").intValue, 0, GFX_RENDERER_COUNT - 1);
    if (NUM(@"texFilter")) g_gfx.texFilter = clampi(NUM(@"texFilter").intValue, 0, TEX_COUNT - 1);
    // Settings saved before the Metal renderer defaulted to "Fastest" texture
    // filtering; on the GPU filtering is free, so move them to the game's own filters once.
    if (!NUM(@"filterMigrated") && g_gfx.texFilter == TEX_FASTEST) g_gfx.texFilter = TEX_GAME;
    if (NUM(@"escExitsFullScreen")) g_gfx.escExitsFullScreen = NUM(@"escExitsFullScreen").boolValue;
    if (NUM(@"renderScale")) g_gfx.renderScale = clampi(NUM(@"renderScale").intValue, 1, 4);
    #undef NUM
    nv2a_raster_set_texture_filter(g_gfx.texFilter);
    metal_backend_set_render_scale(g_gfx.renderScale);
    // The software renderer reads its thread count once, when it starts.
    if (g_gfx.renderThreads > 0 && !getenv("HALO_RASTER_THREADS")) {
        char buf[16];
        snprintf(buf, sizeof buf, "%d", g_gfx.renderThreads);
        setenv("HALO_RASTER_THREADS", buf, 1);
    }
}

// ------------------------------------------------------------- geometry
static double target_aspect(CGSize s) {
    switch (g_gfx.aspect) {
    case GFX_ASPECT_16_9: return 16.0 / 9.0;
    case GFX_ASPECT_STRETCH: return s.height > 0 ? s.width / s.height : 4.0 / 3.0;
    default: return 4.0 / 3.0;
    }
}

CGRect graphics_output_rect(CGSize s, double srcW, double srcH) {
    (void)srcW;
    double target = target_aspect(s), w = s.width, h = s.height;
    BOOL wider = s.width / fmax(s.height, 1) > target;
    if (g_gfx.scaling == GFX_SCALE_INTEGER && srcH > 0) {
        double k = floor(fmin(s.height / srcH, s.width / (srcH * target)));
        if (k >= 1) { h = srcH * k; w = h * target; }
        else if (wider) { h = s.height; w = h * target; }
        else { w = s.width; h = w / target; }
    } else if (g_gfx.scaling == GFX_SCALE_FILL) {
        if (wider) { w = s.width; h = w / target; } else { h = s.height; w = h * target; }
    } else {
        if (wider) { h = s.height; w = h * target; } else { w = s.width; h = w / target; }
    }
    return CGRectMake(floor((s.width - w) / 2), floor((s.height - h) / 2), w, h);
}

// ------------------------------------------------------- window helpers
static BOOL is_full_screen(void) {
    NSWindow *w = gGameWindow;
    return w && (w.styleMask & NSWindowStyleMaskFullScreen) != 0;
}
BOOL graphics_is_full_screen(void) { return is_full_screen(); }

static void update_cursor(void) {
    BOOL want = is_full_screen() && g_gfx.hideCursor;
    if (want && !gCursorHidden) { [NSCursor hide]; gCursorHidden = YES; }
    if (!want && gCursorHidden) { [NSCursor unhide]; gCursorHidden = NO; }
}

static void set_window_scale(int k) {
    NSWindow *w = gGameWindow;
    if (!w || is_full_screen()) return;
    double target = g_gfx.aspect == GFX_ASPECT_16_9 ? 16.0 / 9.0 : 4.0 / 3.0;
    NSSize size = NSMakeSize(round(480.0 * k * target), 480.0 * k);
    NSRect vis = (w.screen ?: NSScreen.mainScreen).visibleFrame;
    NSRect frame = [w frameRectForContentRect:NSMakeRect(0, 0, size.width, size.height)];
    if (frame.size.width > vis.size.width || frame.size.height > vis.size.height) {
        double f = fmin(vis.size.width / frame.size.width, vis.size.height / frame.size.height);
        size = NSMakeSize(floor(size.width * f), floor(size.height * f));
        frame = [w frameRectForContentRect:NSMakeRect(0, 0, size.width, size.height)];
    }
    frame.origin.x = vis.origin.x + (vis.size.width - frame.size.width) / 2;
    frame.origin.y = vis.origin.y + (vis.size.height - frame.size.height) / 2;
    [w setFrame:frame display:YES animate:YES];
}

// ------------------------------------------------------ performance overlay
static void update_stats(void) {
    static uint32_t lastFlips, lastVblanks;
    static double lastTime;
    double now = CACurrentMediaTime();
    uint32_t flips = nv2a_stat_flips(), vbl = nv2a_stat_vblanks();
    if (lastTime > 0 && now > lastTime) {
        double dt = now - lastTime;
        double fps = (flips - lastFlips) / dt, speed = (vbl - lastVblanks) / dt / 60.0 * 100.0;
        gStatsLabel.stringValue = [NSString stringWithFormat:@" %4.1f fps   speed %3.0f%% ", fps, speed];
    }
    lastFlips = flips; lastVblanks = vbl; lastTime = now;
}

static void apply_stats_visibility(void) {
    gStatsLabel.hidden = !g_gfx.showStats;
    if (g_gfx.showStats && !gStatsTimer) {
        gStatsTimer = [NSTimer scheduledTimerWithTimeInterval:0.5 repeats:YES
                                                        block:^(NSTimer *t) { (void)t; update_stats(); }];
        update_stats();
    } else if (!g_gfx.showStats && gStatsTimer) {
        [gStatsTimer invalidate];
        gStatsTimer = nil;
    }
}

// ----------------------------------------------------------- the UI
@interface GraphicsSettingsUI : NSObject
@property(nonatomic, strong) NSView *view;
@property(nonatomic, strong) NSButton *fullScreenBox, *startFullBox, *hideCursorBox, *escBox, *statsBox;
@property(nonatomic, strong) NSPopUpButton *sizePopup, *aspectPopup, *scalePopup, *filterPopup, *texPopup, *threadsPopup, *rendererPopup, *resPopup;
@property(nonatomic, strong) NSSlider *scanSlider, *brightSlider, *gammaSlider;
@property(nonatomic, strong) NSTextField *scanLabel, *brightLabel, *gammaLabel;
+ (instancetype)shared;
@end

@implementation GraphicsSettingsUI

+ (instancetype)shared {
    static GraphicsSettingsUI *s;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ s = [GraphicsSettingsUI new]; });
    return s;
}

static NSTextField *label(NSString *t) {
    NSTextField *l = [NSTextField labelWithString:t];
    l.alignment = NSTextAlignmentRight;
    return l;
}

static NSPopUpButton *popup(NSArray<NSString *> *items, id target, SEL action) {
    NSPopUpButton *p = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    [p addItemsWithTitles:items];
    p.target = target;
    p.action = action;
    [p.widthAnchor constraintGreaterThanOrEqualToConstant:240].active = YES;
    return p;
}

static NSView *slider_row(NSSlider *s, NSTextField *value) {
    [s.widthAnchor constraintEqualToConstant:200].active = YES;
    [value.widthAnchor constraintEqualToConstant:50].active = YES;
    NSStackView *st = [NSStackView stackViewWithViews:@[s, value]];
    st.spacing = 8;
    return st;
}

- (void)buildView {
    self.fullScreenBox = [NSButton checkboxWithTitle:@"Full screen" target:self action:@selector(toggleFullScreen:)];
    self.startFullBox = [NSButton checkboxWithTitle:@"Start in full screen" target:self action:@selector(changed:)];
    self.hideCursorBox = [NSButton checkboxWithTitle:@"Hide the pointer in full screen" target:self action:@selector(changed:)];
    self.escBox = [NSButton checkboxWithTitle:@"Esc leaves full screen" target:self action:@selector(changed:)];
    self.statsBox = [NSButton checkboxWithTitle:@"Show frame rate and speed" target:self action:@selector(changed:)];

    self.sizePopup = popup(@[@"1× (640 × 480)", @"2× (1280 × 960)", @"3× (1920 × 1440)", @"4× (2560 × 1920)"],
                           self, @selector(sizeChosen:));
    [self.sizePopup insertItemWithTitle:@"Choose a size…" atIndex:0];
    self.aspectPopup = popup(aspect_names(), self, @selector(changed:));
    self.scalePopup = popup(scale_names(), self, @selector(changed:));
    self.filterPopup = popup(filter_names(), self, @selector(changed:));
    self.texPopup = popup(texture_names(), self, @selector(changed:));
    self.resPopup = popup(@[@"1× (640 × 480, original)", @"2× (1280 × 960)", @"3× (1920 × 1440)", @"4× (2560 × 1920)"],
                          self, @selector(changed:));
    NSMutableArray *threadNames = [NSMutableArray array];
    for (int i = 0; i < NTHREADCHOICES; i++)
        [threadNames addObject:kThreadChoices[i] ? [NSString stringWithFormat:@"%d", kThreadChoices[i]]
                                                 : [NSString stringWithFormat:@"Automatic (%ld, one per core)",
                                                    (long)NSProcessInfo.processInfo.activeProcessorCount]];
    self.threadsPopup = popup(threadNames, self, @selector(changed:));
    self.rendererPopup = popup(@[@"Metal (GPU, fast)", @"CPU (reference)"], self, @selector(changed:));

    self.scanSlider = [NSSlider sliderWithValue:0 minValue:0 maxValue:1 target:self action:@selector(changed:)];
    self.brightSlider = [NSSlider sliderWithValue:1 minValue:0.5 maxValue:1.5 target:self action:@selector(changed:)];
    self.gammaSlider = [NSSlider sliderWithValue:1 minValue:0.6 maxValue:1.6 target:self action:@selector(changed:)];
    self.scanLabel = [NSTextField labelWithString:@""];
    self.brightLabel = [NSTextField labelWithString:@""];
    self.gammaLabel = [NSTextField labelWithString:@""];

    NSTextField *threadNote = [NSTextField labelWithString:@"Renderer and threads take effect the next time the game starts."];
    threadNote.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    threadNote.textColor = NSColor.secondaryLabelColor;
    NSTextField *aspectNote = [NSTextField labelWithString:@"Halo draws a 4:3 picture; 16:9 stretches it."];
    aspectNote.font = threadNote.font;
    aspectNote.textColor = NSColor.secondaryLabelColor;
    NSTextField *texNote = [NSTextField labelWithString:@"With the CPU renderer, smoother textures cost speed."];
    texNote.font = threadNote.font;
    texNote.textColor = NSColor.secondaryLabelColor;
    NSView *empty = NSGridCell.emptyContentView;

    NSGridView *grid = [NSGridView gridViewWithViews:@[
        @[label(@"Display:"), self.fullScreenBox],
        @[empty, self.startFullBox],
        @[empty, self.hideCursorBox],
        @[empty, self.escBox],
        @[label(@"Window size:"), self.sizePopup],
        @[label(@"Aspect ratio:"), self.aspectPopup],
        @[empty, aspectNote],
        @[label(@"Scaling:"), self.scalePopup],
        @[label(@"Screen filter:"), self.filterPopup],
        @[label(@"Render resolution:"), self.resPopup],
        @[label(@"Texture filtering:"), self.texPopup],
        @[empty, texNote],
        @[label(@"Scanlines:"), slider_row(self.scanSlider, self.scanLabel)],
        @[label(@"Brightness:"), slider_row(self.brightSlider, self.brightLabel)],
        @[label(@"Gamma:"), slider_row(self.gammaSlider, self.gammaLabel)],
        @[label(@"Overlay:"), self.statsBox],
        @[label(@"Renderer:"), self.rendererPopup],
        @[label(@"CPU render threads:"), self.threadsPopup],
        @[empty, threadNote],
    ]];
    grid.rowSpacing = 8;
    grid.columnSpacing = 10;
    grid.yPlacement = NSGridCellPlacementCenter;
    [grid columnAtIndex:0].xPlacement = NSGridCellPlacementTrailing;
    NSSize gs = grid.fittingSize;

    CGFloat footer = 56, width = gs.width + 48, height = gs.height + 40 + footer;
    NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
    self.view = content;
    grid.translatesAutoresizingMaskIntoConstraints = YES;
    grid.frame = NSMakeRect(24, footer + 16, gs.width, gs.height);
    [content addSubview:grid];

    NSButton *restore = [NSButton buttonWithTitle:@"Restore Defaults" target:self action:@selector(restoreDefaults:)];
    restore.frame = NSMakeRect(14, 12, 150, 32);
    [content addSubview:restore];
    NSButton *done = [NSButton buttonWithTitle:@"Done" target:nil action:@selector(performClose:)];
    done.frame = NSMakeRect(width - 104, 12, 90, 32);
    [content addSubview:done];
}

- (void)refresh {
    self.fullScreenBox.state = is_full_screen() ? NSControlStateValueOn : NSControlStateValueOff;
    self.startFullBox.state = g_gfx.startFullScreen ? NSControlStateValueOn : NSControlStateValueOff;
    self.hideCursorBox.state = g_gfx.hideCursor ? NSControlStateValueOn : NSControlStateValueOff;
    self.escBox.state = g_gfx.escExitsFullScreen ? NSControlStateValueOn : NSControlStateValueOff;
    [self.texPopup selectItemAtIndex:g_gfx.texFilter];
    [self.resPopup selectItemAtIndex:g_gfx.renderScale - 1];
    self.statsBox.state = g_gfx.showStats ? NSControlStateValueOn : NSControlStateValueOff;
    self.sizePopup.enabled = !is_full_screen();
    [self.sizePopup selectItemAtIndex:0];
    [self.aspectPopup selectItemAtIndex:g_gfx.aspect];
    [self.scalePopup selectItemAtIndex:g_gfx.scaling];
    [self.filterPopup selectItemAtIndex:g_gfx.filter];
    int ti = 0;
    for (int i = 0; i < NTHREADCHOICES; i++) if (kThreadChoices[i] == g_gfx.renderThreads) ti = i;
    [self.threadsPopup selectItemAtIndex:ti];
    [self.rendererPopup selectItemAtIndex:g_gfx.renderer];
    self.scanSlider.doubleValue = g_gfx.scanlines;
    self.brightSlider.doubleValue = g_gfx.brightness;
    self.gammaSlider.doubleValue = g_gfx.gamma;
    self.scanLabel.stringValue = [NSString stringWithFormat:@"%d%%", (int)lround(g_gfx.scanlines * 100)];
    self.brightLabel.stringValue = [NSString stringWithFormat:@"%d%%", (int)lround(g_gfx.brightness * 100)];
    self.gammaLabel.stringValue = [NSString stringWithFormat:@"%.2f", g_gfx.gamma];
}

- (NSView *)settingsView {
    if (!self.view) [self buildView];
    [self refresh];
    return self.view;
}

- (void)changed:(id)sender {
    if (sender == self.startFullBox) g_gfx.startFullScreen = self.startFullBox.state == NSControlStateValueOn;
    if (sender == self.hideCursorBox) g_gfx.hideCursor = self.hideCursorBox.state == NSControlStateValueOn;
    if (sender == self.escBox) g_gfx.escExitsFullScreen = self.escBox.state == NSControlStateValueOn;
    if (sender == self.texPopup) g_gfx.texFilter = (int)self.texPopup.indexOfSelectedItem;
    if (sender == self.resPopup) g_gfx.renderScale = (int)self.resPopup.indexOfSelectedItem + 1;
    if (sender == self.statsBox) g_gfx.showStats = self.statsBox.state == NSControlStateValueOn;
    if (sender == self.aspectPopup) g_gfx.aspect = (int)self.aspectPopup.indexOfSelectedItem;
    if (sender == self.scalePopup) g_gfx.scaling = (int)self.scalePopup.indexOfSelectedItem;
    if (sender == self.filterPopup) g_gfx.filter = (int)self.filterPopup.indexOfSelectedItem;
    if (sender == self.rendererPopup) g_gfx.renderer = (int)self.rendererPopup.indexOfSelectedItem;
    if (sender == self.threadsPopup) g_gfx.renderThreads = kThreadChoices[self.threadsPopup.indexOfSelectedItem];
    if (sender == self.scanSlider) g_gfx.scanlines = (float)(round(self.scanSlider.doubleValue * 100) / 100);
    if (sender == self.brightSlider) g_gfx.brightness = (float)(round(self.brightSlider.doubleValue * 100) / 100);
    if (sender == self.gammaSlider) g_gfx.gamma = (float)(round(self.gammaSlider.doubleValue * 100) / 100);
    [self applyAll];
}

- (void)applyAll {
    save_settings();
    nv2a_raster_set_texture_filter(g_gfx.texFilter);
    metal_backend_set_render_scale(g_gfx.renderScale);
    update_cursor();
    apply_stats_visibility();
    [self refresh];
}

- (void)toggleFullScreen:(id)sender {
    (void)sender;
    [gGameWindow toggleFullScreen:nil];
}

- (void)sizeChosen:(NSPopUpButton *)p {
    NSInteger i = p.indexOfSelectedItem;
    if (i >= 1) set_window_scale((int)i);
    [p selectItemAtIndex:0];
}

- (void)restoreDefaults:(id)sender {
    (void)sender;
    reset_defaults();
    [self applyAll];
}

@end

// ------------------------------------------------------------------- setup
void graphics_install(NSWindow *gameWindow, MTKView *view) {
    gGameWindow = gameWindow;
    gGameView = view;

    gStatsLabel = [NSTextField labelWithString:@""];
    gStatsLabel.font = [NSFont monospacedDigitSystemFontOfSize:13 weight:NSFontWeightMedium];
    gStatsLabel.textColor = NSColor.whiteColor;
    gStatsLabel.drawsBackground = YES;
    gStatsLabel.backgroundColor = [NSColor colorWithWhite:0 alpha:0.55];
    gStatsLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [view addSubview:gStatsLabel];
    [gStatsLabel.leadingAnchor constraintEqualToAnchor:view.leadingAnchor constant:10].active = YES;
    [gStatsLabel.topAnchor constraintEqualToAnchor:view.topAnchor constant:10].active = YES;
    apply_stats_visibility();

    NSNotificationCenter *nc = NSNotificationCenter.defaultCenter;
    void (^fsChanged)(NSNotification *) = ^(NSNotification *n) {
        (void)n;
        update_cursor();
        [[GraphicsSettingsUI shared] refresh];
    };
    [nc addObserverForName:NSWindowDidEnterFullScreenNotification object:gameWindow queue:nil usingBlock:fsChanged];
    [nc addObserverForName:NSWindowDidExitFullScreenNotification object:gameWindow queue:nil usingBlock:fsChanged];
    // Show the pointer while another window (e.g. a settings window) is in use.
    [nc addObserverForName:NSWindowDidResignKeyNotification object:gameWindow queue:nil
                usingBlock:^(NSNotification *n) { (void)n; if (gCursorHidden) { [NSCursor unhide]; gCursorHidden = NO; } }];
    [nc addObserverForName:NSWindowDidBecomeKeyNotification object:gameWindow queue:nil
                usingBlock:^(NSNotification *n) { (void)n; update_cursor(); }];

    if (g_gfx.startFullScreen)
        dispatch_async(dispatch_get_main_queue(), ^{ if (!is_full_screen()) [gameWindow toggleFullScreen:nil]; });
}

NSView *graphics_settings_view(void) { return [[GraphicsSettingsUI shared] settingsView]; }
void graphics_toggle_full_screen(void) { [gGameWindow toggleFullScreen:nil]; }
