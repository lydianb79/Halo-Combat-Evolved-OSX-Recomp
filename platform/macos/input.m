// input.m — controller and keyboard input for the macOS shell.
//
// Every Xbox control can be bound to two keyboard keys or mouse buttons and to
// one element of a game controller (Input page of Settings → Settings…).
// The mapping is stored in the user defaults. The game samples input once per
// game tick, so a press that starts and ends between two samples is latched
// and reported once.
#import <Carbon/Carbon.h>
#import <Cocoa/Cocoa.h>
#import <GameController/GameController.h>
#include <math.h>
#include <stdatomic.h>

#import "input.h"
#import "graphics.h"

// ----------------------------------------------------------------- controls
enum { K_DIGITAL, K_ANALOG, K_AXIS_POS, K_AXIS_NEG };
typedef struct {
    const char *name;
    int kind, index;          // digital bit / analog button / stick axis
    int key1, key2, pad;      // defaults
} Control;

// Binding codes: 0..127 key codes (modifier keys included), 300 + n mouse button n.
#define NONE -1
#define MOUSE(n) (300 + (n))
#define NCODES 340

// Controller elements a control can be mapped to (index into kPadNames).
enum { P_NONE, P_A, P_B, P_X, P_Y, P_LB, P_RB, P_LT, P_RT, P_MENU, P_OPTIONS, P_L3, P_R3,
       P_DUP, P_DDOWN, P_DLEFT, P_DRIGHT, P_LSUP, P_LSDOWN, P_LSLEFT, P_LSRIGHT,
       P_RSUP, P_RSDOWN, P_RSLEFT, P_RSRIGHT, P_COUNT };
static NSString *const kPadNames[P_COUNT] = {
    @"—", @"A / Cross", @"B / Circle", @"X / Square", @"Y / Triangle", @"Left shoulder", @"Right shoulder",
    @"Left trigger", @"Right trigger", @"Menu / Start", @"View / Options", @"Left stick click",
    @"Right stick click", @"D-pad up", @"D-pad down", @"D-pad left", @"D-pad right",
    @"Left stick up", @"Left stick down", @"Left stick left", @"Left stick right",
    @"Right stick up", @"Right stick down", @"Right stick left", @"Right stick right",
};

static const Control kControls[] = {
    {"D-pad up",            K_DIGITAL, 0, 126, NONE, P_DUP},
    {"D-pad down",          K_DIGITAL, 1, 125, NONE, P_DDOWN},
    {"D-pad left",          K_DIGITAL, 2, 123, NONE, P_DLEFT},
    {"D-pad right",         K_DIGITAL, 3, 124, NONE, P_DRIGHT},
    {"START",               K_DIGITAL, 4, 35, NONE, P_MENU},          // P
    {"BACK",                K_DIGITAL, 5, 48, NONE, P_OPTIONS},       // Tab
    {"Left stick click",    K_DIGITAL, 6, 8, NONE, P_L3},             // C
    {"Right stick click",   K_DIGITAL, 7, 9, NONE, P_R3},             // V
    {"A",                   K_ANALOG, 0, 49, 36, P_A},                // Space, Return
    {"B",                   K_ANALOG, 1, 53, 51, P_B},                // Esc, Backspace
    {"X",                   K_ANALOG, 2, 15, NONE, P_X},              // R
    {"Y",                   K_ANALOG, 3, 12, NONE, P_Y},              // Q
    {"Black",               K_ANALOG, 4, 3, NONE, P_RB},              // F
    {"White",               K_ANALOG, 5, 14, NONE, P_LB},             // E
    {"Left trigger",        K_ANALOG, 6, 56, NONE, P_LT},             // Left Shift
    {"Right trigger",       K_ANALOG, 7, MOUSE(0), 59, P_RT},         // Left click, Left Control
    {"Move forward",        K_AXIS_POS, 1, 13, NONE, P_LSUP},         // W
    {"Move back",           K_AXIS_NEG, 1, 1, NONE, P_LSDOWN},        // S
    {"Strafe left",         K_AXIS_NEG, 0, 0, NONE, P_LSLEFT},        // A
    {"Strafe right",        K_AXIS_POS, 0, 2, NONE, P_LSRIGHT},       // D
    {"Look up",             K_AXIS_POS, 3, 126, 34, P_RSUP},          // Up, I
    {"Look down",           K_AXIS_NEG, 3, 125, 40, P_RSDOWN},        // Down, K
    {"Look left",           K_AXIS_NEG, 2, 123, 38, P_RSLEFT},        // Left, J
    {"Look right",          K_AXIS_POS, 2, 124, 37, P_RSRIGHT},       // Right, L
};
#define NCTL ((int)(sizeof kControls / sizeof kControls[0]))

static NSString *const kSectionTitles[] = {@"Buttons", @"Movement (left stick)", @"Look (right stick)"};
static int section_of(int c) { return c < 16 ? 0 : c < 20 ? 1 : 2; }

// ------------------------------------------------------------------- state
static int gKey1[NCTL], gKey2[NCTL], gPad[NCTL];
static BOOL gInvertLook;
static double gDeadZone = 0.15;
static bool gCodeDown[NCODES];
static _Atomic uint32_t gCtlDown, gCtlLatched;     // bit c = control c held / pressed since last poll
static __weak NSWindow *gGameWindow;

static NSString *const kDefaultsBindings = @"HaloInput.bindings.v1";
static NSString *const kDefaultsInvert = @"HaloInput.invertLook";
static NSString *const kDefaultsDeadZone = @"HaloInput.deadZone";

static void reset_defaults(void) {
    for (int c = 0; c < NCTL; c++) { gKey1[c] = kControls[c].key1; gKey2[c] = kControls[c].key2; gPad[c] = kControls[c].pad; }
    gInvertLook = NO;
    gDeadZone = 0.15;
}

static void save_settings(void) {
    NSMutableArray *a = [NSMutableArray array];
    for (int c = 0; c < NCTL; c++) [a addObject:@[@(gKey1[c]), @(gKey2[c]), @(gPad[c])]];
    NSUserDefaults *d = NSUserDefaults.standardUserDefaults;
    [d setObject:a forKey:kDefaultsBindings];
    [d setBool:gInvertLook forKey:kDefaultsInvert];
    [d setDouble:gDeadZone forKey:kDefaultsDeadZone];
}

static int valid_code(id n) {
    int v = [n isKindOfClass:NSNumber.class] ? [n intValue] : NONE;
    return (v >= 0 && v < NCODES) ? v : NONE;
}

static void load_settings(void) {
    reset_defaults();
    NSUserDefaults *d = NSUserDefaults.standardUserDefaults;
    NSArray *a = [d arrayForKey:kDefaultsBindings];
    if (a.count == (NSUInteger)NCTL) {
        for (int c = 0; c < NCTL; c++) {
            NSArray *e = a[c];
            if (![e isKindOfClass:NSArray.class] || e.count != 3) continue;
            gKey1[c] = valid_code(e[0]);
            gKey2[c] = valid_code(e[1]);
            int p = [e[2] isKindOfClass:NSNumber.class] ? [e[2] intValue] : P_NONE;
            gPad[c] = (p >= 0 && p < P_COUNT) ? p : P_NONE;
        }
    }
    gInvertLook = [d boolForKey:kDefaultsInvert];
    if ([d objectForKey:kDefaultsDeadZone]) gDeadZone = fmin(fmax([d doubleForKey:kDefaultsDeadZone], 0.0), 0.5);
}

// Recompute which controls are held after a key/mouse change; latch new presses.
static void update_controls(void) {
    uint32_t down = 0;
    for (int c = 0; c < NCTL; c++)
        if ((gKey1[c] != NONE && gCodeDown[gKey1[c]]) || (gKey2[c] != NONE && gCodeDown[gKey2[c]])) down |= 1u << c;
    uint32_t old = atomic_exchange(&gCtlDown, down);
    atomic_fetch_or(&gCtlLatched, down & ~old);
}

static void release_all(void) {
    memset(gCodeDown, 0, sizeof gCodeDown);
    update_controls();
}

// ---------------------------------------------------------- key names
static NSString *key_name(int code) {
    if (code == NONE) return @"—";
    if (code >= MOUSE(0)) {
        int b = code - MOUSE(0);
        return b == 0 ? @"Left click" : b == 1 ? @"Right click" : b == 2 ? @"Middle click"
                                                                    : [NSString stringWithFormat:@"Mouse button %d", b + 1];
    }
    switch (code) {
    case 36: return @"Return";        case 76: return @"Enter";         case 48: return @"Tab";
    case 49: return @"Space";         case 51: return @"Backspace";     case 117: return @"Forward Delete";
    case 53: return @"Esc";           case 115: return @"Home";         case 119: return @"End";
    case 116: return @"Page Up";      case 121: return @"Page Down";    case 71: return @"Clear";
    case 123: return @"←";            case 124: return @"→";            case 125: return @"↓";
    case 126: return @"↑";            case 56: return @"Left Shift";    case 60: return @"Right Shift";
    case 59: return @"Left Control";  case 62: return @"Right Control"; case 58: return @"Left Option";
    case 61: return @"Right Option";  case 57: return @"Caps Lock";     case 63: return @"Fn";
    case 122: return @"F1";  case 120: return @"F2";  case 99: return @"F3";   case 118: return @"F4";
    case 96: return @"F5";   case 97: return @"F6";   case 98: return @"F7";   case 100: return @"F8";
    case 101: return @"F9";  case 109: return @"F10"; case 103: return @"F11"; case 111: return @"F12";
    default: break;
    }
    // Everything else: the character this key produces on the current keyboard layout.
    NSString *name = nil;
    TISInputSourceRef src = TISCopyCurrentKeyboardLayoutInputSource();
    if (src) {
        CFDataRef data = (CFDataRef)TISGetInputSourceProperty(src, kTISPropertyUnicodeKeyLayoutData);
        if (data) {
            const UCKeyboardLayout *layout = (const UCKeyboardLayout *)CFDataGetBytePtr(data);
            UInt32 dead = 0;
            UniChar chars[4];
            UniCharCount len = 0;
            if (UCKeyTranslate(layout, (UInt16)code, kUCKeyActionDisplay, 0, LMGetKbdType(),
                               kUCKeyTranslateNoDeadKeysBit, &dead, 4, &len, chars) == noErr && len > 0)
                name = [[NSString stringWithCharacters:chars length:len] uppercaseString];
        }
        CFRelease(src);
    }
    BOOL keypad = (code >= 65 && code <= 92 && code != 66 && code != 68 && code != 70 && code != 72 && code != 73 &&
                   code != 74 && code != 77 && code != 79 && code != 80 && code != 90);
    if (name.length && keypad) return [@"Keypad " stringByAppendingString:name];
    if (name.length) return name;
    return [NSString stringWithFormat:@"Key %d", code];
}

// Device-dependent modifier bits (IOKit NX_DEVICE*KEYMASK) tell left from right.
static bool modifier_down(int code, NSEventModifierFlags f) {
    switch (code) {
    case 56: return (f & 0x00000002) != 0;     // left shift
    case 60: return (f & 0x00000004) != 0;     // right shift
    case 59: return (f & 0x00000001) != 0;     // left control
    case 62: return (f & 0x00002000) != 0;     // right control
    case 58: return (f & 0x00000020) != 0;     // left option
    case 61: return (f & 0x00000040) != 0;     // right option
    case 57: return (f & NSEventModifierFlagCapsLock) != 0;
    case 63: return (f & NSEventModifierFlagFunction) != 0;
    default: return false;
    }
}
static const int kModifierCodes[] = {56, 60, 59, 62, 58, 61, 57, 63};

// --------------------------------------------------------- controller
static float pad_value(GCExtendedGamepad *g, int el) {
    switch (el) {
    case P_A: return g.buttonA.value;
    case P_B: return g.buttonB.value;
    case P_X: return g.buttonX.value;
    case P_Y: return g.buttonY.value;
    case P_LB: return g.leftShoulder.value;
    case P_RB: return g.rightShoulder.value;
    case P_LT: return g.leftTrigger.value;
    case P_RT: return g.rightTrigger.value;
    case P_MENU: return g.buttonMenu.value;
    case P_OPTIONS: return g.buttonOptions.value;
    case P_L3: return g.leftThumbstickButton.value;
    case P_R3: return g.rightThumbstickButton.value;
    case P_DUP: return g.dpad.up.value;
    case P_DDOWN: return g.dpad.down.value;
    case P_DLEFT: return g.dpad.left.value;
    case P_DRIGHT: return g.dpad.right.value;
    case P_LSUP: return fmaxf(g.leftThumbstick.yAxis.value, 0);
    case P_LSDOWN: return fmaxf(-g.leftThumbstick.yAxis.value, 0);
    case P_LSLEFT: return fmaxf(-g.leftThumbstick.xAxis.value, 0);
    case P_LSRIGHT: return fmaxf(g.leftThumbstick.xAxis.value, 0);
    case P_RSUP: return fmaxf(g.rightThumbstick.yAxis.value, 0);
    case P_RSDOWN: return fmaxf(-g.rightThumbstick.yAxis.value, 0);
    case P_RSLEFT: return fmaxf(-g.rightThumbstick.xAxis.value, 0);
    case P_RSRIGHT: return fmaxf(g.rightThumbstick.xAxis.value, 0);
    default: return 0;
    }
}

static void hook_controller(GCController *pad) {
    if (!pad.extendedGamepad) return;
    pad.extendedGamepad.valueChangedHandler = ^(GCExtendedGamepad *g, GCControllerElement *el) {
        (void)el;
        uint32_t latch = 0;
        for (int c = 0; c < 16; c++)                       // buttons only; sticks are sampled continuously
            if (gPad[c] != P_NONE && pad_value(g, gPad[c]) > 0.5f) latch |= 1u << c;
        atomic_fetch_or(&gCtlLatched, latch);
    };
}

// ------------------------------------------------------------- polling
void input_poll(void *user, uint16_t *buttons, uint8_t analog[8], int16_t axes[4]) {
    (void)user;
    uint32_t k = atomic_load(&gCtlDown) | atomic_exchange(&gCtlLatched, 0);
    GCController *pad = GCController.current ?: GCController.controllers.firstObject;
    GCExtendedGamepad *g = pad.extendedGamepad;
    float val[NCTL];
    for (int c = 0; c < NCTL; c++) {
        float v = (k & (1u << c)) ? 1.0f : 0.0f;
        if (g && gPad[c] != P_NONE) v = fmaxf(v, pad_value(g, gPad[c]));
        val[c] = v;
    }
    *buttons = 0;
    memset(analog, 0, 8);
    float ax[4] = {0, 0, 0, 0};
    for (int c = 0; c < NCTL; c++) {
        const Control *ct = &kControls[c];
        switch (ct->kind) {
        case K_DIGITAL: if (val[c] > 0.5f) *buttons |= (uint16_t)(1u << ct->index); break;
        case K_ANALOG: analog[ct->index] = (uint8_t)fminf(val[c] * 255.0f + 0.5f, 255.0f); break;
        case K_AXIS_POS: ax[ct->index] += val[c]; break;
        case K_AXIS_NEG: ax[ct->index] -= val[c]; break;
        }
    }
    float dz = (float)gDeadZone;
    for (int i = 0; i < 4; i++) {
        float v = fmaxf(-1.0f, fminf(1.0f, ax[i]));
        float m = fabsf(v);
        v = m <= dz ? 0.0f : copysignf((m - dz) / (1.0f - dz), v);
        if (i == 3 && gInvertLook) v = -v;
        axes[i] = (int16_t)lrintf(v * 32767.0f);
    }
}

// ----------------------------------------------------- settings window
@interface InputFlippedView : NSView
@end
@implementation InputFlippedView
- (BOOL)isFlipped { return YES; }
@end

@interface InputSettings : NSObject
@property(nonatomic, strong) NSView *view;
@property(nonatomic, strong) NSMutableArray<NSButton *> *keyButtons;      // 2 per control
@property(nonatomic, strong) NSMutableArray<NSPopUpButton *> *padPopups;
@property(nonatomic, strong) NSButton *invertBox;
@property(nonatomic, strong) NSSlider *deadZoneSlider;
@property(nonatomic, strong) NSTextField *deadZoneLabel;
@property(nonatomic) NSInteger captureTag;                                // -1: not capturing
+ (instancetype)shared;
@end

@implementation InputSettings

+ (instancetype)shared {
    static InputSettings *s;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ s = [InputSettings new]; s.captureTag = -1; });
    return s;
}

static NSTextField *make_label(NSString *text, BOOL bold) {
    NSTextField *l = [NSTextField labelWithString:text];
    if (bold) l.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
    return l;
}

- (void)buildView {
    self.keyButtons = [NSMutableArray array];
    self.padPopups = [NSMutableArray array];
    NSMutableArray *rows = [NSMutableArray array];
    [rows addObject:@[make_label(@"Xbox control", YES), make_label(@"Key", YES), make_label(@"Alternate key", YES),
                      make_label(@"Controller", YES)]];
    int section = -1;
    for (int c = 0; c < NCTL; c++) {
        if (section_of(c) != section) {
            section = section_of(c);
            NSTextField *t = make_label(kSectionTitles[section], YES);
            t.textColor = NSColor.secondaryLabelColor;
            [rows addObject:@[t, NSGridCell.emptyContentView, NSGridCell.emptyContentView, NSGridCell.emptyContentView]];
        }
        NSButton *b1 = [NSButton buttonWithTitle:@"" target:self action:@selector(keyClicked:)];
        NSButton *b2 = [NSButton buttonWithTitle:@"" target:self action:@selector(keyClicked:)];
        b1.tag = c * 2; b2.tag = c * 2 + 1;
        for (NSButton *b in @[b1, b2]) {
            [b.widthAnchor constraintEqualToConstant:150].active = YES;
            [self.keyButtons addObject:b];
        }
        NSPopUpButton *p = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
        for (int e = 0; e < P_COUNT; e++) [p addItemWithTitle:kPadNames[e]];
        p.tag = c;
        p.target = self;
        p.action = @selector(padChanged:);
        [p.widthAnchor constraintEqualToConstant:170].active = YES;
        [self.padPopups addObject:p];
        [rows addObject:@[make_label([NSString stringWithUTF8String:kControls[c].name], NO), b1, b2, p]];
    }
    NSGridView *grid = [NSGridView gridViewWithViews:rows];
    grid.rowSpacing = 6;
    grid.columnSpacing = 12;
    grid.yPlacement = NSGridCellPlacementCenter;
    NSSize gs = grid.fittingSize;

    InputFlippedView *doc = [[InputFlippedView alloc] initWithFrame:NSMakeRect(0, 0, gs.width + 40, gs.height + 32)];
    grid.translatesAutoresizingMaskIntoConstraints = YES;
    grid.frame = NSMakeRect(20, 16, gs.width, gs.height);
    [doc addSubview:grid];

    CGFloat footer = 120, width = gs.width + 40, height = fmin(gs.height + 32, 560) + footer;
    NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
    self.view = content;
    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(0, footer, width, height - footer)];
    scroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    scroll.hasVerticalScroller = YES;
    scroll.drawsBackground = NO;
    scroll.documentView = doc;
    [content addSubview:scroll];

    NSTextField *help = make_label(@"Click a key field, then press a key or mouse button. Click the field again while it "
                                   @"waits to clear it. ⌘ shortcuts stay with the menus.", NO);
    help.textColor = NSColor.secondaryLabelColor;
    help.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    help.frame = NSMakeRect(20, 86, width - 40, 18);
    [content addSubview:help];

    self.invertBox = [NSButton checkboxWithTitle:@"Invert look (up/down)" target:self action:@selector(optionsChanged:)];
    self.invertBox.frame = NSMakeRect(20, 52, 220, 22);
    [content addSubview:self.invertBox];

    NSTextField *dzTitle = make_label(@"Stick dead zone", NO);
    dzTitle.frame = NSMakeRect(260, 54, 110, 18);
    [content addSubview:dzTitle];
    self.deadZoneSlider = [NSSlider sliderWithValue:gDeadZone minValue:0 maxValue:0.5 target:self
                                             action:@selector(optionsChanged:)];
    self.deadZoneSlider.frame = NSMakeRect(370, 52, 160, 22);
    [content addSubview:self.deadZoneSlider];
    self.deadZoneLabel = make_label(@"", NO);
    self.deadZoneLabel.frame = NSMakeRect(538, 54, 50, 18);
    [content addSubview:self.deadZoneLabel];

    NSButton *restore = [NSButton buttonWithTitle:@"Restore Defaults" target:self action:@selector(restoreDefaults:)];
    restore.frame = NSMakeRect(14, 12, 150, 32);
    [content addSubview:restore];
    NSButton *done = [NSButton buttonWithTitle:@"Done" target:nil action:@selector(performClose:)];
    done.frame = NSMakeRect(width - 104, 12, 90, 32);
    done.autoresizingMask = NSViewMinXMargin;
    [content addSubview:done];

    [self refresh];
    // Scroll to the top of the list.
    [doc scrollPoint:NSMakePoint(0, 0)];
}

- (void)refresh {
    for (NSButton *b in self.keyButtons) {
        int c = (int)(b.tag / 2);
        int code = (b.tag & 1) ? gKey2[c] : gKey1[c];
        b.title = (b.tag == self.captureTag) ? @"Press a key…" : key_name(code);
    }
    for (NSPopUpButton *p in self.padPopups) [p selectItemAtIndex:gPad[p.tag]];
    self.invertBox.state = gInvertLook ? NSControlStateValueOn : NSControlStateValueOff;
    self.deadZoneSlider.doubleValue = gDeadZone;
    self.deadZoneLabel.stringValue = [NSString stringWithFormat:@"%d%%", (int)lround(gDeadZone * 100)];
}

- (NSView *)settingsView {
    if (!self.view) [self buildView];
    [self refresh];
    return self.view;
}

- (void)setBinding:(NSInteger)tag code:(int)code {
    int c = (int)(tag / 2);
    if (tag & 1) gKey2[c] = code; else gKey1[c] = code;
    save_settings();
    update_controls();
}

- (void)keyClicked:(NSButton *)b {
    if (self.captureTag == b.tag) {                 // second click while waiting: clear
        self.captureTag = -1;
        [self setBinding:b.tag code:NONE];
    } else {
        self.captureTag = b.tag;
    }
    [self refresh];
}

- (void)finishCapture:(int)code {
    NSInteger tag = self.captureTag;
    self.captureTag = -1;
    if (tag >= 0) [self setBinding:tag code:code];
    [self refresh];
}

- (NSButton *)captureButton {
    for (NSButton *b in self.keyButtons) if (b.tag == self.captureTag) return b;
    return nil;
}

- (void)padChanged:(NSPopUpButton *)p {
    gPad[p.tag] = (int)p.indexOfSelectedItem;
    save_settings();
}

- (void)optionsChanged:(id)sender {
    (void)sender;
    gInvertLook = self.invertBox.state == NSControlStateValueOn;
    gDeadZone = round(self.deadZoneSlider.doubleValue * 100) / 100;
    save_settings();
    [self refresh];
}

- (void)restoreDefaults:(id)sender {
    (void)sender;
    self.captureTag = -1;
    reset_defaults();
    save_settings();
    update_controls();
    [self refresh];
}

- (void)settingsWillClose {
    self.captureTag = -1;
    [self refresh];
}

// Called for every key/mouse event in the app. Returns the event to let it
// through, or nil to consume it.
- (NSEvent *)handleEvent:(NSEvent *)e {
    BOOL commandHeld = (e.modifierFlags & NSEventModifierFlagCommand) != 0;
    if (self.captureTag >= 0) {
        switch (e.type) {
        case NSEventTypeKeyDown:
            if (commandHeld) return e;
            if (!e.isARepeat) [self finishCapture:e.keyCode];
            return nil;
        case NSEventTypeKeyUp: return commandHeld ? e : nil;
        case NSEventTypeFlagsChanged:
            if (e.keyCode != 54 && e.keyCode != 55 && modifier_down(e.keyCode, e.modifierFlags))
                [self finishCapture:e.keyCode];
            return e;
        case NSEventTypeLeftMouseDown: {
            NSButton *b = [self captureButton];
            if (b && e.window == b.window && NSPointInRect([b convertPoint:e.locationInWindow fromView:nil], b.bounds))
                return e;                                                   // the click that clears
            [self finishCapture:MOUSE(0)];
            return nil;
        }
        case NSEventTypeRightMouseDown: [self finishCapture:MOUSE(1)]; return nil;
        case NSEventTypeOtherMouseDown: [self finishCapture:MOUSE((int)fmin(e.buttonNumber, 31))]; return nil;
        default: return e;
        }
    }

    // Normal play: only events aimed at the game window drive the controls.
    NSWindow *game = gGameWindow;
    if (!game || e.window != game || !game.isKeyWindow) return e;
    if (e.keyCode == 53 && !commandHeld && g_gfx.escExitsFullScreen && graphics_is_full_screen() &&
        (e.type == NSEventTypeKeyDown || e.type == NSEventTypeKeyUp)) {
        if (e.type == NSEventTypeKeyDown && !e.isARepeat) [game toggleFullScreen:nil];    // Esc leaves full screen
        return nil;
    }
    switch (e.type) {
    case NSEventTypeKeyDown:
    case NSEventTypeKeyUp:
        if (commandHeld) return e;                                          // ⌘Q, ⌘M, ⌘, …
        if (e.keyCode < 128) gCodeDown[e.keyCode] = e.type == NSEventTypeKeyDown;
        update_controls();
        return nil;                                                         // no system beep
    case NSEventTypeFlagsChanged:
        for (size_t i = 0; i < sizeof kModifierCodes / sizeof kModifierCodes[0]; i++)
            gCodeDown[kModifierCodes[i]] = modifier_down(kModifierCodes[i], e.modifierFlags);
        update_controls();
        return e;
    case NSEventTypeLeftMouseDown: case NSEventTypeRightMouseDown: case NSEventTypeOtherMouseDown:
    case NSEventTypeLeftMouseUp: case NSEventTypeRightMouseUp: case NSEventTypeOtherMouseUp: {
        BOOL down = e.type == NSEventTypeLeftMouseDown || e.type == NSEventTypeRightMouseDown ||
                    e.type == NSEventTypeOtherMouseDown;
        NSView *v = game.contentView;
        if (down && !NSPointInRect([v convertPoint:e.locationInWindow fromView:nil], v.bounds)) return e;  // title bar
        int b = (int)fmin(e.buttonNumber, 31);
        gCodeDown[MOUSE(b)] = down;
        update_controls();
        return e;
    }
    default: return e;
    }
}

@end

// ------------------------------------------------------------------- setup
void input_install(NSWindow *gameWindow) {
    gGameWindow = gameWindow;
    load_settings();
    NSEventMask mask = NSEventMaskKeyDown | NSEventMaskKeyUp | NSEventMaskFlagsChanged | NSEventMaskLeftMouseDown |
                       NSEventMaskLeftMouseUp | NSEventMaskRightMouseDown | NSEventMaskRightMouseUp |
                       NSEventMaskOtherMouseDown | NSEventMaskOtherMouseUp;
    [NSEvent addLocalMonitorForEventsMatchingMask:mask handler:^NSEvent *(NSEvent *e) {
        return [[InputSettings shared] handleEvent:e];
    }];
    // Keys released while the game window is not focused never send key-up.
    NSNotificationCenter *nc = NSNotificationCenter.defaultCenter;
    [nc addObserverForName:NSWindowDidResignKeyNotification object:gameWindow queue:nil
                usingBlock:^(NSNotification *n) { (void)n; release_all(); }];
    [nc addObserverForName:NSApplicationDidResignActiveNotification object:nil queue:nil
                usingBlock:^(NSNotification *n) { (void)n; release_all(); }];
    for (GCController *pad in GCController.controllers) hook_controller(pad);
    [nc addObserverForName:GCControllerDidConnectNotification object:nil queue:nil
                usingBlock:^(NSNotification *n) { hook_controller(n.object); }];
}

NSView *input_settings_view(void) { return [[InputSettings shared] settingsView]; }
void input_settings_will_close(void) { [[InputSettings shared] settingsWillClose]; }
