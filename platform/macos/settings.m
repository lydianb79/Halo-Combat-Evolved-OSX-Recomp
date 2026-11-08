// settings.m — the "Settings" menu (Settings…, Full Screen) and the Settings
// window, a toolbar-tabbed window holding the Input page (input.m), the
// Graphics page (graphics.m) and the Audio page (audio.m).
#import <Cocoa/Cocoa.h>

#import "audio.h"
#import "graphics.h"
#import "input.h"
#import "settings.h"

@interface SettingsController : NSObject <NSWindowDelegate, NSMenuDelegate>
@property(nonatomic, strong) NSWindow *window;
+ (instancetype)shared;
@end

@implementation SettingsController

+ (instancetype)shared {
    static SettingsController *s;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ s = [SettingsController new]; });
    return s;
}

static NSTabViewItem *page(NSString *title, NSString *symbol, NSView *view) {
    NSViewController *vc = [NSViewController new];
    vc.view = view;
    vc.title = title;
    vc.preferredContentSize = view.frame.size;
    NSTabViewItem *item = [NSTabViewItem tabViewItemWithViewController:vc];
    item.label = title;
    if (@available(macOS 11.0, *)) item.image = [NSImage imageWithSystemSymbolName:symbol accessibilityDescription:title];
    return item;
}

- (void)show:(id)sender {
    (void)sender;
    if (!self.window) {
        NSTabViewController *tabs = [NSTabViewController new];
        tabs.tabStyle = NSTabViewControllerTabStyleToolbar;
        [tabs addTabViewItem:page(@"Input", @"gamecontroller", input_settings_view())];
        [tabs addTabViewItem:page(@"Graphics", @"display", graphics_settings_view())];
        [tabs addTabViewItem:page(@"Audio", @"speaker.wave.2", audio_settings_view())];
        self.window = [NSWindow windowWithContentViewController:tabs];
        self.window.styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable;
        if (@available(macOS 11.0, *)) self.window.toolbarStyle = NSWindowToolbarStylePreference;
        self.window.releasedWhenClosed = NO;
        self.window.delegate = self;
        [self.window center];
    } else {
        input_settings_view();       // refresh the pages
        graphics_settings_view();
        audio_settings_view();
    }
    [self.window makeKeyAndOrderFront:nil];
}

- (void)windowWillClose:(NSNotification *)n {
    (void)n;
    input_settings_will_close();
}

- (void)toggleFullScreen:(id)sender {
    (void)sender;
    graphics_toggle_full_screen();
}

- (void)menuNeedsUpdate:(NSMenu *)menu {
    for (NSMenuItem *it in menu.itemArray)
        if (it.action == @selector(toggleFullScreen:))
            it.state = graphics_is_full_screen() ? NSControlStateValueOn : NSControlStateValueOff;
}

@end

void settings_add_menu(NSMenu *bar) {
    SettingsController *sc = [SettingsController shared];
    NSMenuItem *item = [NSMenuItem new];
    NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Settings"];
    menu.delegate = sc;
    NSMenuItem *open = [menu addItemWithTitle:@"Settings…" action:@selector(show:) keyEquivalent:@","];
    open.target = sc;
    NSMenuItem *fs = [menu addItemWithTitle:@"Full Screen" action:@selector(toggleFullScreen:) keyEquivalent:@"f"];
    fs.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagControl;
    fs.target = sc;
    item.submenu = menu;
    [bar addItem:item];
}
