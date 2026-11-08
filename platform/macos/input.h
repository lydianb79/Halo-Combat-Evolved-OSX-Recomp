// input.h — controller and keyboard input for the macOS shell, with a
// configurable mapping (Input page of Settings → Settings…).
#pragma once
#import <Cocoa/Cocoa.h>
#include <stdint.h>

// Start listening to keyboard, mouse and game controllers. Keys only reach the
// game while `gameWindow` is the key window.
void input_install(NSWindow *gameWindow);

// The "Input" page of the Settings window, and its close notification.
NSView *input_settings_view(void);
void input_settings_will_close(void);

// g_host.poll_input callback: Xbox gamepad state for port 0.
void input_poll(void *user, uint16_t *buttons, uint8_t analog[8], int16_t axes[4]);
