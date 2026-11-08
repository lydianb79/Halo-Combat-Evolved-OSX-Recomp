// audio.h — sound output (AVAudioEngine) and the Audio settings page.
#pragma once
#import <Cocoa/Cocoa.h>

// Start sound output; call once the game is running.
void audio_install(void);
// The "Audio" page of the Settings window.
NSView *audio_settings_view(void);
