// audio.m — sound output for the macOS shell, and the Audio page of the
// Settings window.
//
// The emulated APU (runtime/apu.c) mixes the game's voices to 48 kHz stereo
// into a ring buffer, paced by the game clock. An AVAudioSourceNode pulls from
// that ring on Core Audio's thread; the engine converts to the output device's
// rate. A cushion of ~80 ms absorbs the bursty production; if the ring runs
// far ahead (device clock slower than ours) the excess is skipped.
#import <AVFoundation/AVFoundation.h>
#import <Cocoa/Cocoa.h>
#include <math.h>
#include <string.h>
#include <stdatomic.h>

#import "audio.h"
#include "../../runtime/runtime.h"
#include "../../runtime/gpdsp.h"

static AVAudioEngine *gEngine;
static _Atomic float gVolume = 1.0f;          // 0..1, after mute and background handling
static float gUserVolume = 0.8f;
static BOOL gMuted, gMuteInBackground = YES, gAppActive = YES;

static NSString *const kVolumeKey = @"HaloAudio.volume";
static NSString *const kMutedKey = @"HaloAudio.muted";
static NSString *const kBackgroundKey = @"HaloAudio.muteInBackground";
static NSString *const kReverbKey = @"HaloAudio.reverb";
static float gReverb = 1.0f;                     // 0..4 (shown as 0-400%)
static NSString *const kEffectsKey = @"HaloAudio.effects";
static int gEffects = 0;                         // 0: Xbox audio DSP, 1: built-in reverb

static BOOL dsp_possible(void) {
    const char *e = getenv("HALO_DSP");
    return gpdsp_compiled() && !(e && *e == '0');
}

static void apply_volume(void) {
    float v = gMuted || (gMuteInBackground && !gAppActive) ? 0.0f : gUserVolume;
    atomic_store(&gVolume, v);
}

static void save(void) {
    NSUserDefaults *d = NSUserDefaults.standardUserDefaults;
    [d setFloat:gUserVolume forKey:kVolumeKey];
    [d setBool:gMuted forKey:kMutedKey];
    [d setBool:gMuteInBackground forKey:kBackgroundKey];
    [d setFloat:gReverb forKey:kReverbKey];
    [d setInteger:gEffects forKey:kEffectsKey];
}

// Linear up to 0.75, then a smooth knee towards 1: the sum of many voices
// can exceed full scale.
static inline float soft_limit(float x) {
    float m = fabsf(x);
    if (m <= 0.75f) return x;
    m = 0.75f + 0.25f * tanhf((m - 0.75f) * 4.0f);
    return x < 0 ? -m : m;
}

void audio_install(void) {
    NSUserDefaults *d = NSUserDefaults.standardUserDefaults;
    if ([d objectForKey:kVolumeKey]) gUserVolume = [d floatForKey:kVolumeKey];
    gMuted = [d boolForKey:kMutedKey];
    if ([d objectForKey:kBackgroundKey]) gMuteInBackground = [d boolForKey:kBackgroundKey];
    if ([d objectForKey:kReverbKey]) gReverb = [d floatForKey:kReverbKey];
    apu_set_reverb_level(gReverb);
    gEffects = (int)[d integerForKey:kEffectsKey];
    apu_set_dsp_effects(gEffects == 0);
    apply_volume();

    [NSNotificationCenter.defaultCenter addObserverForName:NSApplicationDidBecomeActiveNotification object:nil
                                                     queue:nil usingBlock:^(NSNotification *n) {
        (void)n; gAppActive = YES; apply_volume();
    }];
    [NSNotificationCenter.defaultCenter addObserverForName:NSApplicationDidResignActiveNotification object:nil
                                                     queue:nil usingBlock:^(NSNotification *n) {
        (void)n; gAppActive = NO; apply_volume();
    }];

    gEngine = [AVAudioEngine new];
    AVAudioFormat *fmt = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:48000 channels:2];  // float, planar
    AVAudioSourceNode *src = [[AVAudioSourceNode alloc] initWithFormat:fmt renderBlock:
        ^OSStatus(BOOL *silence, const AudioTimeStamp *ts, AVAudioFrameCount frames, AudioBufferList *out) {
            (void)ts;
            static float tmp[4096 * 2];
            if (frames > 4096) frames = 4096;
            // The game produces sound in bursts (whenever the emulator's ticker
            // gets to run), so keep a cushion of ~80 ms: after running dry, wait
            // until the cushion is back before playing again; if the ring grows
            // past ~250 ms (device clock slower than ours), skip the excess.
            static int priming = 1;
            int buffered = apu_audio_buffered();
            if (buffered > 12000) {
                int drop = buffered - 4800;
                while (drop > 0) { int k = drop > 4096 ? 4096 : drop; apu_pull_audio(tmp, k); drop -= k; }
                buffered = 4800;
            }
            if (priming && buffered < 3840) {
                memset(tmp, 0, frames * 2 * sizeof(float));
                for (AVAudioFrameCount i = 0; i < frames; i++) {
                    ((float *)out->mBuffers[0].mData)[i] = 0;
                    if (out->mNumberBuffers > 1) ((float *)out->mBuffers[1].mData)[i] = 0;
                }
                *silence = YES;
                return noErr;
            }
            priming = 0;
            int got = apu_pull_audio(tmp, (int)frames);
            if (got < (int)frames) priming = 1;          // ran dry: rebuild the cushion
            float v = atomic_load(&gVolume);
            float *l = out->mBuffers[0].mData, *r = out->mNumberBuffers > 1 ? out->mBuffers[1].mData : NULL;
            for (AVAudioFrameCount i = 0; i < frames; i++) {
                float a = tmp[i * 2] * v, b = tmp[i * 2 + 1] * v;
                l[i] = soft_limit(a);
                if (r) r[i] = soft_limit(b);
            }
            *silence = got == 0;
            return noErr;
        }];
    [gEngine attachNode:src];
    NSError *err = nil;
    if (@available(macOS 27.0, *)) {
        if (![gEngine connect:src to:gEngine.mainMixerNode format:fmt error:&err]) {
            NSLog(@"audio: could not connect the output: %@", err);
            return;
        }
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        [gEngine connect:src to:gEngine.mainMixerNode format:fmt];
#pragma clang diagnostic pop
    }
    if (![gEngine startAndReturnError:&err]) NSLog(@"audio: could not start output: %@", err);
}

// ---------------------------------------------------------------- settings page
@interface AudioSettingsUI : NSObject
@property(nonatomic, strong) NSView *view;
@property(nonatomic, strong) NSSlider *volume;
@property(nonatomic, strong) NSTextField *volumeLabel;
@property(nonatomic, strong) NSButton *mute, *background;
@property(nonatomic, strong) NSSlider *reverb;
@property(nonatomic, strong) NSPopUpButton *effects;
@property(nonatomic, strong) NSTextField *reverbLabel;
@end

@implementation AudioSettingsUI

+ (instancetype)shared {
    static AudioSettingsUI *s;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ s = [AudioSettingsUI new]; });
    return s;
}

static NSTextField *label(NSString *t) {
    NSTextField *l = [NSTextField labelWithString:t];
    l.alignment = NSTextAlignmentRight;
    return l;
}

- (void)build {
    self.volume = [NSSlider sliderWithValue:gUserVolume minValue:0 maxValue:1 target:self action:@selector(changed:)];
    [self.volume.widthAnchor constraintEqualToConstant:200].active = YES;
    self.volumeLabel = [NSTextField labelWithString:@""];
    [self.volumeLabel.widthAnchor constraintEqualToConstant:50].active = YES;
    NSStackView *row = [NSStackView stackViewWithViews:@[self.volume, self.volumeLabel]];
    row.spacing = 8;
    self.reverb = [NSSlider sliderWithValue:gReverb minValue:0 maxValue:4 target:self action:@selector(changed:)];
    [self.reverb.widthAnchor constraintEqualToConstant:200].active = YES;
    self.reverbLabel = [NSTextField labelWithString:@""];
    [self.reverbLabel.widthAnchor constraintEqualToConstant:50].active = YES;
    NSStackView *rrow = [NSStackView stackViewWithViews:@[self.reverb, self.reverbLabel]];
    rrow.spacing = 8;
    self.effects = [NSPopUpButton new];
    [self.effects addItemsWithTitles:@[@"Xbox audio DSP (accurate)", @"Built-in reverb (lighter)"]];
    self.effects.target = self;
    self.effects.action = @selector(changed:);
    self.mute = [NSButton checkboxWithTitle:@"Mute" target:self action:@selector(changed:)];
    self.background = [NSButton checkboxWithTitle:@"Mute while another app is in front" target:self action:@selector(changed:)];
    NSTextField *note = [NSTextField labelWithString:
        @"Effects: the Xbox audio DSP runs the game's own effects program (Microsoft's reverb and 3D\n"
         "processing). The built-in reverb is a lighter approximation that follows the game's rooms;\n"
         "the Reverb level applies to it only (100% is the default, 0% turns it off). Changes apply at once."];
    note.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    note.textColor = NSColor.secondaryLabelColor;
    NSView *empty = NSGridCell.emptyContentView;
    NSGridView *grid = [NSGridView gridViewWithViews:@[
        @[label(@"Volume:"), row],
        @[label(@"Effects:"), self.effects],
        @[label(@"Reverb:"), rrow],
        @[empty, self.mute],
        @[empty, self.background],
        @[empty, note],
    ]];
    grid.rowSpacing = 8;
    grid.columnSpacing = 10;
    [grid columnAtIndex:0].xPlacement = NSGridCellPlacementTrailing;
    NSSize gs = grid.fittingSize;
    CGFloat footer = 56, width = MAX(gs.width + 48, 420), height = gs.height + 40 + footer;
    NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
    grid.frame = NSMakeRect(24, footer + 16, gs.width, gs.height);
    [content addSubview:grid];
    NSButton *done = [NSButton buttonWithTitle:@"Done" target:nil action:@selector(performClose:)];
    done.frame = NSMakeRect(width - 104, 12, 90, 32);
    [content addSubview:done];
    self.view = content;
}

- (void)refresh {
    self.volume.doubleValue = gUserVolume;
    self.volumeLabel.stringValue = [NSString stringWithFormat:@"%d%%", (int)lround(gUserVolume * 100)];
    self.mute.state = gMuted ? NSControlStateValueOn : NSControlStateValueOff;
    BOOL possible = dsp_possible();
    self.effects.enabled = possible;
    [self.effects selectItemAtIndex:possible ? gEffects : 1];
    self.reverb.enabled = !possible || gEffects == 1;
    self.reverb.doubleValue = gReverb;
    self.reverbLabel.stringValue = [NSString stringWithFormat:@"%d%%", (int)lround(gReverb * 100)];
    self.background.state = gMuteInBackground ? NSControlStateValueOn : NSControlStateValueOff;
}

- (void)changed:(id)sender {
    if (sender == self.volume) gUserVolume = (float)self.volume.doubleValue;
    if (sender == self.mute) gMuted = self.mute.state == NSControlStateValueOn;
    if (sender == self.reverb) { gReverb = (float)self.reverb.doubleValue; apu_set_reverb_level(gReverb); }
    if (sender == self.effects) { gEffects = (int)self.effects.indexOfSelectedItem; apu_set_dsp_effects(gEffects == 0); }
    if (sender == self.background) gMuteInBackground = self.background.state == NSControlStateValueOn;
    apply_volume();
    save();
    [self refresh];
}

@end

NSView *audio_settings_view(void) {
    AudioSettingsUI *ui = [AudioSettingsUI shared];
    if (!ui.view) [ui build];
    [ui refresh];
    return ui.view;
}
