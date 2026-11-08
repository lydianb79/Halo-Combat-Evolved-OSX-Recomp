/* apu.c — the Xbox audio hardware: MCPX APU (0xFE800000, 512 KB) and the
 * AC97 controller (0xFEC00000).
 *
 * What is modelled
 *   - The front end (FE): the PIO method interface DirectSound programs voices
 *     through (0xFE820000), including the "idle voice" trap DirectSound waits
 *     for after stopping a voice, and the APU interrupt.
 *   - The voice processor (VP): every active voice is read from guest memory
 *     (PCM 8/16 bit or Xbox ADPCM, mono or stereo, looping or one-shot),
 *     resampled by its pitch, scaled by its mix-bin volumes and mixed straight
 *     to 48 kHz stereo for the host.
 *   - The global/encode processors (GP/EP, two DSP56300s running Microsoft's
 *     effects and encoder code) are NOT emulated. Their mailbox is acknowledged
 *     so DirectSound never waits; effects such as reverb are therefore missing.
 *   - AC97: just enough for DirectSound's initialisation (codec ready, channel
 *     resets). The AC97 DMA buffers are filled by the EP, so they stay silent;
 *     the host gets the VP mix instead.
 *
 * Environment: HALO_TRACE_APU=1 logs methods; HALO_DUMP_WAV=path writes the
 * mixed output (headless verification). */
#include "runtime.h"
#include "cpu.h"
#include "nv2a.h"
#include "gpdsp.h"
void nv2a_raster_vstats(void);
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define APU_BASE 0xFE800000u
#define APU_SIZE 0x80000u
#define ACI_BASE 0xFEC00000u
#define ACI_SIZE 0x1000u

#define APU_REG(off) (*(uint32_t *)(g_mem + APU_BASE + (off)))

/* registers */
#define NV_PAPU_ISTS        0x1000      /* interrupt status, write 1 to clear */
#define   ISTS_GINTSTS      0x001
#define   ISTS_FETINTSTS    0x010       /* FE trapped */
#define NV_PAPU_IEN         0x1004
#define NV_PAPU_FECTL       0x1100
#define   FECTL_MODE_MASK   0x0E0       /* 0x00 free running, 0x80 halted, 0xE0 trapped */
#define   FECTL_TRAPREASON  0xF00       /* 0xF00: trapped on a method */
#define NV_PAPU_FETFORCE1   0x1504
#define NV_PAPU_FEDECMETH   0x1300
#define NV_PAPU_FEDECPARAM  0x1304
#define NV_PAPU_XGSCNT      0x200C      /* global sample counter */
#define NV_PAPU_VPVADDR     0x202C      /* voice records (physical) */
#define NV_PAPU_VPSGEADDR   0x2030      /* voice buffer scatter-gather list (physical) */
#define NV_PAPU_VPSSLADDR   0x2034      /* stream segment list (physical): 8 bytes per segment */
#define NV_PAPU_GPSADDR     0x2040      /* GP scratch scatter-gather list */
#define NV_PAPU_FENADDR     0x115C      /* notifiers (physical): 16 bytes each, status in byte 15 */
#define   ISTS_FEVINTSTS    0x040       /* a voice notifier was written */
#define NV_PAPU_GPRST       0x3FFFC

/* FE methods (PIO, 0xFE820000 + offset) */
#define PIO_FREE            0x010
#define PIO_ANTECEDENT      0x120
#define PIO_VOICE_ON        0x124
#define PIO_VOICE_OFF       0x128
#define PIO_VOICE_RELEASE   0x12C
#define PIO_VOICE_PAUSE     0x140
#define PIO_SYNCHRONIZE     0x104
#define PIO_CURRENT_SSL     0x190
#define PIO_SSL_SEGMENT     0x600       /* 0x600-0x7FC: offset/length pairs, relative to the current SSL */
#define PIO_CURRENT_VOICE   0x2F8
#define PIO_VOICE_LOCK      0x2FC
#define PIO_BUF_BASE        0x3A0
#define PIO_BUF_LBO         0x3A4
#define PIO_BUF_CBO         0x3D8
#define PIO_BUF_EBO         0x3DC
#define PIO_INBUF_SGE       0x804
#define PIO_INBUF_SGE_ADDR  0x808

/* voice record (0x80 bytes) */
#define V_VBIN       0x00   /* mix bins 0-5, 5 bits each (0,5,10,16,21,26) */
#define V_FMT        0x04   /* bins 6,7 (bits 0,5), sample/container size, stereo, loop, stream */
#define V_BASE       0x20   /* buffer base: byte address in SGE space (bits 23:0) */
#define V_LBO        0x24   /* loop start, in samples */
/* streams: 0x20 holds the playing segment list (PSL), 0x5C the next one (NSL):
 * first segment index in bits 23:8, segment count in 7:0 */
#define V_PSL        0x20
#define V_NSL        0x5C
#define V_STATE      0x54
#define   STATE_PAUSED   (1u << 18)
#define   STATE_NEW      (1u << 20)
#define   STATE_ACTIVE   (1u << 21)
#define V_CBO        0x58   /* current position, in samples (bits 23:0) */
#define V_EBO        0x5C   /* end position, in samples */
#define V_VOLA       0x60
#define V_VOLB       0x64
#define V_VOLC       0x68
#define V_PITCH_LINK 0x7C   /* pitch (signed, 1/4096 octave) in 31:16, next voice in 15:0 */

#define FMT_STREAM    (1u << 24)
#define FMT_LOOP      (1u << 25)
#define FMT_STEREO    (1u << 27)

#define MAX_VOICES 256
#define OUT_RATE 48000

static int g_apu_trace = -1;
static uint32_t g_logged;
static uint32_t g_apu_kint;           /* KINTERRUPT of the APU (level 5) */
static uint32_t g_cur_voice;

static uint32_t phys(uint32_t p) { return GUEST_RAM_MIRROR1 | (p & (GUEST_RAM_SIZE - 1)); }

static void trace_init(void) {
    if (g_apu_trace >= 0) return;
    g_apu_trace = getenv("HALO_TRACE_APU") ? atoi(getenv("HALO_TRACE_APU")) : 0;
}

int apu_owns(uint32_t a) { return a - APU_BASE < APU_SIZE || a - ACI_BASE < ACI_SIZE; }
void apu_set_interrupt(uint32_t kinterrupt) { g_apu_kint = kinterrupt; }

/* ------------------------------------------------------------- voices */
static uint32_t voice_addr(uint32_t h) {
    uint32_t base = APU_REG(NV_PAPU_VPVADDR);
    return base && h < MAX_VOICES ? phys(base) + h * 0x80 : 0;
}
static uint32_t vrd(uint32_t h, uint32_t off) { uint32_t a = voice_addr(h); return a ? rd32(a + off) : 0; }
static void vwr(uint32_t h, uint32_t off, uint32_t v) { uint32_t a = voice_addr(h); if (a) wr32(a + off, v); }
static void vset(uint32_t h, uint32_t off, uint32_t mask, uint32_t v) { vwr(h, off, (vrd(h, off) & ~mask) | (v & mask)); }

/* Idle-voice traps: the FE stops on "SE2FE_IDLE_VOICE" (method 0x8000, the
 * voice handle as parameter) and DirectSound finishes stopping the voice in
 * its interrupt handler. One trap at a time; the rest wait in a queue. */
static uint16_t g_trapq[512];
static int g_trapq_n;
static int g_trap_live;

static void queue_idle_trap(uint32_t h) {
    if (!(APU_REG(NV_PAPU_FETFORCE1) & 0x8000)) return;   /* idle-voice trap not requested */
    for (int i = 0; i < g_trapq_n; i++) if (g_trapq[i] == h) return;
    if (g_trapq_n < 512) g_trapq[g_trapq_n++] = (uint16_t)h;
}

static void raise_irq(void) {
    if (g_apu_kint && (APU_REG(NV_PAPU_ISTS) & APU_REG(NV_PAPU_IEN))) kernel_raise_interrupt(g_apu_kint);
}

static void deliver_trap(void) {
    if (g_trap_live || !g_trapq_n) return;
    if ((APU_REG(NV_PAPU_FECTL) & FECTL_MODE_MASK) != 0) return;   /* FE halted by software */
    uint32_t h = g_trapq[0];
    memmove(g_trapq, g_trapq + 1, (size_t)--g_trapq_n * sizeof g_trapq[0]);
    g_trap_live = 1;
    APU_REG(NV_PAPU_FEDECMETH) = 0x8000;
    APU_REG(NV_PAPU_FEDECPARAM) = h;
    APU_REG(NV_PAPU_FECTL) = (APU_REG(NV_PAPU_FECTL) & ~(FECTL_MODE_MASK | FECTL_TRAPREASON)) | 0xE0 | FECTL_TRAPREASON;
    APU_REG(NV_PAPU_ISTS) |= ISTS_FETINTSTS | ISTS_GINTSTS;
    if (g_apu_trace) runtime_log("[apu] idle trap for voice %03x", h);
    raise_irq();
}

/* Notifier n (0, 1: global; 2 + 4*voice + k: per voice, k 0/1 = stream
 * segment list A/B done). Status byte 15: 0x80 pending, 0x01 done. */
static void notify(uint32_t n, int irq) {
    uint32_t base = APU_REG(NV_PAPU_FENADDR);
    if (!base) return;
    wr8(phys(base) + n * 16 + 15, 1);
    if (g_apu_trace) runtime_log("[apu] notifier %u done (ISTS %08x IEN %08x)", n, APU_REG(NV_PAPU_ISTS), APU_REG(NV_PAPU_IEN));
    if (irq) {
        APU_REG(NV_PAPU_ISTS) |= ISTS_FEVINTSTS | ISTS_GINTSTS;
        raise_irq();
    }
}

typedef struct { uint32_t seg, pos; int side, starved; } StreamState;
static StreamState g_stream[MAX_VOICES];
static uint32_t g_cur_ssl;

static void voice_off(uint32_t h) {
    uint32_t st = vrd(h, V_STATE);
    vwr(h, V_STATE, st & ~(STATE_ACTIVE | STATE_NEW));
    queue_idle_trap(h);
}

static void fe_method(uint32_t m, uint32_t v) {
    uint32_t h = v & 0xFFFF;
    switch (m) {
    case PIO_ANTECEDENT: break;                          /* DirectSound links the voice lists itself */
    case PIO_VOICE_ON:
        vset(h, V_STATE, STATE_ACTIVE | STATE_NEW | STATE_PAUSED, STATE_ACTIVE | STATE_NEW);
        if (h < MAX_VOICES) memset(&g_stream[h], 0, sizeof g_stream[h]);
        break;
    case PIO_SYNCHRONIZE: notify(0, 0); break;           /* FE caught up: notifier 0 done */
    case PIO_CURRENT_SSL: g_cur_ssl = v; break;
    case PIO_VOICE_OFF:
    case PIO_VOICE_RELEASE:                              /* no envelopes: a release ends the voice */
        voice_off(h);
        break;
    case PIO_VOICE_PAUSE:
        vset(h, V_STATE, STATE_PAUSED, (v & (1u << 18)) ? STATE_PAUSED : 0);
        break;
    case PIO_CURRENT_VOICE: g_cur_voice = h; break;
    case PIO_VOICE_LOCK: break;
    case PIO_BUF_BASE: vset(g_cur_voice, V_BASE, 0x00FFFFFF, v); break;
    case PIO_BUF_LBO:  vset(g_cur_voice, V_LBO, 0x00FFFFFF, v); break;
    case PIO_BUF_CBO:  vset(g_cur_voice, V_CBO, 0x00FFFFFF, v); break;
    case PIO_BUF_EBO:  vset(g_cur_voice, V_EBO, 0x00FFFFFF, v); break;
    case 0x37C:        vset(g_cur_voice, V_PITCH_LINK, 0xFFFF0000, v); break;   /* pitch only, not the link */
    case PIO_INBUF_SGE_ADDR: {                           /* one entry of the voice buffer SGE list */
        uint32_t list = APU_REG(NV_PAPU_VPSGEADDR), idx = APU_REG(0x20000 + PIO_INBUF_SGE);
        if (list) { wr32(phys(list) + idx * 8, v & ~0xFFFu); wr32(phys(list) + idx * 8 + 4, 0); }
        break;
    }
    default:
        if (m >= 0x300 && m < 0x380) vwr(g_cur_voice, m - 0x300, v);   /* the record field at m - 0x300 */
        else if (m >= PIO_SSL_SEGMENT && m < PIO_SSL_SEGMENT + 0x200) {
            uint32_t list = APU_REG(NV_PAPU_VPSSLADDR);
            uint32_t idx = g_cur_ssl + (m - PIO_SSL_SEGMENT) / 8;
            if (list) wr32(phys(list) + idx * 8 + (m & 4), v);
        }
        break;
    }
}

/* ------------------------------------------------------------- sample fetch */
static int g_direct;                                  /* 1: stream segment, addresses are physical */
static uint32_t sge_byte(uint32_t addr) {             /* byte address in the voice SGE space -> guest */
    if (g_direct) return phys(addr);
    uint32_t list = APU_REG(NV_PAPU_VPSGEADDR);
    if (!list) return 0;
    uint32_t page = rd32(phys(list) + (addr >> 12) * 8);
    return page ? phys((page & ~0xFFFu) + (addr & 0xFFF)) : 0;
}

static const int8_t ima_index[16] = {-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8};
static const int16_t ima_step[89] = {
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166,
    1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845,
    8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767};

/* One Xbox ADPCM block (36 bytes per channel, 64 samples) -> 16-bit samples. */
typedef struct { uint32_t addr; int16_t s[2][64]; } AdpcmCache;
static AdpcmCache g_adpcm[MAX_VOICES];

static void adpcm_decode(uint32_t blk, int channels, int16_t out[2][64]) {
    for (int ch = 0; ch < channels; ch++) {
        uint32_t hdr = sge_byte(blk + (uint32_t)ch * 4);
        int pred = hdr ? (int16_t)rd16(hdr) : 0, idx = hdr ? rd8(hdr + 2) : 0;
        if (idx > 88) idx = 88;
        int n = 0;
        out[ch][n++] = (int16_t)pred;
        /* data: 4-byte words alternating between channels, 8 samples per word */
        for (int w = 0; n < 64; w++) {
            uint32_t a = sge_byte(blk + (uint32_t)channels * 4 + (uint32_t)(w * channels + ch) * 4);
            uint32_t word = a ? rd32(a) : 0;
            for (int k = 0; k < 8 && n < 64; k++) {
                int nib = (word >> (k * 4)) & 15, step = ima_step[idx], diff = step >> 3;
                if (nib & 1) diff += step >> 2;
                if (nib & 2) diff += step >> 1;
                if (nib & 4) diff += step;
                pred += (nib & 8) ? -diff : diff;
                if (pred > 32767) pred = 32767;
                if (pred < -32768) pred = -32768;
                idx += ima_index[nib];
                if (idx < 0) idx = 0;
                if (idx > 88) idx = 88;
                out[ch][n++] = (int16_t)pred;
            }
        }
    }
}

/* Sample `pos` of voice h: left and right (mono: both the same). */
static void fetch(uint32_t h, uint32_t fmt, uint32_t base, uint32_t pos, float *l, float *r) {
    int stereo = (fmt & FMT_STEREO) != 0, chans = stereo ? 2 : 1;
    uint32_t container = (fmt >> 30) & 3, ssize = (fmt >> 28) & 3;
    if (container == 2) {                                   /* ADPCM */
        uint32_t blk = base + (pos / 64) * 36u * (uint32_t)chans;
        AdpcmCache *c = &g_adpcm[h];
        if (c->addr != blk + 1) { adpcm_decode(blk, chans, c->s); c->addr = blk + 1; }
        *l = c->s[0][pos % 64] / 32768.0f;
        *r = stereo ? c->s[1][pos % 64] / 32768.0f : *l;
        return;
    }
    uint32_t bytes = container == 0 ? 1 : container == 1 ? 2 : 4;
    float v[2] = {0, 0};
    for (int ch = 0; ch < chans; ch++) {
        uint32_t a = sge_byte(base + (pos * (uint32_t)chans + (uint32_t)ch) * bytes);
        if (!a) continue;
        if (bytes == 1) v[ch] = ((int)rd8(a) - 128) / 128.0f;               /* 8-bit PCM is unsigned */
        else if (bytes == 2) v[ch] = (int16_t)rd16(a) / 32768.0f;
        else v[ch] = ssize == 2 ? (float)((int32_t)(rd32(a) << 8) >> 8) / 8388608.0f : (float)(int32_t)rd32(a) / 2147483648.0f;
    }
    *l = v[0];
    *r = stereo ? v[1] : v[0];
}

/* 12-bit attenuation -> linear gain (0xFFF = silent). */
static float gain(uint32_t att) {
    att &= 0xFFF;
    if (att >= 0xFFF) return 0.0f;
    return powf(10.0f, -(float)att / (64.0f * 20.0f));
}

/* ------------------------------------------------------------- mixing */
static float g_frac[MAX_VOICES];

/* ------------------------------------------------------------- reverb
 * On the Xbox the I3DL2 reverb runs as a program on the APU's GP DSP, fed by
 * mix bin 10 (DSMIXBIN_I3DL2). The DSP is not emulated; instead the voices'
 * bin-10 sends go into a mono bus that a Freeverb network (8 combs + 4
 * allpasses per side, Jezar's public-domain design) turns into a stereo
 * tail. The game decides per sound how much reverb it gets, and the room
 * character follows the game's I3DL2 listener settings (rv_follow_room):
 * decay time, high-frequency damping, level, pre-delay and diffusion are read
 * from DirectSound's listener object whenever the player enters a new sound
 * environment. HALO_REVERB=0 turns it off, HALO_REVERB=<0-400> sets the level
 * in percent (default 100); HALO_REVERB_FIXED=1 keeps the old fixed room. */
#define RV_COMBS 8
#define RV_APS 4
static const int rv_comb_len[RV_COMBS] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
static const int rv_ap_len[RV_APS] = {556, 441, 341, 225};
typedef struct { float *buf; int len, pos; float store; } RvComb;
typedef struct { float *buf; int len, pos; } RvAp;
static RvComb g_rv_comb[2][RV_COMBS];
static RvAp g_rv_ap[2][RV_APS];
static float g_rv_level = -1.0f;               /* -1: not set up yet */
static float g_rv_user = 1.0f;                 /* from the Audio settings */
static int g_rv_env;                           /* HALO_REVERB given: it wins */
static float g_revin[64];
static int g_bin_stats = -1;                     /* HALO_APU_BINS=1: log how much each mix bin is used */
static double g_bin_use[32];
static const float RV_FEEDBACK = 0.84f, RV_DAMP = 0.2f, RV_INGAIN = 0.015f, RV_WET = 9.6f;   /* wet: about 15-20 dB below the dry mix with Halo's send levels */
#define RV_PRE_MAX 9600                          /* pre-delay line: up to 0.2 s */
static float g_rv_pre[RV_PRE_MAX];
static int g_rv_pre_pos;
/* current (smoothed) and target room parameters */
typedef struct { float fb, damp, wet, ap, pre; } RvRoom;
static RvRoom g_rv_cur = {0.84f, 0.2f, 1.0f, 0.5f, 0}, g_rv_tgt = {0.84f, 0.2f, 1.0f, 0.5f, 0};

/* Halo (this build) keeps its IDirectSound8 pointer at 0x3a473c and passes
 * it to IDirectSound8_SetI3DL2Listener (code at 0x28fa3). DirectSound stores
 * the DSI3DL2LISTENER (12 dwords) at +0x184 of the listener object, which is
 * [CDirectSound + 0xC], with CDirectSound = interface - 8. */
static int rv_guest_ok(uint32_t a) { return a && ((a >= 0x10000 && a < 0x08000000) || (a >= 0x80000000u && a < 0x88000000u)); }

static void rv_follow_room(void) {
    static int mode = -1;                        /* 1: follow the game; 0: fixed room */
    static uint32_t last[12];
    if (mode < 0) {
        const char *f = getenv("HALO_REVERB_FIXED");
        mode = !(f && *f && *f != '0') && rd32(0x28fa3) == 0x473c0d8bu && rd32(0x28fc9) == 0x16ac65e8u;
        runtime_log("[apu] reverb room: %s", mode ? "follows the game's I3DL2 settings" : "fixed");
    }
    if (!mode) return;
    uint32_t ids = rd32(0x3a473c), lis = 0;
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("HALO_TRACE_ROOM") != NULL;
    if (rv_guest_ok(ids) && rv_guest_ok(ids - 8)) lis = rd32(ids - 8 + 0xC);
    if (dbg) {
        static uint32_t pi, pl;
        if (ids != pi || lis != pl) runtime_log("[apu] room: IDirectSound %08x listener %08x", ids, lis);
        pi = ids; pl = lis;
    }
    if (!rv_guest_ok(lis)) return;
    uint32_t v[12];
    for (int i = 0; i < 12; i++) v[i] = rd32(lis + 0x184 + 4 * (uint32_t)i);
    if (!memcmp(v, last, sizeof v)) return;
    memcpy(last, v, sizeof v);
    if (dbg) runtime_log("[apu] room raw: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x",
                         v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]);
    int32_t room = (int32_t)v[0], room_hf = (int32_t)v[1], reverb = (int32_t)v[7];
    float decay, hf_ratio, refl_delay, rev_delay, diffusion;
    memcpy(&decay, &v[3], 4); memcpy(&hf_ratio, &v[4], 4); memcpy(&refl_delay, &v[6], 4);
    memcpy(&rev_delay, &v[8], 4); memcpy(&diffusion, &v[9], 4);
    if (!(decay >= 0.1f && decay <= 20.0f) || !(hf_ratio >= 0.1f && hf_ratio <= 2.0f)) return;   /* not a listener block */
    if (room < -10000 || room > 0 || reverb < -10000 || reverb > 2000) return;
    /* comb feedback for the decay time (RT60) over the mean comb delay (~31 ms) */
    const float mean_delay = 1387.0f * 48000.0f / 44100.0f / 48000.0f;
    float fb = powf(10.0f, -3.0f * mean_delay / decay);
    if (fb > 0.97f) fb = 0.97f;
    /* high frequencies decay faster when the ratio is below 1; RoomHF dulls the whole tail */
    float damp = 0.1f + 0.4f * (1.0f - hf_ratio) + (float)(-room_hf) / 10000.0f * 0.1f;   /* Freeverb's own range is 0-0.4 */
    if (damp < 0.0f) damp = 0.0f;
    if (damp > 0.5f) damp = 0.5f;
    /* late-reverb level relative to the generic room (Room -1000 mB, Reverb +200 mB),
     * corrected for the tail energy that the feedback itself adds */
    float lvl = room <= -10000 || reverb <= -10000 ? 0.0f : powf(10.0f, (float)(room + reverb + 800) / 2000.0f);
    lvl *= sqrtf((1.0f - fb * fb) / (1.0f - RV_FEEDBACK * RV_FEEDBACK));
    if (lvl > 4.0f) lvl = 4.0f;
    float pre = (refl_delay + rev_delay) * 48000.0f;
    if (!(pre >= 0)) pre = 0;
    if (pre > RV_PRE_MAX - 1) pre = RV_PRE_MAX - 1;
    float ap = 0.3f + 0.25f * (diffusion >= 0 && diffusion <= 100 ? diffusion : 100.0f) / 100.0f;
    g_rv_tgt = (RvRoom){fb, damp, lvl, ap, pre};
    static int logged;
    if (logged++ < 200)
        runtime_log("[apu] room: Room %d RoomHF %d Decay %.2fs HFratio %.2f Reverb %d delay %.3f -> feedback %.3f damp %.2f level %.2f",
                    room, room_hf, decay, hf_ratio, reverb, refl_delay + rev_delay, fb, damp, lvl);
}

static void reverb_init(void) {
    const char *e = getenv("HALO_REVERB");
    g_rv_env = e != NULL;
    g_rv_level = e ? (float)atoi(e) / 100.0f : g_rv_user;
    if (g_rv_level < 0) g_rv_level = 0;
    if (g_rv_level > 4) g_rv_level = 4;
    for (int c = 0; c < 2; c++) {                              /* lengths scaled from 44.1 to 48 kHz; right side +23 */
        for (int i = 0; i < RV_COMBS; i++) {
            int len = (rv_comb_len[i] + c * 23) * 48000 / 44100;
            g_rv_comb[c][i] = (RvComb){calloc((size_t)len, sizeof(float)), len, 0, 0};
        }
        for (int i = 0; i < RV_APS; i++) {
            int len = (rv_ap_len[i] + c * 23) * 48000 / 44100;
            g_rv_ap[c][i] = (RvAp){calloc((size_t)len, sizeof(float)), len, 0};
        }
    }
    if (g_rv_level > 0) runtime_log("[apu] reverb on (level %.0f%%)", g_rv_level * 100.0f);
}

void apu_set_reverb_level(float level) {
    if (level < 0) level = 0;
    if (level > 4) level = 4;
    g_rv_user = level;
    if (g_rv_level >= 0 && !g_rv_env) g_rv_level = level;   /* already running: takes effect at once */
}

/* in: mono reverb sends; out: interleaved stereo (accumulated) */
static void reverb_process(const float *in, float *out, int n) {
    if (g_rv_level <= 0) return;
    static unsigned blocks;
    if ((blocks++ & 63) == 0) rv_follow_room();               /* every ~43 ms */
    /* glide toward the new room over ~0.1 s so changes do not click */
    RvRoom *r = &g_rv_cur; const RvRoom *t = &g_rv_tgt; const float k = 0.01f;
    r->fb += (t->fb - r->fb) * k; r->damp += (t->damp - r->damp) * k; r->wet += (t->wet - r->wet) * k;
    r->ap += (t->ap - r->ap) * k; r->pre += (t->pre - r->pre) * k;
    const float fb = r->fb, damp = r->damp, apg = r->ap, wet = RV_WET * g_rv_level * r->wet;
    const int pre = (int)r->pre;
    for (int i = 0; i < n; i++) {
        g_rv_pre[g_rv_pre_pos] = in[i] * RV_INGAIN;
        int rp = g_rv_pre_pos - pre; if (rp < 0) rp += RV_PRE_MAX;
        float x = g_rv_pre[rp];
        if (++g_rv_pre_pos >= RV_PRE_MAX) g_rv_pre_pos = 0;
        for (int c = 0; c < 2; c++) {
            float y = 0;
            for (int k2 = 0; k2 < RV_COMBS; k2++) {
                RvComb *cb = &g_rv_comb[c][k2];
                float o = cb->buf[cb->pos];
                cb->store = o * (1.0f - damp) + cb->store * damp;
                cb->buf[cb->pos] = x + cb->store * fb;
                if (++cb->pos >= cb->len) cb->pos = 0;
                y += o;
            }
            for (int k2 = 0; k2 < RV_APS; k2++) {
                RvAp *ap = &g_rv_ap[c][k2];
                float b = ap->buf[ap->pos];
                ap->buf[ap->pos] = y + b * apg;
                if (++ap->pos >= ap->len) ap->pos = 0;
                y = b - y;
            }
            out[i * 2 + c] += y * wet;
        }
    }
}

/* A stream voice has two segment-list slots, A (record 0x20) and B (0x5C),
 * each naming a run of segments in the SSL table (first in bits 23:8, count
 * in 7:0). The voice plays A, then B, then A again: when a slot is used up its
 * notifier (2 + 4*voice + slot) is set, its count cleared, and the game
 * refills it while the other slot plays. An empty slot starves the voice. */
/* ------------------------------------------------------------- voice filters
 * DirectSound puts a filter mode in CFG_MISC bits 17:16 (0 bypass, 1 DLS2,
 * 2 parametric EQ, 3 I3DL2; 3D voices get 3) and the coefficients in
 * TAR_FCA (0x74, from SetFilter) and TAR_FCB (0x78). For I3DL2, FCB holds
 * two one-pole low-pass coefficients computed by DirectSound's 3D code
 * (0x198632): bits 15:0 for the direct path, 31:16 for the reverb send.
 * Each is a pole a in 0.16 fixed point, chosen so that the power gain at
 * the listener's HF reference (5 kHz) matches the DirectHF / RoomHF levels
 * (obstruction, occlusion, distance): y += (1 - a) * (x - y). 0 is no
 * filtering, 0xFFFF fully closed. HALO_NO_VFILTER=1 turns this off. */
#define V_MISC 0x18
#define V_FCA  0x74
#define V_FCB  0x78
typedef struct { float kd, kr; float yl, yr, yrl, yrr; int on; } VFilt;
static VFilt g_vf[MAX_VOICES];
static int g_vf_enable = -1;
static uint64_t g_vf_stats[4], g_vf_active;          /* voices seen per mode; voices actually filtered */

static void vfilt_setup(uint32_t h) {
    VFilt *f = &g_vf[h];
    if (g_vf_enable < 0) g_vf_enable = !getenv("HALO_NO_VFILTER");
    uint32_t mode = (vrd(h, V_MISC) >> 16) & 3, fcb = vrd(h, V_FCB);
    if (g_bin_stats > 0) g_vf_stats[mode]++;
    f->on = 0;
    if (!g_vf_enable || mode != 3) return;
    float ad = (float)(fcb & 0xFFFF) / 65536.0f, ar = (float)(fcb >> 16) / 65536.0f;
    f->kd = 1.0f - ad;
    f->kr = 1.0f - ar;
    f->on = (fcb & 0xFFFF) != 0 || (fcb >> 16) != 0;
    if (f->on && g_bin_stats > 0) g_vf_active++;
}

/* Where one voice's samples go: either folded to stereo (plus the reverb
 * bus), or into the 32 mix bins the GP DSP reads (bins != NULL). */
typedef struct {
    float g[2][2];                       /* [source channel][output side] */
    float rev[2];                        /* [source channel] -> reverb bus (bin 10) */
    int nslot;
    uint8_t bin[8], src[8];
    float gain[8];
    VFilt *f;
    float (*bins)[32];
} VoiceMix;

static inline void emit(VoiceMix *vm, int i, float sl, float sr, float *out) {
    float dl = sl, dr = sr, rl = sl, rr = sr;
    VFilt *f = vm->f;
    if (f->on) {
        f->yl += f->kd * (sl - f->yl);  f->yr += f->kd * (sr - f->yr);
        f->yrl += f->kr * (sl - f->yrl); f->yrr += f->kr * (sr - f->yrr);
        dl = f->yl; dr = f->yr; rl = f->yrl; rr = f->yrr;
    }
    if (vm->bins) {
        for (int k = 0; k < vm->nslot; k++) {
            int b = vm->bin[k];
            float v = b == 10 ? (vm->src[k] ? rr : rl) : (vm->src[k] ? dr : dl);
            vm->bins[b][i] += v * vm->gain[k];
        }
        return;
    }
    out[i * 2] += dl * vm->g[0][0] + dr * vm->g[1][0];
    out[i * 2 + 1] += dl * vm->g[0][1] + dr * vm->g[1][1];
    g_revin[i] += rl * vm->rev[0] + rr * vm->rev[1];
}

static void mix_stream(uint32_t h, uint32_t fmt, float step, VoiceMix *vm, float *out, int n) {
    StreamState *ss = &g_stream[h];
    uint32_t list = APU_REG(NV_PAPU_VPSSLADDR);
    if (!list) return;
    float frac = g_frac[h];
    g_direct = 1;
    for (int i = 0, guard = 0; i < n && guard < 8; ) {
        uint32_t field = ss->side ? V_NSL : V_PSL;
        uint32_t sl = vrd(h, field), first = (sl >> 8) & 0xFFFF, count = sl & 0xFF;
        if (!count || ss->seg >= count) {                    /* starved: slot not (re)filled yet */
            if (g_apu_trace && !ss->starved) runtime_log("[apu] voice %03x starved on slot %d (%08x)", h, ss->side, sl);
            ss->starved = 1;
            break;
        }
        ss->starved = 0;
        uint32_t e = phys(list) + (first + ss->seg) * 8, addr = rd32(e), len = rd32(e + 4) & 0xFFFF;
        if (ss->pos >= len) {                                /* segment used up */
            ss->pos = 0;
            guard++;
            if (++ss->seg >= count) {                        /* slot used up: hand it back */
                int done = ss->side;
                vwr(h, field, sl & ~0xFFu);                  /* before notifying: the game refills it at once */
                ss->side ^= 1;
                ss->seg = 0;
                notify(2 + h * 4 + (uint32_t)done, 1);
            }
            continue;
        }
        guard = 0;
        float l0, r0, l1, r1;
        fetch(h, fmt, addr, ss->pos, &l0, &r0);
        if (ss->pos + 1 < len) fetch(h, fmt, addr, ss->pos + 1, &l1, &r1); else { l1 = l0; r1 = r0; }
        emit(vm, i, l0 + (l1 - l0) * frac, r0 + (r1 - r0) * frac, out);
        frac += step;
        while (frac >= 1.0f) { frac -= 1.0f; ss->pos++; }
        i++;
    }
    g_direct = 0;
    g_frac[h] = frac;
}

/* Mix `n` frames at 48 kHz into out[] (interleaved stereo, accumulated), or,
 * with bins, into the 32 mix bins for the GP DSP. */
static void mix_voices(float *out, int n, float (*bins)[32]) {
    if (!APU_REG(NV_PAPU_VPVADDR) || !APU_REG(NV_PAPU_VPSGEADDR)) return;
    for (uint32_t h = 0; h < MAX_VOICES; h++) {
        uint32_t st = vrd(h, V_STATE);
        if (!(st & STATE_ACTIVE) || (st & STATE_PAUSED)) continue;
        uint32_t fmt = vrd(h, V_FMT);
        if (st & STATE_NEW) {
            vwr(h, V_STATE, st & ~STATE_NEW); g_frac[h] = 0; g_adpcm[h].addr = 0;
            g_vf[h].yl = g_vf[h].yr = g_vf[h].yrl = g_vf[h].yrr = 0;
        }
        uint32_t vbin = vrd(h, V_VBIN), va = vrd(h, V_VOLA), vb = vrd(h, V_VOLB), vc = vrd(h, V_VOLC);
        uint32_t vbins[8] = {vbin & 31, (vbin >> 5) & 31, (vbin >> 10) & 31, (vbin >> 16) & 31,
                            (vbin >> 21) & 31, (vbin >> 26) & 31, fmt & 31, (fmt >> 5) & 31};
        uint32_t vol[8] = {(va >> 4) & 0xFFF, va >> 20, (vb >> 4) & 0xFFF, vb >> 20, (vc >> 4) & 0xFFF, vc >> 20,
                           (va & 15) | (vb & 15) << 4 | (vc & 15) << 8,
                           ((va >> 16) & 15) | ((vb >> 16) & 15) << 4 | ((vc >> 16) & 15) << 8};
        /* Mix bins -> stereo: front L/R, centre, rear L/R and the 3D crosstalk
         * bins; LFE and the effect sends (reverb etc.) are dropped. A stereo
         * voice feeds its left channel to the even slots, its right to the odd. */
        int stereo = (fmt & FMT_STEREO) != 0;
        VoiceMix vm = {.f = &g_vf[h], .bins = bins};
        float (*g)[2] = vm.g, *rev = vm.rev;
        for (int i = 0; i < 8; i++) {
            float a = gain(vol[i]), L = 0, R = 0;
            if (a == 0) continue;
            if (g_bin_stats > 0) g_bin_use[vbins[i]] += a;
            vm.bin[vm.nslot] = (uint8_t)vbins[i]; vm.src[vm.nslot] = (uint8_t)(stereo ? (i & 1) : 0); vm.gain[vm.nslot++] = a;
            if (vbins[i] == 10) { rev[stereo ? (i & 1) : 0] += a; continue; }
            switch (vbins[i]) {
            case 0: case 6: L = 1; break;
            case 1: case 7: R = 1; break;
            case 2: L = R = 0.7071f; break;
            case 4: case 8: L = 0.7071f; break;
            case 5: case 9: R = 0.7071f; break;
            default: break;
            }
            int src = stereo ? (i & 1) : 0;
            g[src][0] += a * L;
            g[src][1] += a * R;
        }
        if (g_rv_level <= 0 && !bins) rev[0] = rev[1] = 0;
        if (bins ? vm.nslot == 0 : (g[0][0] == 0 && g[0][1] == 0 && g[1][0] == 0 && g[1][1] == 0 && rev[0] == 0 && rev[1] == 0)) continue;
        vfilt_setup(h);
        int pitch = (int16_t)(vrd(h, V_PITCH_LINK) >> 16);
        float step = exp2f((float)pitch / 4096.0f);
        if (fmt & FMT_STREAM) { mix_stream(h, fmt, step, &vm, out, n); continue; }
        uint32_t base = vrd(h, V_BASE) & 0x00FFFFFF, lbo = vrd(h, V_LBO) & 0x00FFFFFF;
        uint32_t ebo = vrd(h, V_EBO) & 0x00FFFFFF, cbo = vrd(h, V_CBO) & 0x00FFFFFF;
        int loop = (fmt & FMT_LOOP) != 0, ended = 0;
        float frac = g_frac[h];
        for (int i = 0; i < n && !ended; i++) {
            float l0, r0, l1, r1;
            uint32_t nxt = cbo >= ebo ? (loop ? lbo : cbo) : cbo + 1;
            fetch(h, fmt, base, cbo, &l0, &r0);
            fetch(h, fmt, base, nxt, &l1, &r1);
            emit(&vm, i, l0 + (l1 - l0) * frac, r0 + (r1 - r0) * frac, out);
            frac += step;
            while (frac >= 1.0f) {
                frac -= 1.0f;
                if (cbo < ebo) cbo++;
                else if (loop) cbo = lbo;
                else { ended = 1; break; }
            }
        }
        g_frac[h] = frac;
        vset(h, V_CBO, 0x00FFFFFF, cbo);
        if (ended) voice_off(h);
    }
}

/* ------------------------------------------------------------- output ring */
#define RING_FRAMES 32768                                    /* ~680 ms */
static float g_ring[RING_FRAMES * 2];
static uint32_t g_ring_w, g_ring_r;                          /* frame counters (free running) */
static FILE *g_wav;
static uint32_t g_wav_frames;
static int g_host_pulls;                                     /* a host consumer exists */

static void wav_header(FILE *f, uint32_t frames) {
    uint32_t data = frames * 4, riff = 36 + data, fmt_len = 16, rate = OUT_RATE, brate = OUT_RATE * 4;
    uint16_t pcm = 1, ch = 2, align = 4, bits = 16;
    fseek(f, 0, SEEK_SET);
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt_len, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&brate, 4, 1, f);
    fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    fseek(f, 0, SEEK_END);
}

static void ring_push(const float *s, int n) {
    if (g_host_pulls) {
        for (int i = 0; i < n; i++) {
            uint32_t w = g_ring_w;
            if (w - __atomic_load_n(&g_ring_r, __ATOMIC_ACQUIRE) >= RING_FRAMES) break;   /* host stalled: drop */
            g_ring[(w % RING_FRAMES) * 2] = s[i * 2];
            g_ring[(w % RING_FRAMES) * 2 + 1] = s[i * 2 + 1];
            __atomic_store_n(&g_ring_w, w + 1, __ATOMIC_RELEASE);
        }
    }
    if (g_wav) {
        for (int i = 0; i < n * 2; i++) {
            float v = s[i] < -1 ? -1 : s[i] > 1 ? 1 : s[i];
            int16_t q = (int16_t)(v * 32767.0f);
            fwrite(&q, 2, 1, g_wav);
        }
        g_wav_frames += (uint32_t)n;
        if ((g_wav_frames & 0x3FFF) < (uint32_t)n) { wav_header(g_wav, g_wav_frames); fflush(g_wav); }
    }
}

/* Host side (its audio thread): take up to n frames of interleaved stereo
 * float; the rest is filled with silence. Returns the frames delivered. */
int apu_pull_audio(float *out, int n) {
    g_host_pulls = 1;
    uint32_t r = g_ring_r, avail = __atomic_load_n(&g_ring_w, __ATOMIC_ACQUIRE) - r;
    int got = (int)(avail < (uint32_t)n ? avail : (uint32_t)n);
    for (int i = 0; i < got; i++) {
        out[i * 2] = g_ring[((r + (uint32_t)i) % RING_FRAMES) * 2];
        out[i * 2 + 1] = g_ring[((r + (uint32_t)i) % RING_FRAMES) * 2 + 1];
    }
    for (int i = got; i < n; i++) out[i * 2] = out[i * 2 + 1] = 0;
    __atomic_store_n(&g_ring_r, r + (uint32_t)got, __ATOMIC_RELEASE);
    return got;
}
/* Frames waiting in the ring (the host keeps latency in check with this). */
int apu_audio_buffered(void) {
    g_host_pulls = 1;                                         /* a host is listening: start filling the ring */
    return (int)(__atomic_load_n(&g_ring_w, __ATOMIC_ACQUIRE) - g_ring_r);
}

/* ------------------------------------------------------------- AC97 */
/* Bus-master channels: PCM out at 0x110, S/PDIF out at 0x170.
 * +0 BDBAR (32), +4 CIV (8), +5 LVI (8), +6 SR (16), +8 PICB (16), +A PIV (8), +B CR (8) */
static void aci_write(uint32_t off, int size) {
    uint8_t *r = g_mem + ACI_BASE;
    for (uint32_t ch = 0x110; ch <= 0x170; ch += 0x60) {
        if (off <= ch + 0xB && off + (uint32_t)size > ch + 0xB && (r[ch + 0xB] & 2)) {   /* RR: reset channel */
            memset(r + ch + 4, 0, 6);
            r[ch + 6] = 1;                                                   /* SR: DMA controller halted */
            r[ch + 0xA] = 0;
            r[ch + 0xB] = 0;
        }
    }
}

static uint32_t aci_read(uint32_t off, uint32_t v) {
    switch (off & ~3u) {
    case 0x130: v |= 0x100; break;            /* GLOB_STA: primary codec ready */
    case 0x134: v &= ~1u; break;              /* CAS: codec access semaphore free */
    }
    return v;
}

/* ------------------------------------------------------------- MMIO */
uint32_t apu_read(uint32_t a, int size, uint32_t mem_value) {
    (void)size;
    if (a - ACI_BASE < ACI_SIZE) return aci_read(a - ACI_BASE, mem_value);
    if (a - APU_BASE == 0x20000 + PIO_FREE) return 0x80;   /* method FIFO empty: methods run on write */
    uint32_t o = a - APU_BASE;
    if (o - 0x30000 < 0x10000u || o - 0x50000 < 0x10000u) {   /* GP / EP DSP memories */
        uint32_t v;
        if (gpdsp_mmio_read(o >= 0x50000, o & 0xFFFF, &v)) return v;
    }
    return mem_value;
}

/* Called before the value is stored in the register file. Returns 1 when the
 * write is fully handled here (nothing to store). */
int apu_write_pre(uint32_t a, uint32_t v, int size) {
    if (a - APU_BASE == NV_PAPU_ISTS && size == 4) {       /* write 1 to clear */
        APU_REG(NV_PAPU_ISTS) &= ~v;
        return 1;
    }
    return 0;
}

void apu_write(uint32_t a, uint32_t v, int size) {
    trace_init();
    if (a - ACI_BASE < ACI_SIZE) { aci_write(a - ACI_BASE, size); return; }
    uint32_t off = a - APU_BASE;
    if (size != 4) return;
    if (off == 0x1334) {
        /* FEMEMDATA: the FE writes the value to the physical address in
         * FEMEMADDR (0x1324). DirectSound uses it as a fence and polls for it
         * with KeStallExecutionProcessor inside its DPC. */
        uint32_t dst = APU_REG(0x1324);
        if (dst) wr32(phys(dst), v);
        return;
    }
    if (off == NV_PAPU_FECTL) {
        if ((v & FECTL_MODE_MASK) == 0 && g_trap_live) {     /* software resumed the FE after the trap */
            g_trap_live = 0;
            APU_REG(NV_PAPU_FECTL) = v & ~FECTL_TRAPREASON;
        }
        return;
    }
    if (off - 0x30000 < 0x10000u || off - 0x50000 < 0x10000u) gpdsp_mmio_write(off >= 0x50000, off & 0xFFFF, v);   /* GP / EP */
    if (off >= 0x30000) {                                /* DSP memories (GP 0x30000-0x3FFFF, EP 0x50000-) */
        static int tr = -1, logged;
        if (tr < 0) tr = getenv("HALO_TRACE_GP") != NULL;
        if (tr && logged++ < 400) runtime_log("[gp] write %05x = %08x", off, v);
    }
    if (off >= 0x20000 && off < 0x30000) {
        uint32_t m = off - 0x20000;
        if (g_apu_trace && g_logged < 100000 && m != PIO_CURRENT_VOICE && m != PIO_VOICE_LOCK &&
            !(m >= 0x400 && m < 0x440) && m != 0x160 && !(m >= 0x200 && m < 0x280)) {
            g_logged++;
            runtime_log("[apu] method %03x = %08x (voice %03x)", m, v, g_cur_voice);
        }
        fe_method(m, v);
    }
}

/* ------------------------------------------------------------- GP mailbox */
/* DirectSound sends effect commands to the GP's code through a mailbox in the
 * first GP scratch page (+0x800, status word +0x10). Acknowledge each one. */
static void gp_mailbox_ack(void) {
    uint32_t sge = APU_REG(NV_PAPU_GPSADDR);
    if (!sge || !(APU_REG(NV_PAPU_GPRST) & 1)) return;
    uint32_t page = rd32(phys(sge)) & ~0xFFFu;
    if (!page) return;
    uint32_t status = phys(page) + 0x810;
    if (rd32(status)) {
        static int tr = -1, logged;
        if (tr < 0) tr = getenv("HALO_TRACE_GP") != NULL;
        if (tr && logged++ < 200) {            /* diagnosis: the command block DirectSound sends to the DSP */
            uint32_t b = phys(page) + 0x800;
            runtime_log("[gp] mailbox %08x %08x %08x %08x | status %08x | %08x %08x %08x %08x %08x %08x %08x %08x",
                        rd32(b), rd32(b + 4), rd32(b + 8), rd32(b + 12), rd32(status), rd32(b + 0x14), rd32(b + 0x18),
                        rd32(b + 0x1C), rd32(b + 0x20), rd32(b + 0x24), rd32(b + 0x28), rd32(b + 0x2C), rd32(b + 0x30));
            if (rd32(b) == 4 && logged < 6) {          /* the parameter block: scratch word offset b+4, count b+12 */
                uint32_t woff = rd32(b + 4), cnt = rd32(b + 12);
                char line[1200]; int o = 0;
                for (uint32_t k = 0; k < cnt && k < 132 && o < 1150; k++) {
                    uint32_t byte = (woff + k) * 4, pg = rd32(phys(sge) + (byte >> 12) * 8) & ~0xFFFu;
                    uint32_t w = pg ? rd32(phys(pg) + (byte & 0xFFF)) : 0xDEAD;
                    o += snprintf(line + o, sizeof line - (size_t)o, " %06x", w & 0xFFFFFF);
                }
                runtime_log("[gp] block:%s", line);
            }
        }
        wr32(status, 0);
    }
}

/* ------------------------------------------------------------- effects choice */
static volatile int g_use_dsp = 1;                           /* Audio settings: Xbox DSP or built-in reverb */
void apu_set_dsp_effects(int on) {
    if (g_use_dsp != !!on) runtime_log("[apu] effects: %s", on ? "Xbox audio DSP" : "built-in reverb");
    g_use_dsp = !!on;
}

/* ------------------------------------------------------------- clock */
#define FRAME 32                                             /* the VP works in 32-sample frames */
void apu_tick(uint64_t now_ns) {
    static uint64_t t0, done;                                /* VP frames produced since t0 */
    trace_init();
    if (!t0) {
        t0 = now_ns;
        const char *w = getenv("HALO_DUMP_WAV");
        if (w && (g_wav = fopen(w, "wb"))) wav_header(g_wav, 0);
    }
    int dsp = g_use_dsp && gpdsp_running();
    if (!dsp) gp_mailbox_ack();                              /* with the GP running, its own program answers */
    deliver_trap();
    {
        static uint64_t next;
        if (now_ns >= next) { if (next) nv2a_raster_vstats(); next = now_ns + 10000000000ull; }
    }
    uint64_t due = (now_ns - t0) * OUT_RATE / 1000000000ull / FRAME;
    /* The ticker can be held off for a while (the game thread keeps the guest
     * lock while it decodes video): catch up on up to half a second, so the
     * stream positions and the sound stay continuous; beyond that, skip. */
    if (due > done + OUT_RATE / 2 / FRAME) done = due - OUT_RATE / 2 / FRAME;
    while (done < due) {
        float buf[FRAME * 2];
        memset(buf, 0, sizeof buf);
        if (g_rv_level < 0) { reverb_init(); g_bin_stats = getenv("HALO_APU_BINS") != NULL; }
        if (g_bin_stats > 0) {
            static uint64_t frames;
            if (++frames % (OUT_RATE * 10 / FRAME) == 0) {
                char line[512]; int o = 0;
                for (int b = 0; b < 32; b++) if (g_bin_use[b] > 0) o += snprintf(line + o, sizeof line - (size_t)o, " %d:%.0f", b, g_bin_use[b]);
                runtime_log("[apu] mix bin use (sum of voice gains per frame):%s", o ? line : " none");
                runtime_log("[apu] filter modes (voice-frames): bypass %llu, DLS2 %llu, ParamEQ %llu, I3DL2 %llu; filtered %llu",
                            (unsigned long long)g_vf_stats[0], (unsigned long long)g_vf_stats[1], (unsigned long long)g_vf_stats[2],
                            (unsigned long long)g_vf_stats[3], (unsigned long long)g_vf_active);
                memset(g_bin_use, 0, sizeof g_bin_use); memset(g_vf_stats, 0, sizeof g_vf_stats); g_vf_active = 0;
            }
        }
        memset(g_revin, 0, sizeof g_revin);
        if (dsp) {                                           /* the GP DSP mixes the bins and adds the effects */
            static float bins[GP_BINS][GP_FRAME];
            memset(bins, 0, sizeof bins);
            mix_voices(buf, FRAME, bins);
            gpdsp_submit(bins, ring_push);
            APU_REG(NV_PAPU_XGSCNT) += FRAME;
            done++;
            continue;
        }
        mix_voices(buf, FRAME, NULL);
        if (g_bin_stats > 0) {
            static double dry, wet, sends; static uint64_t k;
            float tmp[FRAME * 2]; memset(tmp, 0, sizeof tmp);
            for (int i = 0; i < FRAME * 2; i++) dry += (double)buf[i] * buf[i];
            for (int i = 0; i < FRAME; i++) sends += (double)g_revin[i] * g_revin[i];
            reverb_process(g_revin, tmp, FRAME);
            for (int i = 0; i < FRAME * 2; i++) { wet += (double)tmp[i] * tmp[i]; buf[i] += tmp[i]; }
            if (++k % (OUT_RATE * 10 / FRAME) == 0) {
                runtime_log("[apu] level (rms): dry %.4f, reverb sends %.4f, reverb out %.4f", sqrt(dry / (FRAME * 2.0 * k)),
                            sqrt(sends / (FRAME * 1.0 * k)), sqrt(wet / (FRAME * 2.0 * k)));
            }
        } else reverb_process(g_revin, buf, FRAME);
        ring_push(buf, FRAME);
        APU_REG(NV_PAPU_XGSCNT) += FRAME;
        done++;
    }
}
