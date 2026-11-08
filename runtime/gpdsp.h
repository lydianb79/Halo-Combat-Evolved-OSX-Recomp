/* gpdsp.h — the APU's two DSPs (Motorola DSP56300 cores): the GP running
 * DirectSound's effects program and the EP producing the final mix, through the dsp56300 JIT library (MIT, Matt Borgerson).
 * Built only when HALO_DSP is defined (the Makefile does this when the
 * library is available); otherwise these are stubs and the APU keeps its
 * built-in reverb approximation. */
#pragma once
#include <stdint.h>

#define GP_BINS 32               /* mix bins the VP hands to the GP each frame */
#define GP_FRAME 32              /* samples per frame */

int  gpdsp_compiled(void);       /* built with the DSP library */
int  gpdsp_enabled(void);        /* library present and not turned off (HALO_DSP=0) */
int  gpdsp_running(void);        /* the game has started the GP (reset released) */
#define GP_OUT_MAX 256             /* most stereo samples one call can return */
/* ep: 0 for the GP window (0x30000), 1 for the EP window (0x50000); off 0..0xFFFF */
void gpdsp_mmio_write(int ep, uint32_t off, uint32_t v);
int  gpdsp_mmio_read(int ep, uint32_t off, uint32_t *v);  /* 1 when off is DSP memory */
/* Queue one 32-sample frame of the VP's mix bins. A DSP thread runs it and
 * hands the stereo result (interleaved, 32 samples; 256 every eighth frame
 * when the EP runs) to sink. */
void gpdsp_submit(float bins[GP_BINS][GP_FRAME], void (*sink)(const float *stereo, int frames));

/* Audio settings (implemented in apu.c): 1 = effects by the emulated Xbox DSP,
 * 0 = the built-in reverb approximation. Can change while the game runs. */
void apu_set_dsp_effects(int on);
