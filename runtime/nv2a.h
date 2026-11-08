/* nv2a.h — NV2A GPU model (register file, command FIFO, display). */
#pragma once
#include <stdint.h>

void nv2a_register_interrupt(uint32_t irq_level, uint32_t kinterrupt);  /* from KeConnectInterrupt */
void nv2a_tick(uint64_t now_ns);                                        /* ~1 kHz, GIL held */
void nv2a_host_vblank(void);    /* the host display refreshed (any thread): emulated vblanks follow it */
void nv2a_set_display(uint32_t fb, uint32_t pitch, uint32_t mode, uint32_t format); /* AvSetDisplayMode */
void kernel_raise_interrupt(uint32_t kinterrupt);                        /* kernel.c */

/* Counters for the host's performance overlay (read without the guest lock). */
uint32_t nv2a_stat_flips(void);     /* buffer flips requested by the game (its frame count) */
uint32_t nv2a_stat_vblanks(void);   /* emulated vertical blanks (60 per emulated second) */
