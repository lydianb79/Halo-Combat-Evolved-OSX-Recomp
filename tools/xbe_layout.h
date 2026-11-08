/* XBE image header field offsets (shared by tools and runtime). */
#pragma once
#define XBE_BASE              0x104
#define XBE_HEADERS_SIZE      0x108
#define XBE_IMAGE_SIZE        0x10C
#define XBE_NUM_SECTIONS      0x11C
#define XBE_SECTION_HEADERS   0x120
#define XBE_ENTRY             0x128
#define XBE_TLS               0x12C
#define XBE_STACK_COMMIT      0x130
#define XBE_KERNEL_THUNK      0x158
#define XBE_ENTRY_KEY_RETAIL  0xA8FC57ABu
#define XBE_ENTRY_KEY_DEBUG   0x94859D4Bu
#define XBE_THUNK_KEY_RETAIL  0x5B6D40B6u
#define XBE_THUNK_KEY_DEBUG   0xEFB1F152u
