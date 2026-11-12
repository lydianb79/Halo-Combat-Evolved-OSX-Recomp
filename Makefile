# Halo Recomp — build system
#
#   make extract     # pull default.xbe out of the disc image (reads ../*.iso)
#   make recomp      # translate default.xbe -> generated/*.c   (~1-5 min)
#   make app         # build "Halo Recomp.app" (Metal, native arm64/x86_64)
#   make headless    # build the CLI runner used for bring-up/debugging
#   make run         # build + launch the app
#   make debug       # run headless with kernel tracing
#
# Requirements on macOS: Xcode or the Command Line Tools (`xcode-select --install`)
# for clang, llvm-objdump (as `objdump`) and the Metal compiler; Python 3.
# The Metal shader compiler ships with full Xcode; with only the Command Line
# Tools the app falls back to compiling the shader at runtime.

ISO       ?= $(firstword $(wildcard ../*.iso) $(wildcard *.iso))
XBE       ?= extracted/default.xbe
GEN       := generated
BUILD     := build
APP       := $(BUILD)/Halo Recomp.app
APP_Q     := "$(APP)"

UNAME_S   := $(shell uname -s)
CC        ?= clang
OPT       ?= -O1
CFLAGS    += $(OPT) -g -fno-strict-aliasing -Wall -Wno-unused-parameter -Wno-unused-variable \
             -Wno-unused-label -Wno-unused-but-set-variable -Iruntime
GENFLAGS  := $(OPT) -fno-strict-aliasing -w -Iruntime -I$(GEN)
LDLIBS    := -lm -lpthread
JOBS      ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# GP audio DSP: the dsp56300 JIT emulator (MIT, github.com/mborgerson/dsp56300),
# a prebuilt static library downloaded on first build. DSP=0 builds without it
# (the APU then uses its built-in reverb approximation).
DSP       ?= 1
DSP_VER   := 0.1.3
ifeq ($(UNAME_S),Darwin)
DSP_ARCH  := $(if $(filter arm64,$(shell uname -m)),aarch64,x86_64)-apple-darwin
DSP_SYS   := -framework Security
else
DSP_ARCH  := $(shell uname -m)-unknown-linux-gnu
DSP_SYS   := -ldl
endif
DSP_DIR   := $(BUILD)/dsp56300-$(DSP_VER)-$(DSP_ARCH)
DSP_LIB   := $(DSP_DIR)/lib/libdsp56300_emu_ffi.a
ifeq ($(DSP),1)
CFLAGS    += -DHALO_DSP -I$(DSP_DIR)/include
DSP_DEP   := $(DSP_LIB)
LDLIBS    += $(DSP_LIB) $(DSP_SYS) -lpthread -lm
endif

RUNTIME_SRC := runtime/runtime.c runtime/kernel.c runtime/vfs.c runtime/hle.c runtime/nv2a.c runtime/nv2a_raster.c runtime/xinput.c runtime/apu.c runtime/gpdsp.c
RUNTIME_OBJ := $(RUNTIME_SRC:runtime/%.c=$(BUILD)/rt_%.o)
GEN_SRC     := $(sort $(wildcard $(GEN)/recomp_*.c))
GEN_OBJ     := $(patsubst $(GEN)/%.c,$(BUILD)/gen_%.o,$(GEN_SRC))

.PHONY: all extract recomp app headless run debug clean distclean report

all: app

extract: $(XBE)
$(XBE):
	@test -n "$(ISO)" || (echo "No .iso found next to HaloRecomp/. Set ISO=/path/to/Halo.iso"; exit 1)
	python3 tools/xiso.py extract "$(ISO)" extracted default.xbe

recomp: $(GEN)/recomp_table.c
$(GEN)/recomp_table.c: $(XBE) recomp/recomp.py recomp/x86c.py recomp/disasm.py
	python3 recomp/recomp.py $(XBE) $(GEN) --seeds seeds.txt
	@$(MAKE) --no-print-directory build-dirs

report:
	@python3 -c "import json;r=json.load(open('$(GEN)/report.json'));print(json.dumps({k:r[k] for k in ('functions','instructions','coverage_percent','unsupported_instructions')},indent=1))"

build-dirs:
	@mkdir -p $(BUILD)

$(BUILD)/rt_%.o: runtime/%.c runtime/*.h | build-dirs $(DSP_DEP)
	$(CC) $(CFLAGS) -O2 -c $< -o $@

$(BUILD)/gen_%.o: $(GEN)/%.c $(GEN)/recomp_funcs.h runtime/cpu.h | build-dirs
	$(CC) $(GENFLAGS) -c $< -o $@

# The generated sources only exist after `make recomp`; re-invoke make so the
# wildcard picks them up.
headless: $(GEN)/recomp_table.c
	@$(MAKE) --no-print-directory -j$(JOBS) $(BUILD)/halo-headless

$(BUILD)/halo-headless: $(RUNTIME_OBJ) $(GEN_OBJ) host/headless_main.c host/gpu_probe.c
	$(CC) $(CFLAGS) host/headless_main.c host/gpu_probe.c $(RUNTIME_OBJ) $(GEN_OBJ) -o $@ $(LDLIBS)

ifeq ($(UNAME_S),Darwin)
app: $(GEN)/recomp_table.c
	@$(MAKE) --no-print-directory -j$(JOBS) app-bundle

app-bundle: $(RUNTIME_OBJ) $(GEN_OBJ) platform/macos/main.m platform/macos/input.m platform/macos/input.h platform/macos/graphics.m platform/macos/graphics.h platform/macos/settings.m platform/macos/settings.h platform/macos/audio.m platform/macos/audio.h platform/macos/metal_backend.m platform/macos/metal_backend.h platform/macos/Shaders.metal platform/macos/Info.plist platform/macos/AppIcon.icns
	@mkdir -p $(APP_Q)/Contents/MacOS $(APP_Q)/Contents/Resources
	$(CC) $(CFLAGS) -fobjc-arc platform/macos/main.m platform/macos/input.m platform/macos/graphics.m platform/macos/settings.m platform/macos/audio.m platform/macos/metal_backend.m $(RUNTIME_OBJ) $(GEN_OBJ) -o $(APP_Q)/Contents/MacOS/HaloRecomp \
	    $(LDLIBS) -framework Cocoa -framework Metal -framework MetalKit -framework GameController -framework QuartzCore -framework Carbon -framework AVFoundation
	cp platform/macos/Info.plist $(APP_Q)/Contents/Info.plist
	cp platform/macos/AppIcon.icns $(APP_Q)/Contents/Resources/AppIcon.icns
	@if [ "$(DSP)" = 1 ]; then cp $(DSP_DIR)/LICENSE $(APP_Q)/Contents/Resources/dsp56300-LICENSE.txt; cp $(DSP_DIR)/THIRDPARTY-emu $(APP_Q)/Contents/Resources/dsp56300-THIRDPARTY.txt; fi
	@touch $(APP_Q)
	@if xcrun -sdk macosx metal -c platform/macos/Shaders.metal -o $(BUILD)/Shaders.air 2>$(BUILD)/metal.log && \
	    xcrun -sdk macosx metallib $(BUILD)/Shaders.air -o $(APP_Q)/Contents/Resources/default.metallib 2>>$(BUILD)/metal.log; then \
	    echo "compiled Metal shaders"; \
	else \
	    echo "note: Metal compiler unavailable (see $(BUILD)/metal.log); shader will be compiled at launch"; \
	    cp platform/macos/Shaders.metal $(APP_Q)/Contents/Resources/Shaders.metal; \
	fi
	codesign --force --sign - $(APP_Q) 2>/dev/null || true
	@echo "built $(APP)"

run: app
	open $(APP_Q)
else
app:
	@echo "The Metal app builds on macOS only. Use 'make headless' here."
endif

debug: headless
	$(BUILD)/halo-headless --xbe $(XBE) --iso "$(ISO)" --save-root xbox_hdd --trace-kernel

$(DSP_LIB):
	@mkdir -p $(BUILD)
	curl -fsSL https://github.com/mborgerson/dsp56300/releases/download/v$(DSP_VER)/dsp56300-$(DSP_VER)-$(DSP_ARCH).tar.gz -o $(BUILD)/dsp56300.tar.gz
	tar xzf $(BUILD)/dsp56300.tar.gz -C $(BUILD)
	@rm -f $(BUILD)/dsp56300.tar.gz
	@touch $@

clean:
	rm -rf $(BUILD)

# Everything derived from the disc except your saves (xbox_hdd/ is kept).
distclean: clean
	rm -rf $(GEN) extracted missing_seeds.txt
