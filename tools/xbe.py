#!/usr/bin/env python3
"""Parser for original-Xbox executables (XBE).

  python3 tools/xbe.py extracted/default.xbe          # human-readable summary
  python3 tools/xbe.py extracted/default.xbe --json   # machine-readable

Also importable: `from xbe import XBE`.
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_exports import KERNEL_EXPORTS  # noqa: E402

ENTRY_KEYS = {"retail": 0xA8FC57AB, "debug": 0x94859D4B}
THUNK_KEYS = {"retail": 0x5B6D40B6, "debug": 0xEFB1F152}

SEC_WRITABLE = 0x1
SEC_PRELOAD = 0x2
SEC_EXECUTABLE = 0x4


class Section:
    def __init__(self, name, flags, vaddr, vsize, raddr, rsize):
        self.name, self.flags = name, flags
        self.vaddr, self.vsize, self.raddr, self.rsize = vaddr, vsize, raddr, rsize

    @property
    def executable(self):
        return bool(self.flags & SEC_EXECUTABLE)

    def contains(self, va):
        return self.vaddr <= va < self.vaddr + max(self.vsize, self.rsize)

    def __repr__(self):
        return f"<Section {self.name} {self.vaddr:#x}+{self.vsize:#x}>"


class XBE:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        d = self.data
        if d[:4] != b"XBEH":
            raise ValueError(f"{path}: not an XBE (bad magic)")
        u32 = lambda o: struct.unpack_from("<I", d, o)[0]
        self.base = u32(0x104)
        self.headers_size = u32(0x108)
        self.image_size = u32(0x10C)
        self.timestamp = u32(0x114)
        self.cert_addr = u32(0x118)
        nsec, sec_addr = u32(0x11C), u32(0x120)
        self.init_flags = u32(0x124)
        enc_entry, self.tls_addr = u32(0x128), u32(0x12C)
        self.stack_commit = u32(0x130)
        self.heap_reserve, self.heap_commit = u32(0x134), u32(0x138)
        enc_thunk = u32(0x158)
        nlib, lib_addr = u32(0x160), u32(0x164)

        # Retail vs debug: pick the key that puts the entry point inside the image.
        self.kind = "retail"
        for kind, key in ENTRY_KEYS.items():
            ep = enc_entry ^ key
            if self.base <= ep < self.base + self.image_size:
                self.kind = kind
                break
        self.entry = enc_entry ^ ENTRY_KEYS[self.kind]
        self.kernel_thunk = enc_thunk ^ THUNK_KEYS[self.kind]

        self.sections = []
        for i in range(nsec):
            o = sec_addr - self.base + i * 56
            flags, va, vs, ra, rs, na = struct.unpack_from("<IIIIII", d, o)
            name = self.cstr(na)
            self.sections.append(Section(name, flags, va, vs, ra, rs))

        self.libraries = []
        for i in range(nlib):
            o = lib_addr - self.base + i * 16
            name = d[o:o + 8].split(b"\0")[0].decode("ascii", "replace")
            major, minor, build, flags = struct.unpack_from("<HHHH", d, o + 8)
            self.libraries.append({"name": name, "version": f"{major}.{minor}.{build}",
                                   "qfe": flags & 0x1FFF})

        if self.cert_addr:
            co = self.cert_addr - self.base
            self.title_id = u32(co + 8)
            self.title_name = d[co + 12:co + 12 + 80].decode("utf-16-le", "replace").split("\0")[0]
        else:
            self.title_id, self.title_name = 0, ""

        self.tls = None
        if self.tls_addr:
            t = self.read(self.tls_addr, 24)
            if t:
                s, e, idx, cb, zf, ch = struct.unpack("<IIIIII", t)
                self.tls = {"data_start": s, "data_end": e, "index_addr": idx,
                            "callbacks": cb, "zero_fill": zf}

    # --- address helpers -------------------------------------------------
    def cstr(self, va):
        o = va - self.base
        end = self.data.index(b"\0", o)
        return self.data[o:end].decode("ascii", "replace")

    def section_for(self, va):
        for s in self.sections:
            if s.contains(va):
                return s
        return None

    def read(self, va, size):
        """Read `size` bytes of the *loaded* image at virtual address `va`
        (headers or section contents). Bytes past a section's raw size are zero."""
        if self.base <= va < self.base + self.headers_size:
            o = va - self.base
            return self.data[o:o + size]
        s = self.section_for(va)
        if s is None:
            return None
        off = va - s.vaddr
        raw = self.data[s.raddr + off:s.raddr + min(off + size, s.rsize)] if off < s.rsize else b""
        return raw + b"\0" * (size - len(raw))

    def u32(self, va):
        b = self.read(va, 4)
        return struct.unpack("<I", b)[0] if b and len(b) == 4 else None

    def kernel_imports(self):
        """[(thunk_slot_va, ordinal, name)] from the kernel thunk table."""
        out, va = [], self.kernel_thunk
        while True:
            v = self.u32(va)
            if not v:
                break
            ordinal = v & 0x7FFFFFFF
            out.append((va, ordinal, KERNEL_EXPORTS.get(ordinal, f"Ordinal_{ordinal}")))
            va += 4
        return out

    def summary(self):
        return {
            "title": self.title_name, "title_id": f"{self.title_id:08X}", "kind": self.kind,
            "base": self.base, "image_size": self.image_size, "entry": self.entry,
            "kernel_thunk": self.kernel_thunk, "tls": self.tls,
            "stack_commit": self.stack_commit,
            "sections": [{"name": s.name, "vaddr": s.vaddr, "vsize": s.vsize, "raddr": s.raddr,
                          "rsize": s.rsize, "flags": s.flags} for s in self.sections],
            "libraries": self.libraries,
            "kernel_imports": [{"slot": a, "ordinal": o, "name": n} for a, o, n in self.kernel_imports()],
        }


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    x = XBE(argv[1])
    if "--json" in argv:
        print(json.dumps(x.summary(), indent=1))
        return 0
    print(f"Title      : {x.title_name} ({x.title_id:08X}), {x.kind} XBE")
    print(f"Base/size  : {x.base:#x} / {x.image_size:#x}")
    print(f"Entry point: {x.entry:#x}")
    print(f"Kernel thunk table: {x.kernel_thunk:#x}")
    if x.tls:
        print(f"TLS        : {x.tls}")
    print("\nSections:")
    for s in x.sections:
        fl = ("W" if s.flags & SEC_WRITABLE else "-") + ("X" if s.executable else "-") + \
             ("P" if s.flags & SEC_PRELOAD else "-")
        print(f"  {s.name:10} {s.vaddr:#010x} vsize {s.vsize:#9x} raw {s.rsize:#9x}  {fl}")
    print("\nXDK libraries:")
    for lib in x.libraries:
        print(f"  {lib['name']:8} {lib['version']}")
    imps = x.kernel_imports()
    print(f"\nKernel imports ({len(imps)}):")
    for slot, o, n in imps:
        print(f"  [{slot:#x}] {o:3} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
