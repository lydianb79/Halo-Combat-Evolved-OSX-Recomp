#!/usr/bin/env python3
"""List / extract files from an original-Xbox disc image (XISO / XDVDFS).

Works on both "xiso" images (game partition only) and full Redump dumps
(game partition at a fixed offset). Reads the image in place; never modifies it.

  python3 tools/xiso.py list    "Halo.iso"
  python3 tools/xiso.py extract "Halo.iso" extracted/            # everything
  python3 tools/xiso.py extract "Halo.iso" extracted/ default.xbe  # selected files
"""
import os
import struct
import sys

SECTOR = 2048
MAGIC = b"MICROSOFT*XBOX*MEDIA"
# Offsets of the XDVDFS game partition inside the different dump layouts.
PARTITION_OFFSETS = (0, 0x18300000, 0xFD90000, 0x2080000)


class XISO:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.base = None
        for off in PARTITION_OFFSETS:
            self.f.seek(off + 32 * SECTOR)
            if self.f.read(20) == MAGIC:
                self.base = off
                break
        if self.base is None:
            raise SystemExit(f"{path}: not an Xbox disc image (no XDVDFS header found)")
        self.f.seek(self.base + 32 * SECTOR + 20)
        self.root_sector, self.root_size = struct.unpack("<II", self.f.read(8))

    def _read(self, sector, size):
        self.f.seek(self.base + sector * SECTOR)
        return self.f.read(size)

    def _dir(self, sector, size, prefix=""):
        """Walk an XDVDFS directory (an AVL tree stored in a flat table)."""
        if size == 0:
            return
        d = self._read(sector, size)
        stack, seen = [0], set()
        while stack:
            o = stack.pop()
            if o in seen or o * 4 + 14 > len(d):
                continue
            seen.add(o)
            b = o * 4
            left, right, start, length, attr, nlen = struct.unpack_from("<HHIIBB", d, b)
            if left == 0xFFFF:
                continue
            name = d[b + 14:b + 14 + nlen].decode("latin1")
            path = prefix + name
            is_dir = bool(attr & 0x10)
            yield path, start, length, is_dir
            if is_dir:
                yield from self._dir(start, length, path + "/")
            if left:
                stack.append(left)
            if right:
                stack.append(right)

    def entries(self):
        return sorted(self._dir(self.root_sector, self.root_size))

    def extract(self, start, length, out_path):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.f.seek(self.base + start * SECTOR)
        remaining = length
        with open(out_path, "wb") as o:
            while remaining:
                chunk = self.f.read(min(remaining, 8 << 20))
                if not chunk:
                    raise IOError("unexpected end of image")
                o.write(chunk)
                remaining -= len(chunk)


def main(argv):
    if len(argv) < 3 or argv[1] not in ("list", "extract"):
        print(__doc__)
        return 1
    iso = XISO(argv[2])
    ents = iso.entries()
    if argv[1] == "list":
        for path, _, length, is_dir in ents:
            print(f"{'<dir>' if is_dir else length:>12}  {path}")
        return 0
    out = argv[3] if len(argv) > 3 else "extracted"
    wanted = {w.lower() for w in argv[4:]}
    n = 0
    for path, start, length, is_dir in ents:
        if is_dir or (wanted and path.lower() not in wanted):
            continue
        dst = os.path.join(out, *path.split("/"))
        if os.path.exists(dst) and os.path.getsize(dst) == length:
            continue
        print(f"extract {path} ({length} bytes)")
        iso.extract(start, length, dst)
        n += 1
    print(f"done: {n} file(s) written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
