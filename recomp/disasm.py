"""x86-32 disassembly via llvm-objdump (ships with Xcode Command Line Tools as
`objdump`, and with any LLVM install). The raw section bytes are wrapped in a
minimal ELF so objdump can place them at their real virtual address."""
import os
import re
import shutil
import struct
import subprocess
import tempfile

LINE_RE = re.compile(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ?)+?)\s*(?:\t(.*))?$")


def find_objdump():
    for name in ("llvm-objdump", "llvm-objdump-18", "llvm-objdump-17", "llvm-objdump-16",
                 "llvm-objdump-15", "/opt/homebrew/opt/llvm/bin/llvm-objdump",
                 "/usr/local/opt/llvm/bin/llvm-objdump", "objdump"):
        p = shutil.which(name)
        if p:
            try:
                out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
            except OSError:
                continue
            if "LLVM" in out or "llvm" in out:
                return p
    raise SystemExit("llvm-objdump not found. On macOS run `xcode-select --install`; "
                     "elsewhere install LLVM (apt install llvm).")


class Insn:
    __slots__ = ("addr", "size", "raw", "mnem", "ops", "prefix")

    def __init__(self, addr, raw, text):
        self.addr, self.raw = addr, raw
        self.size = len(raw)
        text = re.sub(r"\s*<[^>]*>", "", text or "")   # drop symbol annotations
        text = text.split(" # ")[0].strip()              # drop comments
        parts = text.split("\t")
        parts = [p for p in parts if p != ""]
        self.prefix = None
        if parts and parts[0] in ("rep", "repe", "repne", "repz", "repnz") and len(parts) > 1:
            self.prefix = parts[0]
            parts = parts[1:]
        self.mnem = parts[0].strip() if parts else "<unknown>"
        self.ops = split_operands(parts[1]) if len(parts) > 1 else []

    def __repr__(self):
        return f"{self.addr:x}: {(self.prefix + ' ') if self.prefix else ''}{self.mnem} {', '.join(self.ops)}"


def split_operands(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def make_elf(code, vaddr, path):
    shstr = b"\0.text\0.shstrtab\0"
    data_off = 0x1000
    shoff = (data_off + len(code) + len(shstr) + 3) & ~3
    e = bytearray(b"\x7fELF\x01\x01\x01" + b"\0" * 9)
    e += struct.pack("<HHIIIIIHHHHHH", 2, 3, 1, vaddr, 52, shoff, 0, 52, 32, 1, 40, 3, 2)
    e += struct.pack("<IIIIIIII", 1, data_off, vaddr, vaddr, len(code), len(code), 5, 0x1000)
    e += b"\0" * (data_off - len(e))
    e += code
    stroff = len(e)
    e += shstr
    e += b"\0" * (shoff - len(e))
    e += b"\0" * 40
    e += struct.pack("<IIIIIIIIII", 1, 1, 6, vaddr, data_off, len(code), 0, 0, 16, 0)
    e += struct.pack("<IIIIIIIIII", 7, 3, 0, 0, stroff, len(shstr), 0, 0, 1, 0)
    with open(path, "wb") as f:
        f.write(e)


class Disassembler:
    """Holds decoded instructions for one code region; decodes lazily on misses
    (e.g. code that a linear sweep mis-aligned after an inline jump table)."""

    def __init__(self, code, vaddr, workdir=None):
        self.objdump = find_objdump()
        self.vaddr, self.end = vaddr, vaddr + len(code)
        self.tmp = workdir or tempfile.mkdtemp(prefix="recomp_")
        self.elf = os.path.join(self.tmp, f"code_{vaddr:08x}.elf")
        make_elf(code, vaddr, self.elf)
        self.insns = {}
        self._run(None, None, overwrite=True)

    def _run(self, start, stop, overwrite):
        cmd = [self.objdump, "-d", "--x86-asm-syntax=intel", "--triple=i386-unknown-unknown"]
        if start is not None:
            cmd += [f"--start-address={start:#x}", f"--stop-address={stop:#x}"]
        cmd.append(self.elf)
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        last = None
        for line in out.splitlines():
            m = LINE_RE.match(line)
            if not m:
                continue
            addr = int(m.group(1), 16)
            raw = bytes.fromhex(m.group(2).replace(" ", ""))
            if m.group(3) is None:            # continuation of long raw-byte dump
                if last is not None:
                    last.raw += raw
                    last.size = len(last.raw)
                continue
            ins = Insn(addr, raw, m.group(3))
            if overwrite or addr not in self.insns:
                self.insns[addr] = ins
            last = ins

    def contains(self, addr):
        return self.vaddr <= addr < self.end

    def get(self, addr):
        ins = self.insns.get(addr)
        if ins is None and self.contains(addr):
            # Re-decode from this address; trust the new alignment from here on.
            stop = min(addr + 0x4000, self.end)
            for a in [a for a in self.insns if addr <= a < stop]:
                del self.insns[a]
            self._run(addr, stop, overwrite=True)
            ins = self.insns.get(addr)
        return ins
