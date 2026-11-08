#!/usr/bin/env python3
"""Static recompiler: Xbox XBE (x86-32) -> portable C.

  python3 recomp/recomp.py extracted/default.xbe generated/ [--seeds seeds.txt]

Output (all generated locally from YOUR copy of the game; never commit it):
  generated/recomp_funcs.h     prototypes
  generated/recomp_table.c     address -> function table for indirect calls
  generated/recomp_NNN.c       translated functions
  generated/report.json        coverage: functions, unsupported instructions, etc.
"""
import argparse
import collections
import json
import os
import re
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from disasm import Disassembler  # noqa: E402
from x86c import CC, ROT_HELPERS, Op, Unsupported, translate  # noqa: E402
from xbe import XBE  # noqa: E402

# Sections that hold code. Library sections (statically linked XDK code) are
# recompiled too, but calls into them go through call_indirect() so the
# runtime can replace them with high-level (HLE) implementations, e.g. Direct3D
# entry points -> Metal renderer.
DEFAULT_CODE_SECTIONS = [".text", "D3D", "D3DX", "DSOUND", "XNET", "XPP", "BINK", "BINK32",
                         "BINK32A", "BINK16", "BINK4444", "BINK5551", "BINK16MX", "BINK16X2",
                         "BINK16M", "BINK32MX", "BINK32X2", "BINK32M"]
HLE_SECTIONS = {"D3D", "D3DX", "DSOUND", "XNET", "XPP", "BINK", "BINK32", "BINK32A", "BINK16",
                "BINK4444", "BINK5551", "BINK16MX", "BINK16X2", "BINK16M", "BINK32MX", "BINK32X2",
                "BINK32M"}

TERMINATORS = {"ret", "retn", "jmp", "hlt", "ud2", "iretd", "retf", "ljmp"}


def jcc_cc(m):
    if m.startswith("j") and m[1:] in CC:
        return CC[m[1:]]
    return None


class Program:
    def __init__(self, xbe, code_sections, workdir):
        self.xbe = xbe
        self.regions = []
        for s in xbe.sections:
            if s.name in code_sections:
                code = xbe.read(s.vaddr, s.vsize)
                self.regions.append((s, Disassembler(code, s.vaddr, workdir)))
        self.thunk_lo = xbe.kernel_thunk
        self.thunk_hi = xbe.kernel_thunk + 4 * len(xbe.kernel_imports()) + 4

    def region(self, addr):
        for s, d in self.regions:
            if d.contains(addr):
                return s, d
        return None, None

    def insn(self, addr):
        s, d = self.region(addr)
        return d.get(addr) if d else None

    def is_code(self, addr):
        return self.region(addr)[0] is not None


class Function:
    def __init__(self, entry):
        self.entry = entry
        self.insns = {}          # addr -> Insn
        self.labels = set()
        self.jumptables = {}     # jmp insn addr -> [targets]
        self.calls = set()
        self.tailcalls = set()
        self.bad = []            # addresses that failed to decode


def find_jumptable(prog, fn_insns, ins, entry):
    """Resolve `jmp dword ptr [4*reg + TABLE]`.

    Bounds are not trusted (MSVC's memcpy indexes some tables with negative or
    partially-unused indices), so the table is scanned in both directions and
    every entry that points at nearby code becomes a switch case. Extra cases
    are harmless; missing ones would trap at runtime."""
    op = Op(ins.ops[0])
    if op.kind != "mem" or len(op.terms) != 1 or op.terms[0][1] != 4:
        return None
    table = op.disp
    lo, hi = min(entry, ins.addr) - 0x2000, ins.addr + 0x4000

    def ok(t):
        return t is not None and lo <= t < hi and prog.is_code(t) and prog.insn(t) is not None

    targets, misses = [], 0
    for i in range(1024):
        t = prog.xbe.u32(table + 4 * i)
        if ok(t):
            targets.append(t)
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    for i in range(1, 9):
        t = prog.xbe.u32(table - 4 * i)
        if not ok(t):
            break
        targets.append(t)
    return sorted(set(targets)) or None


def explore(prog, entry, entries):
    """Collect the body of the function at `entry` (intra-procedural edges only)."""
    fn = Function(entry)
    work = [entry]
    while work:
        a = work.pop()
        while a not in fn.insns:
            ins = prog.insn(a)
            if ins is None:
                fn.bad.append(a)
                break
            fn.insns[a] = ins
            m = ins.mnem
            nxt = a + ins.size
            cc = jcc_cc(m)
            if m == "call":
                o = ins.ops[0] if ins.ops else ""
                try:
                    op = Op(o)
                except Unsupported:
                    op = None
                if op is not None and op.kind == "imm":
                    fn.calls.add(op.value & 0xFFFFFFFF)
                a = nxt
                continue
            if cc or m in ("jecxz", "loop", "loope", "loopne"):
                t = Op(ins.ops[0]).value & 0xFFFFFFFF
                if t in entries and t != entry:
                    fn.tailcalls.add(t)
                else:
                    fn.labels.add(t)
                    work.append(t)
                a = nxt
                continue
            if m == "jmp":
                op = Op(ins.ops[0])
                if op.kind == "imm":
                    t = op.value & 0xFFFFFFFF
                    if (t in entries and t != entry) or not prog.is_code(t):
                        fn.tailcalls.add(t)
                    else:
                        fn.labels.add(t)
                        work.append(t)
                elif op.kind == "mem" and op.static_addr() is None:
                    tg = find_jumptable(prog, fn.insns, ins, entry)
                    if tg:
                        fn.jumptables[a] = tg
                        for t in tg:
                            fn.labels.add(t)
                            work.append(t)
                break
            if m in TERMINATORS:
                break
            if nxt in entries and nxt != entry:
                fn.tailcalls.add(nxt)   # falls through into another function
                break
            a = nxt
    return fn


def discover(prog, seeds):
    """Fixpoint over function entries. A function body is re-explored only when
    a newly discovered entry lands inside it (turning a jump into a tail call)."""
    entries = set(seeds)
    funcs = {}
    work = list(entries)
    while work:
        new_entries = set()
        for e in work:
            fn = explore(prog, e, entries)
            funcs[e] = fn
            for t in fn.calls:
                if prog.is_code(t) and t not in entries:
                    new_entries.add(t)
        if not new_entries:
            break
        entries |= new_entries
        stale = [e for e, f in funcs.items() if any(n in f.insns for n in new_entries)]
        work = list(new_entries) + stale
    return funcs


# First two bytes of common MSVC function prologues.
PROLOGUES = {b"\x55\x8b", b"\x83\xec", b"\x81\xec", b"\x8b\x44", b"\x8b\x4c", b"\x8b\x54", b"\x56\x8b",
             b"\x53\x8b", b"\x57\x8b", b"\x56\x57", b"\x53\x56", b"\x55\x56", b"\x55\x57", b"\x51\x53",
             b"\x51\x56", b"\x51\x8b", b"\x8b\x0d", b"\x8b\x15", b"\xd9\x44", b"\x8b\xc1", b"\x33\xc0"}


def data_pointer_seeds(prog):
    """Dwords in data sections that point at plausible function starts
    (vtables, callback tables, thread start routines)."""
    seeds = set()
    xbe = prog.xbe
    for s in xbe.sections:
        if s.name in (".rdata", ".data", "D3D", "DSOUND", "XPP", "XNET", "BINKDATA"):
            data = xbe.read(s.vaddr, s.rsize)
            for off in range(0, len(data) - 3, 4):
                v = struct.unpack_from("<I", data, off)[0]
                sec, dis = prog.region(v)
                if sec is None:
                    continue
                ins = dis.insns.get(v)
                if ins is None and s.name in HLE_SECTIONS and sec.name == s.name:
                    ins = prog.insn(v)          # library tables often point past mixed code/data
                prevb = xbe.read(v - 1, 1)
                if ins is None and v % 16 == 0:
                    ins = prog.insn(v)          # the linear sweep was out of step here (e.g. after a jump table)
                    if ins is not None:
                        seeds.add(v)            # 16-byte aligned and pointed to from data: a function
                        continue
                if ins is None:
                    continue
                if prevb in (b"\xcc", b"\x90", b"\xc3", b"\x00"):
                    seeds.add(v)
                elif v % 16 == 0 and xbe.read(v, 3)[:2] in PROLOGUES:
                    # A 16-byte aligned target that starts like an MSVC function
                    # (game callback tables: script functions, tag handlers)
                    seeds.add(v)
                elif s.name in HLE_SECTIONS and sec.name == s.name and ins.mnem != "<unknown>":
                    # XDK libraries keep their own callback/vtable tables next to their code
                    seeds.add(v)
    return seeds


def after_terminator_seeds(prog):
    """Addresses that follow a ret/jmp (plus 0xCC/0x90 padding) and decode as
    code: almost always function starts. Catches functions reached only via
    pointers built at runtime (callbacks, vtables in heap objects). Harmless
    when wrong: a jump into such an address becomes an equivalent tail call."""
    seeds = set()
    for sec, dis in prog.regions:
        addrs = sorted(dis.insns)
        for i, a in enumerate(addrs[:-1]):
            ins = dis.insns[a]
            if ins.mnem not in ("ret", "retn", "jmp"):
                continue
            j = i + 1
            while j < len(addrs) and dis.insns[addrs[j]].mnem in ("int3", "nop"):
                j += 1
            if j < len(addrs):
                n = dis.insns[addrs[j]]
                if n.mnem not in ("<unknown>", "int3", "add", "adc", "or", "sbb", "and", "xor") or \
                   n.raw[:1] in (b"\x55", b"\x53", b"\x56", b"\x57", b"\x83", b"\x8b", b"\x6a", b"\x68", b"\x51", b"\xa1", b"\xb8", b"\x33"):
                    if n.mnem != "<unknown>" and not n.raw.startswith(b"\x00\x00"):
                        seeds.add(addrs[j])
    return seeds


def imm_code_seeds(prog, funcs):
    """`push imm`/`mov reg, imm` whose value is a function-looking code address."""
    out = set()
    for fn in funcs.values():
        for ins in fn.insns.values():
            # Any immediate that lands on a decoded instruction is treated as a
            # possible entry. False positives are harmless: a jump into such an
            # "entry" becomes an equivalent tail call.
            if ins.mnem in ("push", "mov", "lea") and ins.ops:
                try:
                    o = Op(ins.ops[-1])
                except Unsupported:
                    continue
                if o.kind == "imm":
                    v = o.value & 0xFFFFFFFF
                    sec, dis = prog.region(v)
                    if sec is not None and v in dis.insns and dis.insns[v].mnem not in ("int3", "nop", "<unknown>"):
                        out.add(v)
    return out


# ------------------------------------------------------------------ emission
def fname(a):
    return f"f_{a:08x}"


class Emitter:
    def __init__(self, prog, funcs, hle_ranges):
        self.prog, self.funcs = prog, funcs
        self.hle_ranges = hle_ranges
        self.unsupported = collections.Counter()
        self.unsupported_sites = collections.defaultdict(list)

    def is_hle(self, a):
        return any(lo <= a < hi for lo, hi in self.hle_ranges)

    def call_stmt(self, target, cur_fn):
        """Statement that invokes guest code at `target` (return addr already pushed)."""
        if target in self.funcs and not (self.is_hle(target) and not self.is_hle(cur_fn.entry)):
            return f"{fname(target)}(c);"
        return f"call_indirect(c, 0x{target:x}u);"

    def emit_function(self, fn):
        addrs = sorted(fn.insns)
        chunks = []          # (addr, [lines])
        for i, a in enumerate(addrs):
            ins = fn.insns[a]
            nxt = a + ins.size
            lines = ["    " + b for b in self.emit_insn(fn, ins)]
            m = ins.mnem
            if m not in TERMINATORS and (i + 1 >= len(addrs) or addrs[i + 1] != nxt):
                # control continues somewhere that is not emitted next
                if nxt in fn.insns:
                    lines.append(f"    goto L_{nxt:x};")
                elif nxt in self.funcs:
                    lines.append(f"    {self.call_stmt(nxt, fn)} return;")
                else:
                    lines.append(f'    recomp_trap(c, 0x{nxt:x}, "fell off decoded code");')
            chunks.append((a, lines))
        # Blocks can sit at lower addresses than the entry point (code reached by
        # jumps backwards), so execution must start with an explicit jump to it.
        prologue = []
        if addrs and addrs[0] != fn.entry:
            prologue = [f"    goto L_{fn.entry:x};"]
        body = "\n".join(prologue + ["\n".join(l) for _, l in chunks])
        used = set(int(x, 16) for x in re.findall(r"goto L_([0-9a-f]+);", body))
        out = [f"void {fname(fn.entry)}(CPU *c) {{"] + prologue
        for a, lines in chunks:
            if a in used:
                out.append(f"L_{a:x}:;")
            out.extend(lines)
        out.append("}")
        missing = used - set(fn.insns)
        if missing:
            raise RuntimeError(f"{fname(fn.entry)}: goto to undecoded {sorted(missing)}")
        return "\n".join(out)

    def emit_insn(self, fn, ins):
        a, m = ins.addr, ins.mnem
        tag = f"/*@{a:x}*/ "
        try:
            body = self.control(fn, ins)
            if body is None:
                body = translate(ins, self)
            return [tag + s if k == 0 else s for k, s in enumerate(body or [";"])]
        except Unsupported as e:
            key = f"{(ins.prefix + ' ') if ins.prefix else ''}{m}"
            self.unsupported[key] += 1
            if len(self.unsupported_sites[key]) < 5:
                self.unsupported_sites[key].append(f"{a:x}")
            txt = repr(ins).replace("\\", "\\\\").replace('"', "'")
            return [f'{tag}UNIMPL(c, 0x{a:x}, "{txt}"); /* {e} */']
        except (ValueError, KeyError, IndexError, AttributeError) as e:
            self.unsupported[f"{m} (parse)"] += 1
            txt = repr(ins).replace("\\", "\\\\").replace('"', "'")
            return [f'{tag}UNIMPL(c, 0x{a:x}, "{txt}"); /* {type(e).__name__}: {e} */']

    def control(self, fn, ins):
        m, a = ins.mnem, ins.addr
        nxt = a + ins.size
        cc = jcc_cc(m)
        if cc or m in ("jecxz", "loop", "loope", "loopne"):
            t = Op(ins.ops[0]).value & 0xFFFFFFFF
            cond = {"jecxz": "c->ecx == 0", "loop": "--c->ecx != 0",
                    "loope": "--c->ecx != 0 && CC_E(c)", "loopne": "--c->ecx != 0 && CC_NE(c)"}.get(m)
            cond = cond or f"CC_{cc}(c)"
            if t in fn.insns:
                sp = "SAFEPOINT(c); " if t <= a else ""
                return [f"if ({cond}) {{ {sp}goto L_{t:x}; }}" if sp else f"if ({cond}) goto L_{t:x};"]
            return [f"if ({cond}) {{ {self.call_stmt(t, fn)} return; }}"]
        if m == "jmp":
            op = Op(ins.ops[0])
            if op.kind == "imm":
                t = op.value & 0xFFFFFFFF
                if t in fn.insns and t != fn.entry:
                    return [f"SAFEPOINT(c); goto L_{t:x};" if t <= a else f"goto L_{t:x};"]
                return [f"{self.call_stmt(t, fn)} return;"]
            if a in fn.jumptables:
                tg = fn.jumptables[a]
                cases = " ".join(f"case 0x{t:x}u: goto L_{t:x};" for t in sorted(set(tg)) if t in fn.insns)
                return [f"{{ uint32_t t_ = {op.read(4)}; switch (t_) {{ {cases} "
                        f"default: call_indirect(c, t_); return; }} }}"]
            return [f"{{ uint32_t t_ = {op.read(4)}; call_indirect(c, t_); return; }}"]
        if m == "call":
            op = Op(ins.ops[0])
            if op.kind == "imm":
                t = op.value & 0xFFFFFFFF
                if t == nxt:  # call $+5 ; pop reg  (get EIP idiom)
                    return [f"push32(c, 0x{nxt:x}u);"]
                return [f"push32(c, 0x{nxt:x}u); {self.call_stmt(t, fn)}"]
            return [f"{{ uint32_t t_ = {op.read(4)}; push32(c, 0x{nxt:x}u); call_indirect(c, t_); }}"]
        if m in ("ret", "retn"):
            n = Op(ins.ops[0]).value if ins.ops else 0
            return [f"c->esp += {4 + n}; return;"]
        if m in ("hlt", "ud2", "iretd", "retf", "ljmp", "lcall", "into", "int", "bound"):
            return [f'recomp_trap(c, 0x{a:x}, "{m}");']
        if m == "<unknown>":
            raise Unsupported("undecodable bytes")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xbe")
    ap.add_argument("outdir")
    ap.add_argument("--seeds", help="text file of extra function entry addresses (hex, one per line)")
    ap.add_argument("--funcs-per-file", type=int, default=400)
    ap.add_argument("--no-data-seeds", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    xbe = XBE(args.xbe)
    os.makedirs(args.outdir, exist_ok=True)
    work = os.path.join(args.outdir, ".work")
    os.makedirs(work, exist_ok=True)
    prog = Program(xbe, DEFAULT_CODE_SECTIONS, work)
    print(f"[recomp] decoded {sum(len(d.insns) for _, d in prog.regions)} instructions "
          f"in {len(prog.regions)} code sections ({time.time()-t0:.1f}s)")

    seeds = {xbe.entry}
    if not args.no_data_seeds:
        ds = data_pointer_seeds(prog)
        print(f"[recomp] {len(ds)} function pointers found in data sections")
        seeds |= ds
    if args.seeds and os.path.exists(args.seeds):
        for line in open(args.seeds):
            line = line.split("#")[0].strip()
            if line:
                seeds.add(int(line, 16))
    funcs = discover(prog, seeds)
    for _ in range(4):
        extra = imm_code_seeds(prog, funcs) - set(funcs)
        if not extra:
            break
        print(f"[recomp] {len(extra)} code addresses found in immediates; re-running discovery")
        funcs = discover(prog, set(funcs) | extra)
    if not args.no_data_seeds:
        # Code nothing reaches statically but that looks like a function start
        # (callbacks registered at runtime). Only *unreached* code is seeded, so
        # existing functions are not split.
        for _ in range(3):
            covered = set()
            for f in funcs.values():
                covered.update(f.insns)
            ts = after_terminator_seeds(prog) - covered - set(funcs)
            if not ts:
                break
            print(f"[recomp] {len(ts)} unreached function starts after ret/jmp")
            funcs = discover(prog, set(funcs) | ts)
    print(f"[recomp] {len(funcs)} functions, "
          f"{sum(len(f.insns) for f in funcs.values())} instructions reachable ({time.time()-t0:.1f}s)")

    hle_ranges = [(s.vaddr, s.vaddr + s.vsize) for s in xbe.sections if s.name in HLE_SECTIONS]
    em = Emitter(prog, funcs, hle_ranges)

    order = sorted(funcs)
    with open(os.path.join(args.outdir, "recomp_funcs.h"), "w") as h:
        h.write("/* generated by recomp/recomp.py — do not edit */\n#pragma once\n#include \"cpu.h\"\n\n")
        for a in order:
            h.write(f"void {fname(a)}(CPU *c);\n")
        h.write(ROT_HELPERS)

    nfiles = 0
    for i in range(0, len(order), args.funcs_per_file):
        chunk = order[i:i + args.funcs_per_file]
        path = os.path.join(args.outdir, f"recomp_{nfiles:03d}.c")
        with open(path, "w") as f:
            f.write("/* generated by recomp/recomp.py from the user's own game image — do not edit, do not redistribute */\n")
            f.write('#include "recomp_funcs.h"\n\n')
            for a in chunk:
                f.write(em.emit_function(funcs[a]))
                f.write("\n\n")
        nfiles += 1

    with open(os.path.join(args.outdir, "recomp_table.c"), "w") as f:
        f.write('/* generated */\n#include "recomp_funcs.h"\n#include "runtime.h"\n\n')
        f.write("const RecompFunc g_recomp_funcs[] = {\n")
        for a in order:
            f.write(f"    {{0x{a:08x}u, {fname(a)}}},\n")
        f.write("};\n")
        f.write(f"const uint32_t g_recomp_func_count = {len(order)};\n")
        f.write(f"const uint32_t g_recomp_entry = 0x{xbe.entry:08x}u;\n")

    sec_of = collections.Counter()
    for a in order:
        s, _ = prog.region(a)
        sec_of[s.name] += 1
    total_insns = sum(len(f.insns) for f in funcs.values())
    unsup_total = sum(em.unsupported.values())
    report = {
        "title": xbe.title_name, "entry": f"{xbe.entry:#x}",
        "functions": len(order), "instructions": total_insns,
        "functions_by_section": dict(sec_of),
        "unsupported_instructions": unsup_total,
        "coverage_percent": round(100.0 * (1 - unsup_total / max(total_insns, 1)), 3),
        "unsupported_by_mnemonic": dict(em.unsupported.most_common()),
        "unsupported_examples": dict(em.unsupported_sites),
        "decode_failures": sum(len(f.bad) for f in funcs.values()),
        "unresolved_indirect_jumps": sum(1 for f in funcs.values() for a, i in f.insns.items()
                                         if i.mnem == "jmp" and i.ops and "[" in i.ops[0]
                                         and a not in f.jumptables and Op(i.ops[0]).static_addr() is None),
        "c_files": nfiles,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"[recomp] wrote {nfiles} C files; coverage {report['coverage_percent']}% "
          f"({unsup_total} unsupported of {total_insns}) in {report['seconds']}s")
    if em.unsupported:
        print("[recomp] top unsupported:", ", ".join(f"{k}={v}" for k, v in em.unsupported.most_common(12)))


if __name__ == "__main__":
    main()
