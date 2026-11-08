"""Translate single x86-32 instructions (llvm-objdump Intel syntax) into C
statements against the CPU model in runtime/cpu.h.

Control-flow instructions (jmp/jcc/call/ret) are handled by the function emitter
in recomp.py; this module handles everything else and exposes operand helpers."""
import re

REG32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
REG16 = {"ax": "eax", "cx": "ecx", "dx": "edx", "bx": "ebx", "sp": "esp", "bp": "ebp",
         "si": "esi", "di": "edi"}
REG8L = {"al": "eax", "cl": "ecx", "dl": "edx", "bl": "ebx"}
REG8H = {"ah": "eax", "ch": "ecx", "dh": "edx", "bh": "ebx"}
SIZES = {"byte": 1, "word": 2, "dword": 4, "qword": 8, "tbyte": 10, "xmmword": 16, "fword": 6}
UT = {1: "uint8_t", 2: "uint16_t", 4: "uint32_t", 8: "uint64_t"}
ST = {1: "int8_t", 2: "int16_t", 4: "int32_t", 8: "int64_t"}
CC = {"o": "O", "no": "NO", "b": "B", "c": "B", "nae": "B", "ae": "AE", "nb": "AE", "nc": "AE",
      "e": "E", "z": "E", "ne": "NE", "nz": "NE", "be": "BE", "na": "BE", "a": "A", "nbe": "A",
      "s": "S", "ns": "NS", "p": "P", "pe": "P", "np": "NP", "po": "NP", "l": "L", "nge": "L",
      "ge": "GE", "nl": "GE", "le": "LE", "ng": "LE", "g": "G", "nle": "G"}


class Unsupported(Exception):
    pass


def parse_int(s):
    s = s.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    v = int(s, 16) if s.lower().startswith("0x") else int(s)
    return -v if neg else v


class Op:
    """A parsed operand: kind in {'reg','imm','mem','st','xmm','mm','seg'}."""

    def __init__(self, text):
        self.text = t = text.strip()
        self.size = None
        self.kind = None
        if t in REG32:
            self.kind, self.base, self.size, self.part = "reg", t, 4, "full"
        elif t in REG16:
            self.kind, self.base, self.size, self.part = "reg", REG16[t], 2, "16"
        elif t in REG8L:
            self.kind, self.base, self.size, self.part = "reg", REG8L[t], 1, "8l"
        elif t in REG8H:
            self.kind, self.base, self.size, self.part = "reg", REG8H[t], 1, "8h"
        elif t == "st" or re.fullmatch(r"st\(\d\)", t):
            self.kind, self.idx = "st", (0 if t == "st" else int(t[3]))
        elif re.fullmatch(r"xmm\d", t):
            self.kind, self.idx, self.size = "xmm", int(t[3]), 16
        elif re.fullmatch(r"mm\d", t):
            self.kind, self.idx, self.size = "mm", int(t[2]), 8
        elif t in ("cs", "ds", "es", "fs", "gs", "ss"):
            self.kind = "seg"
        elif re.fullmatch(r"-?(0x[0-9a-f]+|\d+)", t):
            self.kind, self.value = "imm", parse_int(t)
        elif "[" in t:
            self.kind = "mem"
            m = re.match(r"^(?:(\w+) ptr )?(?:(\w+):)?\[(.*)\]$", t)
            if not m:
                raise Unsupported(f"mem operand {t!r}")
            sz, seg, expr = m.groups()
            if sz:
                if sz not in SIZES:
                    raise Unsupported(f"size {sz}")
                self.size = SIZES[sz]
            self.seg = seg
            self.terms, self.disp = [], 0
            for sign, term in re.findall(r"([+-]?)\s*([^+-]+)", expr.replace(" ", "")):
                if term == "eiz" or term.endswith("*eiz") or term.startswith("eiz*"):
                    continue                       # pseudo zero index (padding NOP forms)
                if term in REG32:
                    if sign == "-":
                        raise Unsupported("negative register term")
                    self.terms.append((term, 1))
                elif "*" in term:
                    a, b = term.split("*")
                    scale, reg = (int(a), b) if a.isdigit() else (int(b), a)
                    if reg not in REG32:
                        raise Unsupported(f"index {term}")
                    self.terms.append((reg, scale))
                else:
                    v = parse_int(term)
                    self.disp += -v if sign == "-" else v
            self.disp &= 0xFFFFFFFF
        else:
            raise Unsupported(f"operand {t!r}")

    # --- memory -----------------------------------------------------------
    def addr(self):
        assert self.kind == "mem"
        parts = []
        for reg, scale in self.terms:
            parts.append(f"c->{reg}" + (f"*{scale}u" if scale != 1 else ""))
        if self.disp or not parts:
            parts.append(f"0x{self.disp:x}u")
        if self.seg == "fs":
            parts.append("c->fs_base")
        e = " + ".join(parts)
        return f"(uint32_t)({e})" if len(parts) > 1 else e

    def static_addr(self):
        """Absolute address if the operand is [disp] only, else None."""
        if self.kind == "mem" and not self.terms and self.seg != "fs":
            return self.disp
        return None

    # --- generic read/write ---------------------------------------------
    def read(self, size=None):
        size = size or self.size
        if self.kind == "reg":
            r = f"c->{self.base}"
            return {"full": r, "16": f"(uint16_t){r}", "8l": f"(uint8_t){r}",
                    "8h": f"(uint8_t)({r} >> 8)"}[self.part]
        if self.kind == "imm":
            if size is None:
                size = 4
            return f"0x{self.value & ((1 << (8 * min(size, 4))) - 1):x}u"
        if self.kind == "mem":
            if size is None:
                raise Unsupported("memory operand without size")
            fn = {1: "rd8", 2: "rd16", 4: "rd32", 8: "rd64"}.get(size)
            if not fn:
                raise Unsupported(f"read size {size}")
            return f"{fn}({self.addr()})"
        raise Unsupported(f"read {self.kind}")

    def write(self, val, size=None):
        size = size or self.size
        if self.kind == "reg":
            r = f"c->{self.base}"
            if self.part == "full":
                return f"{r} = {val};"
            fn = {"16": "set16", "8l": "set8l", "8h": "set8h"}[self.part]
            return f"{fn}(&{r}, {val});"
        if self.kind == "mem":
            fn = {1: "wr8", 2: "wr16", 4: "wr32", 8: "wr64"}.get(size)
            if not fn:
                raise Unsupported(f"write size {size}")
            return f"{fn}({self.addr()}, ({UT[size]})({val}));"
        raise Unsupported(f"write {self.kind}")


def opsize(ops):
    for o in ops:
        if o.kind in ("reg", "mem") and o.size:
            return o.size
    return 4


def sext(expr, size):
    return f"(uint32_t)(int32_t)({ST[size]})({expr})" if size < 4 else expr


ALU = {"add": ("+", "LF_ADD"), "sub": ("-", "LF_SUB"), "and": ("&", "LF_LOGIC"),
       "or": ("|", "LF_LOGIC"), "xor": ("^", "LF_LOGIC")}


def translate(ins, ctx):
    """Return a list of C statement strings for a non-control-flow instruction.
    `ctx` gives access to the XBE (for constant reads) — unused for most ops."""
    m, raw = ins.mnem, ins.raw
    if m in ("nop", "wait", "fwait", "lock", "wbinvd", "invd", "invlpg", "sfence", "lfence", "mfence", "prefetcht0", "prefetcht1", "prefetcht2",
             "prefetchnta", "femms", "emms", "cli", "sti", "pause"):
        return []
    ops = [Op(o) for o in ins.ops]
    a = ins.addr
    L = []

    if ins.prefix and m not in STRING_OPS:
        raise Unsupported(f"{ins.prefix} prefix on {m}")

    # ---------------------------------------------------------- data moves
    if m == "mov":
        d, s = ops
        if d.kind == "seg" or s.kind == "seg":
            return [f"/* {ins} (segment move ignored) */"] if s.kind != "seg" else [d.write("0")]
        sz = d.size or s.size or 4
        return [d.write(s.read(sz), sz)]
    if m in ("movzx", "movsx"):
        d, s = ops
        v = s.read()
        if m == "movsx":
            v = f"(uint32_t)(int32_t)({ST[s.size]})({v})"
        else:
            v = f"(uint32_t)({v})"
        return [d.write(v)]
    if m == "lea":
        d, s = ops
        return [d.write(s.addr() if d.size == 4 else f"(uint16_t){s.addr()}")]
    if m == "xchg":
        d, s = ops
        sz = opsize(ops)
        return [f"{{ {UT[sz]} t_ = {d.read(sz)}; {d.write(s.read(sz), sz)} {s.write('t_', sz)} }}"]
    if m == "push":
        (s,) = ops
        if s.kind == "seg":
            return ["push32(c, 0);"]
        if s.size == 2:
            return [f"push16(c, {s.read()});"]
        return [f"{{ uint32_t t_ = {s.read(4)}; push32(c, t_); }}"]
    if m == "pop":
        (d,) = ops
        if d.kind == "seg":
            return ["c->esp += 4;"]
        if d.size == 2:
            return [f"{{ uint32_t t_ = pop16(c); {d.write('t_')} }}"]
        if d.kind == "mem" and any(r == "esp" for r, _ in d.terms):
            return [f"{{ uint32_t t_ = pop32(c); {d.write('t_')} }}"]  # esp already incremented
        return [f"{{ uint32_t t_ = pop32(c); {d.write('t_')} }}"]
    if m == "pushal":
        return ["{ uint32_t t_ = c->esp; push32(c, c->eax); push32(c, c->ecx); push32(c, c->edx); "
                "push32(c, c->ebx); push32(c, t_); push32(c, c->ebp); push32(c, c->esi); push32(c, c->edi); }"]
    if m == "popal":
        return ["c->edi = pop32(c); c->esi = pop32(c); c->ebp = pop32(c); c->esp += 4; "
                "c->ebx = pop32(c); c->edx = pop32(c); c->ecx = pop32(c); c->eax = pop32(c);"]
    if m == "pushfd":
        return ["push32(c, flags_get(c));"]
    if m == "popfd":
        return ["flags_set(c, pop32(c));"]
    if m == "lahf":
        return ["set8h(&c->eax, flags_get(c) & 0xD7);"]
    if m == "sahf":
        return ["flags_set(c, (flags_get(c) & ~0xD5u) | ((c->eax >> 8) & 0xD5u));"]
    if m == "cld":
        return ["c->df = 0;"]
    if m == "std":
        return ["c->df = 1;"]
    if m in ("clc", "stc", "cmc"):
        op = {"clc": "f & ~1u", "stc": "f | 1u", "cmc": "f ^ 1u"}[m]
        return [f"{{ uint32_t f = flags_get(c); flags_set(c, {op}); }}"]
    if m == "cdq":
        return ["c->edx = (uint32_t)((int32_t)c->eax >> 31);"]
    if m == "cwde":
        return ["c->eax = (uint32_t)(int32_t)(int16_t)c->eax;"]
    if m == "cbw":
        return ["set16(&c->eax, (uint16_t)(int16_t)(int8_t)c->eax);"]
    if m == "cwd":
        return ["set16(&c->edx, (uint16_t)((int16_t)c->eax >> 15));"]
    if m == "bswap":
        (d,) = ops
        return [d.write(f"__builtin_bswap32({d.read()})")]
    if m == "xlatb":
        return ["set8l(&c->eax, rd8((uint32_t)(c->ebx + (uint8_t)c->eax)));"]
    if m.startswith("set") and m[3:] in CC:
        (d,) = ops
        return [d.write(f"CC_{CC[m[3:]]}(c) ? 1 : 0", 1)]
    if m.startswith("cmov") and m[4:] in CC:
        d, s = ops
        return [f"if (CC_{CC[m[4:]]}(c)) {d.write(s.read(d.size))}"]

    # ---------------------------------------------------------- ALU
    if m in ALU or m in ("cmp", "test"):
        d, s = ops
        sz = opsize(ops)
        op, lf = ALU.get(m, ("-", "LF_SUB") if m == "cmp" else ("&", "LF_LOGIC"))
        sv = s.read(sz) if s.kind != "imm" else f"0x{s.value & ((1 << 8*sz) - 1):x}u"
        t = UT[sz]
        if m in ("xor", "sub") and d.kind == "reg" and s.kind == "reg" and d.text == s.text:
            L.append(d.write("0", sz))
            L.append(f"setf(c, LF_LOGIC, {sz}, 0, 0, 0);")
            return L
        L.append(f"{{ {t} a_ = {d.read(sz)}, b_ = {sv}, r_ = ({t})(a_ {op} b_);")
        if m not in ("cmp", "test"):
            L.append("  " + d.write("r_", sz))
        L.append(f"  setf(c, {lf}, {sz}, a_, b_, r_); }}")
        return L
    if m in ("adc", "sbb"):
        d, s = ops
        sz = opsize(ops)
        t = UT[sz]
        op, lf = ("+", "LF_ADC") if m == "adc" else ("-", "LF_SBB")
        sv = s.read(sz) if s.kind != "imm" else f"0x{s.value & ((1 << 8*sz) - 1):x}u"
        L.append(f"{{ {t} a_ = {d.read(sz)}, b_ = {sv}; uint32_t ci_ = f_cf(c) ? 1 : 0; "
                 f"{t} r_ = ({t})(a_ {op} b_ {op} ci_);")
        L.append("  " + d.write("r_", sz))
        L.append(f"  setf(c, {lf}, {sz}, a_, b_, r_); c->lf_cin = ci_; }}")
        return L
    if m in ("inc", "dec"):
        (d,) = ops
        sz = d.size
        t = UT[sz]
        op = "+" if m == "inc" else "-"
        return [f"{{ uint32_t cf_ = f_cf(c) ? 1 : 0; {t} a_ = {d.read()}, r_ = ({t})(a_ {op} 1);",
                "  " + d.write("r_"),
                f"  setf(c, {'LF_INC' if m == 'inc' else 'LF_DEC'}, {sz}, a_, 1, r_); c->lf_cin = cf_; }}"]
    if m == "neg":
        (d,) = ops
        t = UT[d.size]
        return [f"{{ {t} a_ = {d.read()}, r_ = ({t})(0 - a_);", "  " + d.write("r_"),
                f"  setf(c, LF_NEG, {d.size}, a_, 0, r_); }}"]
    if m == "not":
        (d,) = ops
        return [d.write(f"~{d.read()}")]
    if m in ("shl", "sal", "shr", "sar"):
        d, s = (ops + [None])[:2]
        sz = d.size
        t = UT[sz]
        cnt = "1" if s is None else (f"{s.value & 31}" if s.kind == "imm" else f"({s.read(1)} & 31)")
        if m in ("shl", "sal"):
            expr, lf = f"({t})((uint32_t)a_ << n_)", "LF_SHL"
        elif m == "shr":
            expr, lf = f"({t})((uint32_t)a_ >> n_)", "LF_SHR"
        else:
            expr, lf = f"({t})(({ST[sz]})a_ >> (n_ > {8*sz-1} ? {8*sz-1} : n_))", "LF_SAR"
        if s is not None and s.kind == "imm" and (s.value & 31) == 0:
            return []
        return [f"{{ uint32_t n_ = {cnt}; if (n_) {{ {t} a_ = {d.read()}; {t} r_ = {expr};",
                "  " + d.write("r_"),
                f"  setf(c, {lf}, {sz}, a_, n_, r_); }} }}"]
    if m in ("rol", "ror", "rcl", "rcr"):
        d, s = (ops + [None])[:2]
        sz = d.size
        cnt = "1" if s is None else (f"{s.value & 31}" if s.kind == "imm" else f"({s.read(1)} & 31)")
        return [f"{{ uint32_t n_ = {cnt}; if (n_) {{ uint32_t v_ = {d.read()}; "
                f"v_ = rot_{m}(c, v_, n_, {sz}); {d.write('v_')} }} }}"]
    if m in ("shld", "shrd"):
        d, s, n = ops
        cnt = f"{n.value & 31}" if n.kind == "imm" else f"({n.read(1)} & 31)"
        if d.size != 4:
            raise Unsupported("16-bit shld/shrd")
        if m == "shld":
            e = "(uint32_t)((((uint64_t)a_ << 32) | b_) << n_ >> 32)"
            cf = "(a_ >> (32 - n_)) & 1"
        else:
            e = "(uint32_t)((((uint64_t)b_ << 32) | a_) >> n_)"
            cf = "(a_ >> (n_ - 1)) & 1"
        return [f"{{ uint32_t n_ = {cnt}; if (n_) {{ uint32_t a_ = {d.read()}, b_ = {s.read()}, r_ = {e};",
                "  " + d.write("r_"),
                f"  uint32_t f_ = flags_get(c) & ~(FL_CF|FL_ZF|FL_SF|FL_PF|FL_OF); "
                f"if ({cf}) f_ |= FL_CF; if (!r_) f_ |= FL_ZF; if (r_ >> 31) f_ |= FL_SF; "
                f"if (!__builtin_parity(r_ & 0xFF)) f_ |= FL_PF; if ((r_ ^ a_) >> 31) f_ |= FL_OF; "
                f"flags_set(c, f_); }} }}"]
    if m in ("bt", "bts", "btr", "btc"):
        d, s = ops
        if d.kind == "mem" and s.kind == "reg":
            addr = f"(uint32_t)({d.addr()} + (uint32_t)(((int32_t){s.read()} >> 5) * 4))"
            bit = f"({s.read()} & 31)"
            rd, wr = f"rd32(a2_)", "wr32(a2_, v_);"
            pre = f"uint32_t a2_ = {addr}; "
        else:
            bit = f"({s.value & (8*d.size-1)})" if s.kind == "imm" else f"({s.read()} & {8*d.size-1})"
            rd, wr, pre = d.read(), d.write("v_"), ""
        upd = {"bt": "", "bts": "v_ |= 1u << b_;", "btr": "v_ &= ~(1u << b_);",
               "btc": "v_ ^= 1u << b_;"}[m]
        return [f"{{ {pre}uint32_t b_ = {bit}; uint32_t v_ = {rd}; uint32_t f_ = flags_get(c) & ~FL_CF; "
                f"if ((v_ >> b_) & 1) f_ |= FL_CF; flags_set(c, f_); {upd} {wr if upd else ''} }}"]
    if m in ("bsf", "bsr"):
        d, s = ops
        fn = "__builtin_ctz(v_)" if m == "bsf" else f"(31 - __builtin_clz(v_))"
        return [f"{{ uint32_t v_ = {s.read()}; uint32_t f_ = flags_get(c) & ~FL_ZF; "
                f"if (!v_) f_ |= FL_ZF; else {{ {d.write(fn)} }} flags_set(c, f_); }}"]
    if m == "imul":
        if len(ops) == 1:
            (s,) = ops
            sz = s.size
            if sz == 4:
                return [f"{{ int64_t r_ = (int64_t)(int32_t)c->eax * (int32_t){s.read()}; "
                        "c->eax = (uint32_t)r_; c->edx = (uint32_t)((uint64_t)r_ >> 32); "
                        "setf(c, LF_MUL, 4, 0, r_ != (int32_t)r_, (uint32_t)r_); }"]
            if sz == 2:
                return [f"{{ int32_t r_ = (int16_t)c->eax * (int16_t){s.read()}; set16(&c->eax, r_); "
                        "set16(&c->edx, (uint32_t)r_ >> 16); setf(c, LF_MUL, 2, 0, r_ != (int16_t)r_, r_); }"]
            return [f"{{ int32_t r_ = (int8_t)c->eax * (int8_t){s.read()}; set16(&c->eax, r_); "
                    "setf(c, LF_MUL, 1, 0, r_ != (int8_t)r_, r_); }"]
        if len(ops) == 2:
            d, s = ops
            x, y = d, s
        else:
            d, x, y = ops
        sz = d.size
        yv = f"{y.value}" if y.kind == "imm" else f"({ST[sz]}){y.read(sz)}"
        return [f"{{ int64_t r_ = (int64_t)({ST[sz]}){x.read(sz)} * (int64_t){yv}; "
                f"{d.write(f'({UT[sz]})r_')} setf(c, LF_MUL, {sz}, 0, r_ != ({ST[sz]})r_, (uint32_t)r_); }}"]
    if m == "mul":
        (s,) = ops
        sz = s.size
        if sz == 4:
            return [f"{{ uint64_t r_ = (uint64_t)c->eax * {s.read()}; c->eax = (uint32_t)r_; "
                    "c->edx = (uint32_t)(r_ >> 32); setf(c, LF_MUL, 4, 0, c->edx != 0, c->eax); }"]
        if sz == 2:
            return [f"{{ uint32_t r_ = (uint16_t)c->eax * (uint32_t){s.read()}; set16(&c->eax, r_); "
                    "set16(&c->edx, r_ >> 16); setf(c, LF_MUL, 2, 0, (r_ >> 16) != 0, r_); }"]
        return [f"{{ uint32_t r_ = (uint8_t)c->eax * (uint32_t){s.read()}; set16(&c->eax, r_); "
                "setf(c, LF_MUL, 1, 0, (r_ >> 8) != 0, r_); }"]
    if m in ("div", "idiv"):
        (s,) = ops
        sz = s.size
        trap = f'recomp_trap(c, 0x{a:x}, "divide error");'
        if sz == 4:
            if m == "div":
                return [f"{{ uint32_t d_ = {s.read()}; if (!d_) {trap} uint64_t n_ = ((uint64_t)c->edx << 32) | c->eax; "
                        f"uint64_t q_ = n_ / d_; if (q_ >> 32) {trap} c->edx = (uint32_t)(n_ % d_); c->eax = (uint32_t)q_; }}"]
            return [f"{{ int32_t d_ = (int32_t){s.read()}; if (!d_) {trap} int64_t n_ = (int64_t)(((uint64_t)c->edx << 32) | c->eax); "
                    f"if (n_ == INT64_MIN && d_ == -1) {trap} int64_t q_ = n_ / d_; if (q_ != (int32_t)q_) {trap} "
                    f"c->edx = (uint32_t)(int32_t)(n_ % d_); c->eax = (uint32_t)(int32_t)q_; }}"]
        if sz == 2:
            if m == "div":
                return [f"{{ uint32_t d_ = {s.read()}; if (!d_) {trap} uint32_t n_ = ((c->edx & 0xFFFF) << 16) | (c->eax & 0xFFFF); "
                        f"set16(&c->eax, n_ / d_); set16(&c->edx, n_ % d_); }}"]
            return [f"{{ int32_t d_ = (int16_t){s.read()}; if (!d_) {trap} int32_t n_ = (int32_t)(((c->edx & 0xFFFF) << 16) | (c->eax & 0xFFFF)); "
                    f"set16(&c->eax, (uint32_t)(n_ / d_)); set16(&c->edx, (uint32_t)(n_ % d_)); }}"]
        if m == "div":
            return [f"{{ uint32_t d_ = {s.read()}; if (!d_) {trap} uint32_t n_ = c->eax & 0xFFFF; "
                    f"set8l(&c->eax, n_ / d_); set8h(&c->eax, n_ % d_); }}"]
        return [f"{{ int32_t d_ = (int8_t){s.read()}; if (!d_) {trap} int32_t n_ = (int16_t)c->eax; "
                f"set8l(&c->eax, (uint32_t)(n_ / d_)); set8h(&c->eax, (uint32_t)(n_ % d_)); }}"]
    if m == "xadd":
        d, s = ops
        sz = d.size
        t = UT[sz]
        return [f"{{ {t} a_ = {d.read()}, b_ = {s.read()}, r_ = ({t})(a_ + b_); {s.write('a_')} {d.write('r_')} "
                f"setf(c, LF_ADD, {sz}, a_, b_, r_); }}"]
    if m == "cmpxchg":
        d, s = ops
        sz = d.size
        t = UT[sz]
        acc = Op({4: "eax", 2: "ax", 1: "al"}[sz])
        return [f"{{ {t} a_ = {acc.read()}, v_ = {d.read()}, r_ = ({t})(a_ - v_); setf(c, LF_SUB, {sz}, a_, v_, r_); "
                f"if (a_ == v_) {{ {d.write(s.read())} }} else {{ {acc.write('v_')} }} }}"]
    if m == "rdtsc":
        return ["c->eax = guest_rdtsc_lo(c, &c->edx);"]
    if m == "cpuid":
        return ["guest_cpuid(c);"]
    if m in ("in", "out"):
        if m == "in":
            d, p = ops
            port = p.read(2) if p.kind == "reg" else f"{p.value}"
            return [f"guest_port_io(c, 0x{a:x}, 0, {port}, {d.size});"]
        p, s = ops
        port = p.read(2) if p.kind == "reg" else f"{p.value}"
        return [f"guest_port_io(c, 0x{a:x}, 1, {port}, {s.size});"]
    if m == "leave":
        return ["c->esp = c->ebp; c->ebp = pop32(c);"]
    if m == "int3":
        return [f"guest_int3(c, 0x{a:x});"]

    if m in STRING_OPS:
        return string_op(ins, ops)
    if m.startswith("f"):
        return x87(ins, ops)
    return sse(ins, ops)


# ---------------------------------------------------------------- strings
STRING_OPS = {"movsb", "movsw", "movsd", "stosb", "stosw", "stosd", "lodsb", "lodsw", "lodsd",
              "cmpsb", "cmpsw", "cmpsd", "scasb", "scasw", "scasd"}


def string_op(ins, ops):
    m = ins.mnem
    if m in ("movsd", "cmpsd") and any("xmm" in o for o in ins.ops):
        return sse(ins, ops)
    sz = {"b": 1, "w": 2, "d": 4}[m[-1]]
    base = m[:-1]
    rd = {1: "rd8", 2: "rd16", 4: "rd32"}[sz]
    wr = {1: "wr8", 2: "wr16", 4: "wr32"}[sz]
    t = UT[sz]
    acc = {1: "(uint8_t)c->eax", 2: "(uint16_t)c->eax", 4: "c->eax"}[sz]
    accw = {1: "set8l(&c->eax, v_);", 2: "set16(&c->eax, v_);", 4: "c->eax = v_;"}[sz]
    st = f"str_step(c, {sz})"
    if base == "movs":
        body = f"{wr}(c->edi, {rd}(c->esi)); c->esi += {st}; c->edi += {st};"
    elif base == "stos":
        body = f"{wr}(c->edi, {acc}); c->edi += {st};"
    elif base == "lods":
        body = f"{{ {t} v_ = {rd}(c->esi); {accw} }} c->esi += {st};"
    elif base == "cmps":
        body = (f"{{ {t} a_ = {rd}(c->esi), b_ = {rd}(c->edi); setf(c, LF_SUB, {sz}, a_, b_, ({t})(a_ - b_)); }} "
                f"c->esi += {st}; c->edi += {st};")
    else:  # scas
        body = (f"{{ {t} a_ = {acc}, b_ = {rd}(c->edi); setf(c, LF_SUB, {sz}, a_, b_, ({t})(a_ - b_)); }} "
                f"c->edi += {st};")
    p = ins.prefix
    if not p:
        return [body]
    if base in ("movs", "stos", "lods"):
        if base == "stos" and sz in (1, 4):
            # fast path for memset-style fills
            fill = "(uint8_t)c->eax" if sz == 1 else "c->eax"
            if sz == 1:
                return [f"if (!c->df) {{ memset(g_mem + c->edi, {fill}, c->ecx); c->edi += c->ecx; c->ecx = 0; }} "
                        f"else while (c->ecx) {{ {body} c->ecx--; }}"]
        if base == "movs":
            return [f"if (!c->df && (c->edi >= c->esi + c->ecx*{sz}u || c->edi + c->ecx*{sz}u <= c->esi)) {{ "
                    f"memcpy(g_mem + c->edi, g_mem + c->esi, (size_t)c->ecx * {sz}); "
                    f"c->esi += c->ecx*{sz}u; c->edi += c->ecx*{sz}u; c->ecx = 0; }} "
                    f"else while (c->ecx) {{ {body} c->ecx--; }}"]
        return [f"while (c->ecx) {{ {body} c->ecx--; }}"]
    cond = "CC_E(c)" if p in ("rep", "repe", "repz") else "CC_NE(c)"
    return [f"while (c->ecx) {{ {body} c->ecx--; if (!{cond}) break; }}"]


# -------------------------------------------------------------------- x87
def fmem_load(o):
    if o.size == 4:
        return f"(double)rdf32({o.addr()})"
    if o.size == 8:
        return f"rdf64({o.addr()})"
    if o.size == 10:
        return f"rdf80({o.addr()})"
    raise Unsupported(f"fp load size {o.size}")


def fmem_iload(o):
    return {2: f"(double)(int16_t)rd16({o.addr()})", 4: f"(double)(int32_t)rd32({o.addr()})",
            8: f"(double)(int64_t)rd64({o.addr()})"}[o.size]


ARITH = {"add": "{a} + {b}", "mul": "{a} * {b}", "sub": "{a} - {b}", "subr": "{b} - {a}",
         "div": "{a} / {b}", "divr": "{b} / {a}"}


def x87_regform(ins):
    """Register-form arithmetic decoded from the opcode bytes (avoids the
    historical fsub/fsubr naming ambiguity of the st(i),st forms)."""
    raw = ins.raw
    i = 0
    while raw[i] in (0x66, 0x67, 0xF0, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65):
        i += 1
    op, modrm = raw[i], raw[i + 1]
    reg, sti = (modrm >> 3) & 7, modrm & 7
    names = ["add", "mul", "com", "comp", "sub", "subr", "div", "divr"]
    if op == 0xD8:
        n = names[reg]
        if n in ("com", "comp"):
            return [f"fcompare(c, ST(c,0), ST(c,{sti}));"] + (["fpop(c);"] if n == "comp" else [])
        return [f"ST(c,0) = {ARITH[n].format(a='ST(c,0)', b=f'ST(c,{sti})')};"]
    if op in (0xDC, 0xDE):
        # dest = st(i); /4: st(i) = st0 - st(i) ; /5: st(i) = st(i) - st0 ; /6: st0/st(i) ; /7: st(i)/st0
        n = {0: "{i} + {z}", 1: "{i} * {z}", 4: "{z} - {i}", 5: "{i} - {z}",
             6: "{z} / {i}", 7: "{i} / {z}"}.get(reg)
        if n is None:
            if op == 0xDE and reg == 3 and sti == 1:   # fcompp
                return ["fcompare(c, ST(c,0), ST(c,1)); fpop(c); fpop(c);"]
            if op == 0xDC and reg in (2, 3):
                return [f"fcompare(c, ST(c,0), ST(c,{sti}));"] + (["fpop(c);"] if reg == 3 else [])
            raise Unsupported(ins.mnem)
        L = [f"ST(c,{sti}) = {n.format(i=f'ST(c,{sti})', z='ST(c,0)')};"]
        if op == 0xDE:
            L.append("fpop(c);")
        return L
    raise Unsupported(ins.mnem)


def x87(ins, ops):
    m = ins.mnem
    if m in ("fld", "fild"):
        (s,) = ops
        if s.kind == "st":
            return [f"{{ double v_ = ST(c,{s.idx}); fpush(c, v_); }}"]
        return [f"fpush(c, {fmem_load(s) if m == 'fld' else fmem_iload(s)});"]
    consts = {"fld1": "1.0", "fldz": "0.0", "fldpi": "M_PI", "fldl2e": "M_LOG2E", "fldln2": "M_LN2",
              "fldlg2": "0.30102999566398119521", "fldl2t": "3.32192809488736234787"}
    if m in consts:
        return [f"fpush(c, {consts[m]});"]
    if m in ("fst", "fstp"):
        (d,) = ops
        pop = ["fpop(c);"] if m == "fstp" else []
        if d.kind == "st":
            return [f"ST(c,{d.idx}) = ST(c,0);"] + pop
        if d.size == 4:
            return [f"wrf32({d.addr()}, (float)ST(c,0));"] + pop
        if d.size == 8:
            return [f"wrf64({d.addr()}, ST(c,0));"] + pop
        if d.size == 10:
            return [f"wrf80({d.addr()}, ST(c,0));"] + pop
    if m in ("fist", "fistp", "fisttp"):
        (d,) = ops
        pop = ["fpop(c);"] if m != "fist" else []
        if m == "fisttp":
            cv = {2: "(int16_t)trunc(ST(c,0))", 4: "(int32_t)trunc(ST(c,0))", 8: "(int64_t)trunc(ST(c,0))"}[d.size]
        else:
            cv = {2: "fist16(c, ST(c,0))", 4: "fist32(c, ST(c,0))", 8: "fist64(c, ST(c,0))"}[d.size]
        w = {2: "wr16", 4: "wr32", 8: "wr64"}[d.size]
        return [f"{w}({d.addr()}, ({UT[d.size]}){cv});"] + pop
    base = m[1:]
    pop = base.endswith("p") and base not in ("fcomp",)
    for name in ("add", "mul", "subr", "sub", "divr", "div"):
        if base in (name, name + "p"):
            if len(ops) == 1 and ops[0].kind == "mem":
                return [f"ST(c,0) = {ARITH[name].format(a='ST(c,0)', b=fmem_load(ops[0]))};"]
            if not ops:  # faddp with no operands == faddp st(1), st
                return [f"ST(c,1) = {ARITH[name].format(a='ST(c,1)', b='ST(c,0)')};", "fpop(c);"]
            return x87_regform(ins)
        if base == "i" + name:
            return [f"ST(c,0) = {ARITH[name].format(a='ST(c,0)', b=fmem_iload(ops[0]))};"]
    if m in ("fcom", "fcomp", "fucom", "fucomp"):
        p = ["fpop(c);"] if m.endswith("p") else []
        if not ops:
            return ["fcompare(c, ST(c,0), ST(c,1));"] + p
        o = ops[0]
        b = f"ST(c,{o.idx})" if o.kind == "st" else fmem_load(o)
        return [f"fcompare(c, ST(c,0), {b});"] + p
    if m in ("fcompp", "fucompp"):
        return ["fcompare(c, ST(c,0), ST(c,1)); fpop(c); fpop(c);"]
    if m in ("ficom", "ficomp"):
        return [f"fcompare(c, ST(c,0), {fmem_iload(ops[0])});"] + (["fpop(c);"] if m == "ficomp" else [])
    if m in ("fcomi", "fcomip", "fucomi", "fucomip"):
        o = ops[-1]
        return [f"fcomi(c, ST(c,0), ST(c,{o.idx}));"] + (["fpop(c);"] if m.endswith("p") else [])
    if m == "ftst":
        return ["fcompare(c, ST(c,0), 0.0);"]
    if m == "fxam":
        return ["fxam(c);"]
    if m == "fxch":
        i = ops[0].idx if ops else 1
        return [f"{{ double t_ = ST(c,0); ST(c,0) = ST(c,{i}); ST(c,{i}) = t_; }}"]
    unary = {"fchs": "-ST(c,0)", "fabs": "fabs(ST(c,0))", "fsqrt": "sqrt(ST(c,0))",
             "fsin": "sin(ST(c,0))", "fcos": "cos(ST(c,0))", "frndint": "fround(c, ST(c,0))",
             "f2xm1": "exp2(ST(c,0)) - 1.0"}
    if m in unary:
        return [f"ST(c,0) = {unary[m]};"]
    if m == "fsincos":
        return ["{ double v_ = ST(c,0); ST(c,0) = sin(v_); fpush(c, cos(v_)); }"]
    if m == "fptan":
        return ["ST(c,0) = tan(ST(c,0)); fpush(c, 1.0);"]
    if m == "fpatan":
        return ["ST(c,1) = atan2(ST(c,1), ST(c,0)); fpop(c);"]
    if m == "fyl2x":
        return ["ST(c,1) = ST(c,1) * log2(ST(c,0)); fpop(c);"]
    if m == "fyl2xp1":
        return ["ST(c,1) = ST(c,1) * log2(ST(c,0) + 1.0); fpop(c);"]
    if m == "fscale":
        return ["ST(c,0) = ldexp(ST(c,0), (int)trunc(ST(c,1)));"]
    if m in ("fprem", "fprem1"):
        fn = "fmod" if m == "fprem" else "remainder"
        return [f"ST(c,0) = {fn}(ST(c,0), ST(c,1)); c->fsw &= ~0x0400;"]
    if m in ("fnstsw", "fstsw"):
        (d,) = ops
        return [d.write("fpu_sw(c)", 2)]
    if m in ("fnstcw", "fstcw"):
        return [ops[0].write("c->fcw", 2)]
    if m == "fldcw":
        return [f"c->fcw = {ops[0].read(2)};"]
    if m in ("fninit", "finit"):
        return ["c->fcw = 0x37F; c->fsw = 0; c->ftop = 0;"]
    if m in ("fnclex", "fclex"):
        return ["c->fsw &= 0x7F00;"]
    if m == "ffree":
        return []
    if m == "fincstp":
        return ["c->ftop = (c->ftop + 1) & 7;"]
    if m == "fdecstp":
        return ["c->ftop = (c->ftop - 1) & 7;"]
    if m.startswith("fcmov"):
        cc = {"fcmovb": "CC_B", "fcmove": "CC_E", "fcmovbe": "CC_BE", "fcmovu": "CC_P",
              "fcmovnb": "CC_AE", "fcmovne": "CC_NE", "fcmovnbe": "CC_A", "fcmovnu": "CC_NP"}[m]
        return [f"if ({cc}(c)) ST(c,0) = ST(c,{ops[1].idx});"]
    raise Unsupported(m)


# -------------------------------------------------------------------- SSE
def xr(o, lane="f", i=0):
    return f"c->xmm[{o.idx}].{lane}[{i}]"


MMX_BINOPS = {"paddb", "paddw", "paddd", "paddsb", "paddsw", "paddusb", "paddusw", "psubb", "psubw",
              "psubd", "psubsb", "psubsw", "psubusb", "psubusw", "pmullw", "pmulhw", "pmulhuw", "pmaddwd",
              "pcmpeqb", "pcmpeqw", "pcmpeqd", "pcmpgtb", "pcmpgtw", "pcmpgtd", "packuswb", "packsswb",
              "packssdw", "punpcklbw", "punpcklwd", "punpckldq", "punpckhbw", "punpckhwd", "punpckhdq",
              "pand", "pandn", "por", "pxor", "pavgb", "pavgw", "pmaxub", "pminub", "pmaxsw", "pminsw"}
MMX_SHIFTS = {"psllw", "pslld", "psllq", "psrlw", "psrld", "psrlq", "psraw", "psrad"}


def mm_src(o):
    if o.kind == "mm":
        return f"c->mm[{o.idx}]"
    if o.kind == "mem":
        return f"rd64({o.addr()})"
    raise Unsupported(f"mmx source {o.text}")


def mmx(ins, ops):
    m = ins.mnem
    if m == "emms":
        return []
    if m in MMX_BINOPS:
        d, s = ops
        return [f"c->mm[{d.idx}] = mmx_{m}(c->mm[{d.idx}], {mm_src(s)});"]
    if m in MMX_SHIFTS:
        d, s = ops
        n = f"{s.value & 0xFF}u" if s.kind == "imm" else mm_src(s)
        return [f"c->mm[{d.idx}] = mmx_{m}(c->mm[{d.idx}], (uint64_t)({n}));"]
    if m == "movntq":
        d, s = ops
        return [f"wr64({d.addr()}, c->mm[{s.idx}]);"]
    if m == "movq":
        d, s = ops
        if d.kind == "mm":
            return [f"c->mm[{d.idx}] = {mm_src(s)};"]
        if d.kind == "mem":
            return [f"wr64({d.addr()}, c->mm[{s.idx}]);"]
    if m == "movd":
        d, s = ops
        if d.kind == "mm":
            return [f"c->mm[{d.idx}] = (uint64_t)({s.read(4)});"]
        return [d.write(f"(uint32_t)c->mm[{s.idx}]", 4)]
    if m == "pshufw":
        d, s, imm = ops
        k = imm.value
        return [f"{{ uint64_t v_ = {mm_src(s)}, r_ = 0; for (int i_ = 0; i_ < 4; i_++) "
                f"r_ |= ((v_ >> (16 * ((0x{k:x} >> (2 * i_)) & 3))) & 0xFFFFull) << (16 * i_); c->mm[{d.idx}] = r_; }}"]
    if m == "pextrw":
        d, s, imm = ops
        return [d.write(f"(uint32_t)((c->mm[{s.idx}] >> {16 * (imm.value & 3)}) & 0xFFFF)", 4)]
    if m == "pinsrw":
        d, s, imm = ops
        sh = 16 * (imm.value & 3)
        return [f"c->mm[{d.idx}] = (c->mm[{d.idx}] & ~(0xFFFFull << {sh})) | ((uint64_t)({s.read(2) if s.kind == 'mem' else s.read(4)} & 0xFFFF) << {sh});"]
    if m == "pmovmskb":
        d, s = ops
        return [d.write(f"({{ uint32_t r_ = 0; for (int i_ = 0; i_ < 8; i_++) r_ |= ((c->mm[{s.idx}] >> (8 * i_ + 7)) & 1) << i_; r_; }})", 4)]
    raise Unsupported(m)


def sse(ins, ops):
    m = ins.mnem
    if m == "emms" or any(o.kind == "mm" for o in ops):
        return mmx(ins, ops)
    if m in ("movss",):
        d, s = ops
        if d.kind == "xmm" and s.kind == "mem":
            return [f"c->xmm[{d.idx}].u[0] = rd32({s.addr()}); c->xmm[{d.idx}].u[1] = "
                    f"c->xmm[{d.idx}].u[2] = c->xmm[{d.idx}].u[3] = 0;"]
        if d.kind == "mem":
            return [f"wr32({d.addr()}, {xr(s, 'u')});"]
        return [f"{xr(d, 'u')} = {xr(s, 'u')};"]
    if m in ("movaps", "movups", "movdqa", "movdqu", "movntps"):
        d, s = ops
        if d.kind == "xmm" and s.kind == "xmm":
            return [f"c->xmm[{d.idx}] = c->xmm[{s.idx}];"]
        if d.kind == "xmm":
            return [f"memcpy(&c->xmm[{d.idx}], g_mem + {s.addr()}, 16);"]
        return [f"memcpy(g_mem + {d.addr()}, &c->xmm[{s.idx}], 16);"]
    if m in ("movlps", "movhps"):
        d, s = ops
        q = 0 if m == "movlps" else 1
        if d.kind == "xmm":
            return [f"c->xmm[{d.idx}].q[{q}] = rd64({s.addr()});"]
        return [f"wr64({d.addr()}, c->xmm[{s.idx}].q[{q}]);"]
    if m == "movd":
        d, s = ops
        if d.kind == "xmm":
            return [f"c->xmm[{d.idx}].u[0] = {s.read(4)}; c->xmm[{d.idx}].u[1] = c->xmm[{d.idx}].u[2] = "
                    f"c->xmm[{d.idx}].u[3] = 0;"]
        return [d.write(xr(s, "u"), 4)]
    ps = {"addps": "+", "subps": "-", "mulps": "*", "divps": "/"}
    if m in ps:
        d, s = ops
        src = (f"c->xmm[{s.idx}]" if s.kind == "xmm" else "m_")
        pre = "" if s.kind == "xmm" else f"XMMReg m_; memcpy(&m_, g_mem + {s.addr()}, 16); "
        return [f"{{ {pre}for (int i_ = 0; i_ < 4; i_++) c->xmm[{d.idx}].f[i_] {ps[m]}= {src}.f[i_]; }}"]
    ss = {"addss": "+", "subss": "-", "mulss": "*", "divss": "/"}
    if m in ss:
        d, s = ops
        v = xr(s) if s.kind == "xmm" else f"rdf32({s.addr()})"
        return [f"{xr(d)} {ss[m]}= {v};"]
    if m in ("minss", "maxss", "sqrtss"):
        d, s = ops
        v = xr(s) if s.kind == "xmm" else f"rdf32({s.addr()})"
        e = {"minss": f"({xr(d)} < {v} ? {xr(d)} : {v})", "maxss": f"({xr(d)} > {v} ? {xr(d)} : {v})",
             "sqrtss": f"sqrtf({v})"}[m]
        return [f"{xr(d)} = {e};"]
    if m in ("xorps", "andps", "orps", "andnps"):
        d, s = ops
        op = {"xorps": "^", "andps": "&", "orps": "|", "andnps": "&"}[m]
        src = f"c->xmm[{s.idx}]" if s.kind == "xmm" else "m_"
        pre = "" if s.kind == "xmm" else f"XMMReg m_; memcpy(&m_, g_mem + {s.addr()}, 16); "
        dst = f"c->xmm[{d.idx}].u[i_]"
        lhs = f"~{dst}" if m == "andnps" else dst
        return [f"{{ {pre}for (int i_ = 0; i_ < 4; i_++) {dst} = {lhs} {op} {src}.u[i_]; }}"]
    if m == "shufps":
        d, s, imm = ops
        k = imm.value
        src = f"c->xmm[{s.idx}]" if s.kind == "xmm" else "m_"
        pre = "" if s.kind == "xmm" else f"XMMReg m_; memcpy(&m_, g_mem + {s.addr()}, 16); "
        return [f"{{ {pre}XMMReg t_; t_.u[0] = c->xmm[{d.idx}].u[{k & 3}]; t_.u[1] = c->xmm[{d.idx}].u[{(k >> 2) & 3}]; "
                f"t_.u[2] = {src}.u[{(k >> 4) & 3}]; t_.u[3] = {src}.u[{(k >> 6) & 3}]; c->xmm[{d.idx}] = t_; }}"]
    if m in ("comiss", "ucomiss"):
        d, s = ops
        v = xr(s) if s.kind == "xmm" else f"rdf32({s.addr()})"
        return [f"fcomi(c, {xr(d)}, {v});"]
    if m == "cvtsi2ss":
        d, s = ops
        return [f"{xr(d)} = (float)(int32_t){s.read(4)};"]
    if m in ("cvttss2si", "cvtss2si"):
        d, s = ops
        v = xr(s) if s.kind == "xmm" else f"rdf32({s.addr()})"
        fn = "truncf" if m == "cvttss2si" else "nearbyintf"
        return [d.write(f"(uint32_t)(int32_t){fn}({v})")]
    if m in ("unpcklps", "unpckhps"):
        d, s = ops
        h = 2 if m == "unpckhps" else 0
        return [f"{{ XMMReg a_ = c->xmm[{d.idx}], b_ = c->xmm[{s.idx}]; c->xmm[{d.idx}].u[0] = a_.u[{h}]; "
                f"c->xmm[{d.idx}].u[1] = b_.u[{h}]; c->xmm[{d.idx}].u[2] = a_.u[{h+1}]; c->xmm[{d.idx}].u[3] = b_.u[{h+1}]; }}"]
    if m in ("ldmxcsr", "stmxcsr"):
        o = ops[0]
        return [f"c->mxcsr = {o.read(4)};"] if m == "ldmxcsr" else [o.write("c->mxcsr", 4)]
    raise Unsupported(m)


ROT_HELPERS = r"""
static inline uint32_t rot_rol(CPU *c, uint32_t v, uint32_t n, int sz) {
    int bits = sz * 8; uint32_t m = sz_mask(sz); v &= m; n %= bits;
    uint32_t r = n ? ((v << n) | (v >> (bits - n))) & m : v;
    uint32_t f = flags_get(c) & ~(FL_CF | FL_OF);
    if (r & 1) f |= FL_CF;
    if (((r >> (bits - 1)) ^ r) & 1) f |= FL_OF;
    flags_set(c, f); return r;
}
static inline uint32_t rot_ror(CPU *c, uint32_t v, uint32_t n, int sz) {
    int bits = sz * 8; uint32_t m = sz_mask(sz); v &= m; n %= bits;
    uint32_t r = n ? ((v >> n) | (v << (bits - n))) & m : v;
    uint32_t f = flags_get(c) & ~(FL_CF | FL_OF);
    if ((r >> (bits - 1)) & 1) f |= FL_CF;
    if (((r >> (bits - 1)) ^ (r >> (bits - 2))) & 1) f |= FL_OF;
    flags_set(c, f); return r;
}
static inline uint32_t rot_rcl(CPU *c, uint32_t v, uint32_t n, int sz) {
    int bits = sz * 8; uint32_t m = sz_mask(sz); uint32_t cf = f_cf(c) ? 1 : 0; v &= m;
    n %= (bits + 1);
    for (uint32_t i = 0; i < n; i++) { uint32_t out = (v >> (bits - 1)) & 1; v = ((v << 1) | cf) & m; cf = out; }
    uint32_t f = flags_get(c) & ~(FL_CF | FL_OF); if (cf) f |= FL_CF;
    if ((((v >> (bits - 1)) & 1) ^ cf)) f |= FL_OF; flags_set(c, f); return v;
}
static inline uint32_t rot_rcr(CPU *c, uint32_t v, uint32_t n, int sz) {
    int bits = sz * 8; uint32_t m = sz_mask(sz); uint32_t cf = f_cf(c) ? 1 : 0; v &= m;
    n %= (bits + 1);
    for (uint32_t i = 0; i < n; i++) { uint32_t out = v & 1; v = (v >> 1) | (cf << (bits - 1)); cf = out; }
    uint32_t f = flags_get(c) & ~(FL_CF | FL_OF); if (cf) f |= FL_CF;
    if (((v >> (bits - 1)) ^ (v >> (bits - 2))) & 1) f |= FL_OF; flags_set(c, f); return v;
}
"""
