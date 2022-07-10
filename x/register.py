class Register:
    def __init__(self, variants):
        self.variants = variants

    def name(self, size):
        return b"%%%s" % self.variants[size]


a = Register(
    {
        1: b"al",
        2: b"ax",
        4: b"eax",
        8: b"rax",
    }
)
b = Register(
    {
        1: b"bl",
        2: b"bx",
        4: b"ebx",
        8: b"rbx",
    }
)
c = Register(
    {
        1: b"cl",
        2: b"cx",
        4: b"ecx",
        8: b"rcx",
    }
)
d = Register(
    {
        1: b"dl",
        2: b"dx",
        4: b"edx",
        8: b"rdx",
    }
)
ip = Register(
    {
        1: b"ipl",
        2: b"ip",
        4: b"eip",
        8: b"rip",
    }
)
bp = Register(
    {
        1: b"bpl",
        2: b"bp",
        4: b"ebp",
        8: b"rbp",
    }
)
sp = Register(
    {
        1: b"spl",
        2: b"sp",
        4: b"esp",
        8: b"rsp",
    }
)
si = Register(
    {
        1: b"sil",
        2: b"si",
        4: b"esi",
        8: b"rsi",
    }
)
di = Register(
    {
        1: b"dil",
        2: b"di",
        4: b"edi",
        8: b"rdi",
    }
)
r8 = Register(
    {
        1: b"r8b",
        2: b"r8w",
        4: b"r8d",
        8: b"r8",
    }
)
r9 = Register(
    {
        1: b"r9b",
        2: b"r9w",
        4: b"r9d",
        8: b"r9",
    }
)
r10 = Register(
    {
        1: b"r10b",
        2: b"r10w",
        4: b"r10d",
        8: b"r10",
    }
)
r11 = Register(
    {
        1: b"r11b",
        2: b"r11w",
        4: b"r11d",
        8: b"r11",
    }
)
r12 = Register(
    {
        1: b"r12b",
        2: b"r12w",
        4: b"r12d",
        8: b"r12",
    }
)
r13 = Register(
    {
        1: b"r13b",
        2: b"r13w",
        4: b"r13d",
        8: b"r13",
    }
)
r14 = Register(
    {
        1: b"r14b",
        2: b"r14w",
        4: b"r14d",
        8: b"r14",
    }
)
r15 = Register(
    {
        1: b"r15b",
        2: b"r15w",
        4: b"r15d",
        8: b"r15",
    }
)

argument_registers = (di, si, d, c, r8, r9)


class Immediate:
    def __init__(self, value):
        self.value = value

    def __bytes__(self):
        return b"$%s" % str(self.value).encode("ascii")


class Address:
    def __init__(self, reg, size, offset=None, index_reg=None, scale_factor=None):
        self.reg = reg
        self.size = size
        self.offset = offset
        self.index_reg = index_reg
        self.scale_factor = scale_factor

    def __eq__(self, other):
        if not isinstance(other, Address):
            return NotImplemented
        return (
            self.reg == other.reg
            and self.size == other.size
            and self.offset == other.offset
            and self.index_reg == other.index_reg
            and self.scale_factor == other.scale_factor
        )

    def __bytes__(self):
        if not self.is_plain_reg():
            addr = self.reg.name(8)
            if self.index_reg:
                addr += b",%s" % self.index_reg.name(8)
            if self.scale_factor:
                addr += b",%s" % str(self.scale_factor).encode("ascii")
            if isinstance(self.offset, int):
                offset_bytes = str(self.offset).encode("ascii")
            elif self.offset:
                offset_bytes = bytes(self.offset)
            else:
                offset_bytes = b""
            return b"%s(%s)" % (offset_bytes, addr)
        else:
            return self.reg.name(self.size)

    def with_size(self, size):
        return Address(self.reg, size, self.offset, self.index_reg, self.scale_factor)

    def with_offset(self, offset):
        return Address(self.reg, self.size, offset, self.index_reg, self.scale_factor)

    def is_plain_reg(self):
        return self.offset is None and self.index_reg is None
