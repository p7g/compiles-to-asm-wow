"""Crappy GNU assembler peephole optimizer"""

import re


def peephole_opt(asm):
    asm = _remove_jump_to_next_instruction(asm)
    asm = _optimize_mov_0(asm)
    asm = _remove_add_sub_0(asm)
    return asm


def _remove_jump_to_next_instruction(asm):
    lines = asm.splitlines()
    new_lines = []

    i = 0
    while i < len(lines):
        if i + 1 == len(lines):
            new_lines.append(lines[i])
            break
        line_a, line_b = lines[i : i + 2]
        if not line_a.startswith(b"	jmp	") or line_a.removeprefix(
            b"	jmp	"
        ) != line_b.removesuffix(b":"):
            new_lines.append(line_a)
        i += 1

    return b"\n".join(new_lines)


_mov_0_re = re.compile(rb"^\tmov([bwlq])\t\$0, (%-?\w+(?:\(\w+\))?)$")


def _optimize_mov_0(asm):
    new_lines = []

    for line in asm.splitlines():
        match = _mov_0_re.fullmatch(line)
        if match:
            size_suffix, register = match.groups()
            new_lines.append(b"	xor%s	%s, %s" % (size_suffix, register, register))
        else:
            new_lines.append(line)

    return b"\n".join(new_lines)


_add_sub_0_re = re.compile(rb"^\t(add|sub)[bwlq]\t\$0,")


def _remove_add_sub_0(asm):
    new_lines = []

    for line in asm.splitlines():
        if not _add_sub_0_re.match(line):
            new_lines.append(line)

    return b"\n".join(new_lines)


if __name__ == "__main__":
    # test it

    # _remove_jump_to_next_instruction
    assert _remove_jump_to_next_instruction(b"\tjmp\tL2\nL2:") == b"L2:"
    assert _remove_jump_to_next_instruction(b"\tjmp\tL2\nL3:") == b"\tjmp\tL2\nL3:"
    assert (
        _remove_jump_to_next_instruction(b"\tjmp\tL2\nhello\nL2:")
        == b"\tjmp\tL2\nhello\nL2:"
    )

    # _optimize_mov_0
    assert _optimize_mov_0(b"\tmovl\t$0, %eax") == b"\txorl\t%eax, %eax"
    assert _optimize_mov_0(b"\tmovl\t$123, %eax") == b"\tmovl\t$123, %eax"
    assert _optimize_mov_0(b"\tmovq\t$0, %rax") == b"\txorq\t%rax, %rax"

    # _remove_add_sub_0
    assert _remove_add_sub_0(b"\taddq\t$0, %rax") == b""
    assert _remove_add_sub_0(b"\taddl\t$0, %eax") == b""
    assert _remove_add_sub_0(b"\taddq\t$1, %rax") == b"\taddq\t$1, %rax"
