"""Crappy GNU assembler peephole optimizer"""

import re


def peephole_opt(asm):
    asm = _remove_jump_to_next_instruction(asm)
    asm = _optimize_mov_0(asm)
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
        if line_a.startswith(b"	jmp	") and line_a.removeprefix(
            b"	jmp	"
        ) == line_b.removesuffix(b":"):
            i += 2
        else:
            i += 1
            new_lines.append(line_a)

    return b"\n".join(new_lines)


_mov_0_re = re.compile(rb"^\tmov([bwlq])\t\$0, (%-?\w+(?:\(\w+\))?)$")


def _optimize_mov_0(asm):
    new_lines = []

    # movl $0, %eax
    for line in asm.splitlines():
        match = _mov_0_re.fullmatch(line)
        if match:
            size_suffix, register = match.groups()
            new_lines.append(b"	xor%s	%s, %s" % (size_suffix, register, register))
        else:
            new_lines.append(line)

    return b"\n".join(new_lines)


if __name__ == "__main__":
    # test it

    # _remove_jump_to_next_instruction
    assert _remove_jump_to_next_instruction(b"\tjmp\tL2\nL2:") == b""
    assert _remove_jump_to_next_instruction(b"\tjmp\tL2\nL3:") == b"\tjmp\tL2\nL3:"
    assert (
        _remove_jump_to_next_instruction(b"\tjmp\tL2\nhello\nL2:")
        == b"\tjmp\tL2\nhello\nL2:"
    )

    # _optimize_mov_0
    assert _optimize_mov_0(b"\tmovl\t$0, %eax") == b"\txorl\t%eax, %eax"
    assert _optimize_mov_0(b"\tmovl\t$123, %eax") == b"\tmovl\t$123, %eax"
    assert _optimize_mov_0(b"\tmovq\t$0, %rax") == b"\txorq\t%rax, %rax"
