# From

```ts
extern function fputs(string: *u8, file: *u64);
extern var __stdoutp: *u64;
extern var __stderrp: *u64;

function main(): i32 {
    fputs("Hello", __stdoutp);
    fputs("errororor", __stderrp);

    return 0;
}
```

# To

```asm
	.section	__TEXT,__text

	.globl	_main
_main:
	pushq	%rbp
	movq	%rsp, %rbp
	## end prologue
	## function body
	leaq	L1(%rip), %rdi
	movq	___stdoutp@GOTPCREL(%rip), %rax
	movq	0(%rax), %rsi	## __stdoutp
	call	_fputs
	leaq	L2(%rip), %rdi
	movq	___stderrp@GOTPCREL(%rip), %rax
	movq	0(%rax), %rsi	## __stderrp
	call	_fputs
	xorl	%eax, %eax
L0:
	## epilogue
	popq	%rbp
	ret

	.section	__TEXT,__cstring
L1:	.asciz "Hello"
L2:	.asciz "errororor"
```

# Notes

Use like `python -mx mycoolfile.x -o wheretheassemblygoes.s`

Mach-O is hard so it just compiles to an assembly text file, you can compile the code with `clang wheretheassemblygoes.s` and then run like `./a.out`
