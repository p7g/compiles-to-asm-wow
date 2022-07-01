# From

```ts
declare function puts(str: *u8);
declare function strlen(str: *u8): u32;

function inner(): i32 {
    puts("Testing");
    return strlen("wowowowow");
}

function main(argc: i32, argv: **u8): i32 {
    puts("Hello, world!");
    var code = inner();
    return code;
}
```

# To

```asm
	.section	__TEXT,__text

	.globl	_inner
_inner:
	pushq	%rbp
	movq	%rsp, %rbp
	leaq	L1(%rip), %rdi
	call	_puts
	leaq	L2(%rip), %rdi
	call	_strlen
L0:
	popq	%rbp
	ret

	.globl	_main
_main:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$32, %rsp
	movl	%edi, -4(%rbp)
	movq	%rsi, -16(%rbp)
	leaq	L4(%rip), %rdi
	call	_puts
	call	_inner
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
L3:
	addq	$32, %rsp
	popq	%rbp
	ret

	.section	__TEXT,__cstring
L1:	.asciz "Testing"
L2:	.asciz "wowowowow"
L4:	.asciz "Hello, world!"
```

# Notes

Use like `python -mx mycoolfile.x -o wheretheassemblygoes.s`

Mach-O is hard so it just compiles to an assembly text file, you can compile the code with `clang wheretheassemblygoes.s` and then run like `./a.out`
