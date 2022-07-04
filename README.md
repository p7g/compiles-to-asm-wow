# From

```ts
extern function printf(fmt: *u8, intval: i32);

function main(): i32 {
    var i = 0;

    while !(i == 10) {
        printf("%d", i);
        i = i + 1;
    }

    return i;
}
```

# To

```asm
	.section	__TEXT,__text

	.globl	_main
_main:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$16, %rsp
	## end prologue
	## function body
	movl	$0, -4(%rbp)
	jmp	L2
L1:
	leaq	L3(%rip), %rdi
	movl	-4(%rbp), %esi	## i
	call	_printf
	movl	-4(%rbp), %eax	## i
	movl	$1, -8(%rbp)
	addl	-8(%rbp), %eax
	movl	%eax, -4(%rbp)
L2:
	movl	-4(%rbp), %eax
	movl	%eax, -8(%rbp)	## i
	movl	$10, %eax
	cmpl	%eax, -8(%rbp)
	jne	L1
	movl	-4(%rbp), %eax	## i
L0:
	## epilogue
	addq	$16, %rsp
	popq	%rbp
	ret

	.section	__TEXT,__cstring
L3:	.asciz "%d"
```

# Notes

Use like `python -mx mycoolfile.x -o wheretheassemblygoes.s`

Mach-O is hard so it just compiles to an assembly text file, you can compile the code with `clang wheretheassemblygoes.s` and then run like `./a.out`
