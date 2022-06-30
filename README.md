# From

```ts
declare function puts(str: *u8);

function inner(): i32 {
    puts("Testing");
    return 0;
}

function main(argc: i32, argv: **u8): i32 {
    puts("Hello, world!");
    return inner();
}
```

# To

```asm
	.section	__TEXT,__text

	.globl	_inner
_inner:
	leaq	L1(%rip), %rdi
	call	_puts
	xorl	%eax, %eax
	ret

	.globl	_main
_main:
	leaq	L3(%rip), %rdi
	call	_puts
	call	_inner
	ret

	.section	__TEXT,__cstring
L1:
	.asciz "Testing"
L3:
	.asciz "Hello, world!"
```

# Notes

Use like `python -m x mycoolfile.x -o wheretheassemblygoes.s`

Mach-O is hard so it just compiles to an assembly text file, you can compile the code with `clang wheretheassemblygoes.s` and then run like `./a.out`
