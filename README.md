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
	leaq	L2(%rip), %rdi
	call	_strlen
	ret

	.globl	_main
_main:
	leaq	L4(%rip), %rdi
	call	_puts
	call	_inner
	ret

	.section	__TEXT,__cstring
L1:
	.asciz "Testing"
L2:
	.asciz "wowowowow"
L4:
	.asciz "Hello, world!"
```

# Notes

Use like `python -mx mycoolfile.x -o wheretheassemblygoes.s`

Mach-O is hard so it just compiles to an assembly text file, you can compile the code with `clang wheretheassemblygoes.s` and then run like `./a.out`
