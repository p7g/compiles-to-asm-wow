Example of a program like `cat(1)`

```ts
extern type FILE;

extern function fopen(name: *u8, mode: *u8): *FILE;
extern function fclose(fp: *FILE);
extern function fgetc(fp: *FILE): i32;
extern function fputc(c: i32, fp: *FILE);
extern function fprintf(fp: *FILE, fmt: *u8, prog: *u8);
extern function perror(msg: *u8);
extern var __stdoutp: *FILE;
extern var __stderrp: *FILE;

function main(argc: i32, argv: **u8): i32 {
    if argc == 1 {
        fprintf(__stderrp, "usage: %s FILE...\n", argv[0]);
        return 1;
    }

    var i = 1;
    while i != argc {
        var f = fopen(argv[i], "r");
        i += 1;

        if !f {
            perror("fopen");
            return 1;
        }

        var c: i32;
        while (c = fgetc(f)) != -1 {
            fputc(c, __stdoutp);
        }

        fclose(f);
    }

    return 0;
}
```

<details>
<summary>Assembly output</summary>

```asm
	.section	__TEXT,__text

	.globl	_main
_main:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$48, %rsp
	## end prologue
	## Move all the arguments to the stack
	movl	%edi, -4(%rbp)	## argc
	movq	%rsi, -16(%rbp)	## argv
	## function body
	## if
	movl	-4(%rbp), %eax
	movl	%eax, -20(%rbp)	## argc
	movl	$1, %eax
	cmpl	%eax, -20(%rbp)
	jne	L2
L1:	## then
	movq	-16(%rbp), %rax	## argv
	xorl	%ecx, %ecx
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rdx
	movq	___stderrp@GOTPCREL(%rip), %rax
	movq	0(%rax), %rdi	## __stderrp
	leaq	L4(%rip), %rsi
	call	_fprintf
	movl	$1, %eax
	jmp	L0
	jmp	L3
L2:	## else
L3:	## end if
	movl	$1, -24(%rbp)
	jmp	L6
L5:
	movq	-16(%rbp), %rax	## argv
	movl	-24(%rbp), %ecx	## i
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rdi
	leaq	L7(%rip), %rsi
	call	_fopen
	movq	%rax, -32(%rbp)
	movl	$1, %edx
	addl	%edx, -24(%rbp)
	## if
	movq	-32(%rbp), %rax	## f
	cmpq	$0, %rax
	jne	L9
L8:	## then
	leaq	L11(%rip), %rdi
	call	_perror
	movl	$1, %eax
	jmp	L0
	jmp	L10
L9:	## else
L10:	## end if
	jmp	L13
L12:
	movl	-36(%rbp), %edi	## c
	movq	___stdoutp@GOTPCREL(%rip), %rax
	movq	0(%rax), %rsi	## __stdoutp
	call	_fputc
L13:
	movq	-32(%rbp), %rdi	## f
	call	_fgetc
	movl	%eax, %edx
	movl	%edx, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	$-1, %eax
	cmpl	%eax, -20(%rbp)
	jne	L12
	movq	-32(%rbp), %rdi	## f
	call	_fclose
L6:
	movl	-24(%rbp), %eax
	movl	%eax, -20(%rbp)	## i
	movl	-4(%rbp), %eax	## argc
	cmpl	%eax, -20(%rbp)
	jne	L5
	xorl	%eax, %eax
L0:
	## epilogue
	addq	$48, %rsp
	popq	%rbp
	ret

	.section	__TEXT,__cstring
L4:	.asciz "usage: %s FILE...
"
L7:	.asciz "r"
L11:	.asciz "fopen"
```

</details>

# Try it out

Use like `python -mx mycoolfile.x | clang -x assembler`
