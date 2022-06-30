from collections import namedtuple
from x import parse, register
from x.instr import mov, movs


# Type system


class XTypeError(Exception):
    pass


class Type:
    pass


class IntType(Type):
    def __init__(self, size, signed, from_literal=False):
        super().__init__()
        self.size = size
        self.signed = signed
        self.from_literal = from_literal


class PointerType(Type):
    def __init__(self, pointee):
        super().__init__()
        self.pointee = pointee
        self.size = 8


class FunctionType(Type):
    def __init__(self, params, ret):
        super().__init__()
        self.params = params
        self.ret = ret


def analyze_type(ctx, expr):
    if isinstance(expr, parse.IdentExpr):
        # TODO local variables, or really any variables
        # TODO abstract over name resolution
        return ctx.named_functions[expr.name]
    elif isinstance(expr, parse.IntExpr):
        return IntType(4, signed=True, from_literal=True)
    elif isinstance(expr, parse.StringExpr):
        return PointerType(IntType(1, signed=False))
    elif isinstance(expr, parse.CallExpr):
        target_ty = analyze_type(ctx, expr.target)
        if not isinstance(target_ty, FunctionType):
            raise XTypeError(f"Cannot call expression of type {target_ty}")
        if len(expr.args) != len(target_ty.params):
            raise XTypeError(f"Incorrect number of arguments passed to {expr.target}")
        for param, arg in zip(target_ty.params, expr.args):
            if not is_assignable(analyze_type(ctx, arg), param):
                raise XTypeError(f"Cannot pass {arg} as {param}")
        return target_ty.ret
    else:
        raise NotImplementedError(expr)


def is_assignable(from_ty, to_ty):
    if isinstance(from_ty, IntType) and isinstance(to_ty, IntType):
        if from_ty.size <= to_ty.size:
            return True
        elif from_ty.from_literal:
            return True  # TODO: Maybe warn if number will be truncated?
        else:
            return False  # TODO: Explain that explicit cast is needed
    elif isinstance(from_ty, PointerType) and isinstance(to_ty, PointerType):
        return is_same_type(from_ty.pointee, to_ty.pointee)
    else:
        return False


def is_same_type(a, b):
    if isinstance(a, IntType) and isinstance(b, IntType):
        return a.size == b.size
    elif isinstance(a, PointerType) and isinstance(b, PointerType):
        return is_same_type(a.pointee, b.pointee)
    elif isinstance(a, FunctionType) and isinstance(b, FunctionType):
        return (
            len(a.params) == len(b.params)
            and all(
                is_same_type(param_a, param_b)
                for param_a, param_b in zip(a.params, b.params)
            )
            and is_same_type(a.ret, b.ret)
        )
    else:
        raise NotImplementedError((a, b))


# Code generation


def global_name(name):
    return b"_%s" % name


class ProgramContext:
    def __init__(self):
        self.declared_functions = {}
        self.strings = []
        self.named_types = {
            "i8": IntType(1, signed=True),
            "u8": IntType(1, signed=False),
            "i16": IntType(2, signed=True),
            "u16": IntType(2, signed=False),
            "i32": IntType(4, signed=True),
            "u32": IntType(4, signed=False),
            "i64": IntType(8, signed=True),
            "u64": IntType(8, signed=False),
        }
        self.asm = b""

    def emitln(self, line):
        self.asm += b"%s\n" % line


class Label:
    next_idx = 0

    def __init__(self, index):
        self.idx = index

    @classmethod
    def create(cls):
        label = cls(cls.next_idx)
        cls.next_idx += 1
        return label

    def __bytes__(self):
        return b"L%s" % str(self.idx).encode("ascii")


FuncMeta = namedtuple("FuncMeta", "name type end_label")


def xcompile(decls):
    ctx = ProgramContext()

    ctx.emitln(b"	.section	__TEXT,__text")
    ctx.emitln(b"")

    for decl in decls:
        assert isinstance(decl, parse.FuncDecl)

        # Create type for function
        params = [compile_type(ctx, param.type) for param in decl.params]
        ret = compile_type(ctx, decl.ret) if decl.ret else None
        func_ty = FunctionType(params, ret)
        end_label = None if decl.is_proto else Label.create()
        func_meta = FuncMeta(decl.name.encode("ascii"), func_ty, end_label)

        ctx.declared_functions[decl.name] = func_meta

        # Just add the type
        if decl.is_proto:
            continue

        symbol_name = global_name(func_meta.name)
        ctx.emitln(b"	.globl	%s" % symbol_name)
        ctx.emitln(b"%s:" % symbol_name)

        # TODO: local variables

        for stmt in decl.body:
            compile_statement(ctx, func_meta, stmt)

        # TODO: function epilogue
        ctx.emitln(b"%s:" % end_label)
        ctx.emitln(b"	ret")

    if ctx.strings:
        ctx.emitln(b"")
        ctx.emitln(b"	.section	__TEXT,__cstring")

    for label, string in ctx.strings:
        ctx.emitln(b"%s:" % label)
        ctx.emitln(b'	.asciz "%s"' % string.encode("utf-8"))

    return ctx.asm


def compile_type(ctx, ast_ty):
    if isinstance(ast_ty, parse.NamedTypeExpr):
        try:
            return ctx.named_types[ast_ty.name]
        except KeyError:
            raise XTypeError(f"Unknown type {ast_ty.name}")
    elif isinstance(ast_ty, parse.PointerTypeExpr):
        return PointerType(compile_type(ctx, ast_ty.pointee))
    else:
        raise NotImplementedError(repr(ast_ty))


def compile_statement(ctx, func_meta, stmt):
    if isinstance(stmt, parse.ReturnStmt):
        if stmt.expr:
            expr_ty = analyze_type(ctx, stmt.expr)
            if not is_assignable(expr_ty, func_meta.type.ret):
                raise XTypeError(f"Cannot return {expr_ty!r} as {func_meta.type.ret!r}")

            dest = register.Address(register.a, size=expr_ty.size)
            compile_expr(ctx, stmt.expr, dest)
            implicitly_cast(ctx, dest, expr_ty, func_meta.type.ret)

        ctx.emitln(b"	jmp	%s" % func_meta.end_label)
    elif isinstance(stmt, parse.ExprStmt):
        compile_expr(ctx, stmt.expr, None)
    else:
        raise NotImplementedError(stmt)


def compile_expr(ctx, expr, dest):
    if isinstance(expr, parse.IntExpr):
        if dest is None:
            return
        ctx.emitln(
            b"	%s	%s, %s"
            % (
                mov(dest.size),
                register.Immediate(expr.value),
                dest,
            )
        )
    elif isinstance(expr, parse.StringExpr):
        if dest is None:
            return
        label = Label.create()
        ctx.strings.append((label, expr.text))
        ctx.emitln(
            b"	leaq	%s, %s"
            % (
                register.Address(register.ip, size=8, offset=label),
                dest,
            )
        )
    elif isinstance(expr, parse.IdentExpr):
        # TODO: Variables and stuff
        raise NotImplementedError("variables")
    elif isinstance(expr, parse.CallExpr):
        # TODO: Function pointers
        if not isinstance(expr.target, parse.IdentExpr):
            raise NotImplementedError

        func = ctx.declared_functions[expr.target.name]
        assert len(func.type.params) <= len(register.argument_registers)
        assert len(func.type.params) == len(expr.args)

        for dest_reg, param, arg in zip(
            register.argument_registers, func.type.params, expr.args
        ):
            # FIXME: Avoid clobbering the arguments passed to the caller;
            # perhaps put them on the stack until after this function is called?
            compile_expr(
                ctx,
                arg,
                register.Address(
                    dest_reg,
                    size=param.size,
                ),
            )

        ctx.emitln(b"	call	%s" % global_name(func.name))
        if dest is not None:
            ctx.emitln(
                b"	%s	%s, %s"
                % (
                    mov(func.type.ret.size),
                    register.Address(register.a, size=func.type.ret.size),
                    dest,
                )
            )


def implicitly_cast(ctx, addr, from_ty, to_ty):
    if not isinstance(from_ty, IntType) or not isinstance(to_ty, IntType):
        raise NotImplementedError

    if from_ty.size == to_ty.size:
        return

    assert from_ty.size < to_ty.size
    if from_ty.signed:
        mov_instr = movs(from_ty.size, to_ty.size)
    else:
        mov_instr = mov(to_ty.size)

    ctx.emitln(
        b"	%s	%s, %s"
        % (
            mov_instr,
            addr,
            addr,
        )
    )
