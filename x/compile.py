from collections import ChainMap
from x import opt, parse, register
from x.instr import cmp, mov, movs


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

    def __str__(self):
        bit_size = self.size << 3
        return f"i{bit_size}" if self.signed else f"u{bit_size}"


class BoolType(Type):
    size = 1

    def __str__(self):
        return "bool"


class PointerType(Type):
    def __init__(self, pointee):
        super().__init__()
        self.pointee = pointee
        self.size = 8

    def __str__(self):
        return f"*{self.pointee}"


class FunctionType(Type):
    def __init__(self, params, ret):
        super().__init__()
        self.params = params
        self.ret = ret

    def __str__(self):
        params = ", ".join(map(str, self.params))
        return f"function({params}): {self.ret}"


def analyze_type(ctx, expr):
    if isinstance(expr, parse.IdentExpr):
        # TODO abstract over name resolution
        try:
            return ctx.func.resolve_variable(expr.name).type
        except XTypeError:
            pass
        try:
            return ctx.declared_functions[expr.name].type
        except KeyError:
            pass
        raise XTypeError(f"No variable named {expr.name!r} is in scope")
    elif isinstance(expr, parse.IntExpr):
        return IntType(4, signed=True, from_literal=True)
    elif isinstance(expr, parse.BoolExpr):
        return BoolType()
    elif isinstance(expr, parse.StringExpr):
        return PointerType(IntType(1, signed=False))
    elif isinstance(expr, parse.CallExpr):
        target_ty = analyze_type(ctx, expr.target)
        if not isinstance(target_ty, FunctionType):
            raise XTypeError(f"Cannot call expression of type {target_ty}")
        if len(expr.args) != len(target_ty.params):
            # FIXME: assumes target is a NameExpr
            raise XTypeError(
                f"Incorrect number of arguments passed to {expr.target.name!r}"
            )
        for param, arg in zip(target_ty.params, expr.args):
            arg_ty = analyze_type(ctx, arg)
            if not is_assignable(arg_ty, param):
                raise XTypeError(f"Cannot pass {arg_ty} as {param}")
        return target_ty.ret
    elif isinstance(expr, parse.UnaryExpr):
        if expr.op is not parse.UnaryOp.LOGICAL_NOT:
            raise NotImplementedError(expr.op)
        return BoolType()
    elif isinstance(expr, parse.BinaryExpr):
        if expr.op in (parse.BinaryOp.LOGICAL_AND, parse.BinaryOp.LOGICAL_OR):
            return BoolType()
        raise NotImplementedError(expr.op)
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
    # elif isinstance(from_ty, BoolType) and isinstance(to_ty, IntType):
    #     return True
    else:
        return is_same_type(from_ty, to_ty)


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
    elif isinstance(a, BoolType) and isinstance(b, BoolType):
        return True
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
            "bool": BoolType(),
            "i8": IntType(1, signed=True),
            "u8": IntType(1, signed=False),
            "i16": IntType(2, signed=True),
            "u16": IntType(2, signed=False),
            "i32": IntType(4, signed=True),
            "u32": IntType(4, signed=False),
            "i64": IntType(8, signed=True),
            "u64": IntType(8, signed=False),
        }
        self._asm_lines = []
        self.func = None

    def emitln(self, line: bytes):
        assert isinstance(line, bytes)
        self._asm_lines.append(line)

    def insertln(self, pos, line: bytes):
        assert isinstance(line, bytes)
        self._asm_lines.insert(pos, line)

    def next_lineno(self):
        return len(self._asm_lines)

    def comment(self, message):
        self._asm_lines[-1] += b"	/* " + message + b" */"

    def commentln(self, message):
        self._asm_lines.append(b"	/* " + message + b" */")

    @property
    def asm(self):
        return b"\n".join(self._asm_lines)


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


class Variable:
    def __init__(self, name, type_, stack_offset):
        self.name = name
        self.type = type_
        self.stack_offset = stack_offset

    @property
    def addr(self):
        return register.Address(
            register.bp,
            size=self.type.size,
            offset=str(-self.stack_offset).encode("ascii"),
        )


def align(addr, alignment):
    import math

    return math.ceil(addr / alignment) * alignment


assert align(0, 4) == 0, align(0, 4)
assert align(0, 8) == 0, align(0, 8)
assert align(4, 8) == 8, align(4, 8)
assert align(8, 8) == 8, align(8, 8)
assert align(16, 4) == 16, align(16, 4)
assert align(32, 4) == 32, align(32, 4)
assert align(32, 8) == 32, align(32, 8)


class FuncMeta:
    def __init__(self, name, type_, end_label):
        self.name = name
        self.type = type_
        self.end_label = end_label
        self._scope = ChainMap()
        self._next_stack_offset = 0

    def enter_scope(self):
        self._scope = self._scope.new_child()

    def exit_scope(self):
        self._scope = self._scope.parents

    def declare_variable(self, name, type_):
        if name in self._scope.maps[0]:
            raise XTypeError(f"Cannot redeclare variable {name!r}")
        align_to = 4 if type_.size <= 4 else 8 if 4 < type_.size <= 8 else 16
        self._next_stack_offset = align(self._next_stack_offset, align_to) + align_to
        var = Variable(name, type_, self._next_stack_offset)
        self._scope[name] = var
        return var

    def resolve_variable(self, name):
        try:
            return self._scope[name]
        except KeyError:
            raise XTypeError(f"No variable named {name!r}")

    def __repr__(self):
        return f"FuncMeta(name={self.name!r}, type={self.type!r}, end_label={self.end_label!r})"  # noqa E501


def xcompile(decls):
    ctx = ProgramContext()

    ctx.emitln(b"	.section	__TEXT,__text")

    for decl in decls:
        assert isinstance(decl, parse.FuncDecl)

        # Create type for function
        params = [compile_type(ctx, param.type) for param in decl.params]
        ret = compile_type(ctx, decl.ret) if decl.ret else None
        func_ty = FunctionType(params, ret)
        end_label = None if decl.is_proto else Label.create()
        func_meta = FuncMeta(decl.name.encode("ascii"), func_ty, end_label)
        ctx.func = func_meta

        ctx.declared_functions[decl.name] = func_meta

        # Just add the type
        if decl.is_proto:
            continue

        symbol_name = global_name(func_meta.name)
        # TODO: Functions static by default
        ctx.emitln(b"")
        ctx.commentln(func_meta.name)
        ctx.emitln(b"	.globl	%s" % symbol_name)
        ctx.emitln(b"%s:" % symbol_name)

        ctx.emitln(b"	pushq	%rbp")
        ctx.emitln(b"	movq	%rsp, %rbp")
        # The number of locals is only known after compiling the contents of
        # the function so the instruction to allocate stack space is patched in
        # after compiling the body.
        allocate_stack_room = ctx.next_lineno()
        ctx.commentln(b"end prologue")

        # Move all the arguments to the stack
        if params:
            ctx.commentln(b"Move all the arguments to the stack")
        for param_ast, param_ty, arg_reg in zip(
            decl.params, params, register.argument_registers
        ):
            var = func_meta.declare_variable(param_ast.name, param_ty)
            ctx.emitln(
                b"	%s	%s, %s"
                % (
                    mov(param_ty.size),
                    register.Address(arg_reg, size=param_ty.size),
                    var.addr,
                )
            )

        ctx.commentln(b"function body")
        for stmt in decl.body:
            compile_statement(ctx, stmt)

        ctx.emitln(b"%s:" % end_label)

        locals_size = register.Immediate(align(func_meta._next_stack_offset, 16))
        # Patch in sub to reserve stack space for locals
        ctx.insertln(allocate_stack_room, b"	subq	%s, %%rsp" % locals_size)
        ctx.commentln(b"epilogue")
        ctx.emitln(b"	addq	%s, %%rsp" % locals_size)
        ctx.emitln(b"	popq	%rbp")
        ctx.emitln(b"	ret")

        ctx.func = None

    if ctx.strings:
        ctx.emitln(b"")
        ctx.emitln(b"	.section	__TEXT,__cstring")

    for label, string in ctx.strings:
        ctx.emitln(b'%s:	.asciz "%s"' % (label, string.encode("utf-8")))

    return opt.peephole_opt(ctx.asm)


def compile_type(ctx, ast_ty):
    if isinstance(ast_ty, parse.NamedTypeExpr):
        try:
            return ctx.named_types[ast_ty.name]
        except KeyError:
            raise XTypeError(f"Unknown type {ast_ty.name!r}")
    elif isinstance(ast_ty, parse.PointerTypeExpr):
        return PointerType(compile_type(ctx, ast_ty.pointee))
    else:
        raise NotImplementedError(repr(ast_ty))


def compile_statement(ctx, stmt):
    if isinstance(stmt, parse.ReturnStmt):
        if stmt.expr:
            expr_ty = analyze_type(ctx, stmt.expr)
            if not is_assignable(expr_ty, ctx.func.type.ret):
                raise XTypeError(f"Cannot return {expr_ty} as {ctx.func.type.ret}")

            dest = register.Address(register.a, size=expr_ty.size)
            compile_expr(ctx, stmt.expr, dest)
            implicitly_cast(ctx, dest, expr_ty, ctx.func.type.ret)

        ctx.emitln(b"	jmp	%s" % ctx.func.end_label)
    elif isinstance(stmt, parse.VarDecl):
        assert stmt.initializer or stmt.type
        if stmt.type:
            ty = compile_type(ctx, stmt.type)
        elif stmt.initializer:
            ty = analyze_type(ctx, stmt.initializer)
        var = ctx.func.declare_variable(stmt.name, ty)
        if stmt.initializer:
            compile_expr(ctx, stmt.initializer, var.addr)
    elif isinstance(stmt, parse.IfStmt):
        then_label = Label.create()
        else_label = Label.create()
        end_label = Label.create()

        ctx.commentln(b"if")
        compile_logical_expr(ctx, stmt.cond, Jump(else_label, False))
        ctx.emitln(b"%s:" % then_label)
        ctx.comment(b"then")
        ctx.func.enter_scope()
        for then_stmt in stmt.then_body:
            compile_statement(ctx, then_stmt)
        ctx.func.exit_scope()
        ctx.emitln(b"	jmp	%s" % end_label)
        ctx.emitln(b"%s:" % else_label)
        ctx.comment(b"else")
        if stmt.else_body:
            ctx.func.enter_scope()
            for else_stmt in stmt.else_body:
                compile_statement(ctx, else_stmt)
            ctx.func.exit_scope()
        ctx.emitln(b"%s:" % end_label)
        ctx.comment(b"end if")
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
    elif isinstance(expr, parse.BoolExpr):
        if dest is None:
            return
        ctx.emitln(
            b"	%s	%s, %s"
            % (
                mov(dest.size),
                register.Immediate(int(expr.value)),
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
        var = ctx.func.resolve_variable(expr.name)
        if dest != var.addr:
            # FIXME: There should be some kinda move/cast operation
            # FIXME: implicitly cast variable value to destination type if
            # applicable
            ctx.emitln(
                b"	%s	%s, %s"
                % (
                    mov(var.type.size),
                    var.addr,
                    dest,
                )
            )
            ctx.comment(expr.name.encode("ascii"))
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
            # FIXME: implicitly cast arguments to parameter types if applicable
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
            retval_addr = register.Address(register.a, size=func.type.ret.size)
            if dest != retval_addr:
                ctx.emitln(
                    b"	%s	%s, %s"
                    % (
                        mov(func.type.ret.size),
                        retval_addr,
                        dest,
                    )
                )
    elif is_logical_expr(expr):
        false_label = Label.create()
        after_label = Label.create()
        compile_logical_expr(ctx, expr, Jump(false_label, False))
        ctx.emitln(b"	%s	$1, %s" % (mov(dest.size), dest))
        ctx.emitln(b"	jmp	%s" % after_label)
        ctx.emitln(b"%s:" % false_label)
        ctx.emitln(b"	%s	$0, %s" % (mov(dest.size), dest))
        ctx.emitln(b"%s:" % after_label)

    else:
        raise NotImplementedError(expr)


def is_logical_expr(expr):
    return (
        isinstance(expr, parse.UnaryExpr) and expr.op is parse.UnaryOp.LOGICAL_NOT
    ) or (
        isinstance(expr, parse.BinaryExpr)
        and expr.op in (parse.BinaryOp.LOGICAL_OR, parse.BinaryOp.LOGICAL_AND)
    )


class Jump:
    def __init__(self, dest, if_true):
        self.dest = dest
        self.if_true = if_true


def compile_logical_expr(ctx, expr, jump, invert=False):
    if isinstance(expr, parse.UnaryExpr) and expr.op is parse.UnaryOp.LOGICAL_NOT:
        compile_logical_expr(ctx, expr.expr, jump, invert=not invert)
    elif isinstance(expr, parse.BinaryExpr) and expr.op in (
        parse.BinaryOp.LOGICAL_AND,
        parse.BinaryOp.LOGICAL_OR,
    ):
        # De Morgan's law, ask @macdja38
        if (expr.op is parse.BinaryOp.LOGICAL_AND and not invert) or (
            expr.op is parse.BinaryOp.LOGICAL_OR and invert
        ):
            compile_logical_expr(ctx, expr.left, jump, invert)
            compile_logical_expr(ctx, expr.right, jump, invert)
        else:
            after_or = Label.create()
            compile_logical_expr(ctx, expr.left, Jump(after_or, True), invert)
            compile_logical_expr(ctx, expr.right, jump, invert)
            ctx.emitln(b"%s:" % after_or)
    else:
        expr_ty = analyze_type(ctx, expr)
        dest = register.Address(register.a, expr_ty.size)
        compile_expr(ctx, expr, dest)
        ctx.emitln(b"	%s	$0, %s" % (cmp(dest.size), dest))
        ctx.emitln(b"	%s	%s" % (b"jne" if jump.if_true ^ invert else b"je", jump.dest))


def implicitly_cast(ctx, addr, from_ty, to_ty):
    if is_same_type(from_ty, to_ty):
        return

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
