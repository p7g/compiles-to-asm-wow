from collections import ChainMap
from contextlib import ExitStack, contextmanager
from x import opt, parse, register
from x.instr import add, cmp, mov, movs, movz


# Type system


class XTypeError(Exception):
    pass


class Type:
    pass


class VoidType(Type):
    size = 0

    def __str__(self):
        return "void"


class OpaqueType(Type):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name

    @property
    def size(self):
        raise XTypeError("Opaque types have unknown size")


class IntType(Type):
    def __init__(self, size, signed, from_literal=False):
        super().__init__()
        self.size = size
        self.signed = signed
        self.from_literal = from_literal

    def __str__(self):
        bit_size = self.size << 3
        return f"i{bit_size}" if self.signed else f"u{bit_size}"


class PlaceholderType(Type):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @property
    def size(self):
        raise XTypeError(f"Type {self.name} is incomplete")

    def __str__(self):
        return f"placeholder({self.name})"


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


class StructType(Type):
    def __init__(self, name, members, size):
        super().__init__()
        self.name = name
        self.members = {m.name: m for m in members}
        self.size = size

    def get_member(self, name):
        if name not in self.members:
            raise XTypeError(f"Cannot access field {name} of {self}")
        return self.members[name]

    def __str__(self):
        return self.name


class StructMember:
    def __init__(self, name, type_, offset):
        self.name = name
        self.type = type_
        self.offset = offset


def analyze_type(ctx, expr):
    if isinstance(expr, parse.IdentExpr):
        try:
            return ctx.resolve_variable(expr.name).type
        except XTypeError:
            raise XTypeError(f"No variable named {expr.name!r} is in scope")
    elif isinstance(expr, parse.NullExpr):
        return PointerType(VoidType())
    elif isinstance(expr, parse.IntExpr):
        return IntType(4, signed=True, from_literal=True)
    elif isinstance(expr, parse.SizeofExpr):
        return IntType(8, signed=False, from_literal=True)
    elif isinstance(expr, parse.BoolExpr):
        return BoolType()
    elif isinstance(expr, parse.StringExpr):
        return PointerType(IntType(1, signed=False))
    elif isinstance(expr, parse.CharExpr):
        return ctx.named_types["char"]
    elif isinstance(expr, parse.DotExpr):
        target_ty = analyze_type(ctx, expr.target)
        if isinstance(target_ty, PointerType):
            target_ty = target_ty.pointee
        if not isinstance(target_ty, StructType):
            raise XTypeError(
                f"Can only access members of structs or pointers to structs, not {target_ty}"  # noqa E501
            )
        member = target_ty.get_member(expr.member_name)
        return member.type
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
    elif isinstance(expr, parse.IndexExpr):
        target_ty = analyze_type(ctx, expr.target)
        index_ty = analyze_type(ctx, expr.index)
        if not isinstance(target_ty, PointerType):
            raise XTypeError(f"Can only index pointers, not {target_ty}")
        if not isinstance(index_ty, IntType):
            raise XTypeError(f"Pointer index must be integer, not {index_ty}")
        return target_ty.pointee
    elif is_logical_expr(expr):
        return BoolType()
    elif isinstance(expr, parse.UnaryExpr):
        if expr.op is parse.UnaryOp.REFERENCE:
            return PointerType(analyze_type(ctx, expr.expr))
        elif expr.op is parse.UnaryOp.DEREFERENCE:
            expr_ty = analyze_type(ctx, expr.expr)
            if not isinstance(expr_ty, PointerType):
                raise XTypeError(f"Can only dereference pointers, not {expr_ty}")
            return expr_ty.pointee
        else:
            raise NotImplementedError
    elif isinstance(expr, parse.BinaryExpr):
        if expr.op in (parse.BinaryOp.ASSIGNMENT, parse.BinaryOp.ADDITION_ASSIGNMENT):
            return analyze_type(ctx, expr.left)
        elif expr.op is parse.BinaryOp.ADDITION:
            left_ty, right_ty = analyze_type(ctx, expr.left), analyze_type(
                ctx, expr.right
            )
            if (
                not isinstance(left_ty, IntType)
                or not isinstance(right_ty, IntType)
                or not is_assignable(right_ty, left_ty)
            ):
                raise XTypeError(f"Cannot add {right_ty} to {left_ty}")
            return left_ty
        raise NotImplementedError(expr.op)
    else:
        raise NotImplementedError(expr)


def is_assignable(from_ty, to_ty):
    if isinstance(from_ty, VoidType) or isinstance(to_ty, VoidType):
        return False
    elif isinstance(from_ty, IntType) and isinstance(to_ty, IntType):
        if from_ty.size <= to_ty.size:
            return True
        elif from_ty.from_literal:
            return True  # TODO: Maybe warn if number will be truncated?
        else:
            return False  # TODO: Explain that explicit cast is needed
    elif (
        isinstance(from_ty, PointerType)
        and isinstance(to_ty, PointerType)
        and (
            isinstance(from_ty.pointee, VoidType) or isinstance(to_ty.pointee, VoidType)
        )
    ):
        return True
    else:
        return is_same_type(from_ty, to_ty)


def is_same_type(a, b, seen=()):
    if a is b:
        return True

    if (a, b) in seen:
        return False
    else:
        seen = {*seen, (a, b)}

    if isinstance(a, IntType) and isinstance(b, IntType):
        return a.size == b.size
    elif isinstance(a, PointerType) and isinstance(b, PointerType):
        return is_same_type(a.pointee, b.pointee, seen)
    elif isinstance(a, FunctionType) and isinstance(b, FunctionType):
        return (
            len(a.params) == len(b.params)
            and all(
                is_same_type(param_a, param_b, seen)
                for param_a, param_b in zip(a.params, b.params)
            )
            and is_same_type(a.ret, b.ret, seen)
        )
    elif isinstance(a, BoolType) and isinstance(b, BoolType):
        return True
    elif isinstance(a, VoidType) and isinstance(b, VoidType):
        return True
    elif isinstance(a, OpaqueType):
        return a is b
    elif isinstance(a, StructType) and isinstance(b, StructType):
        if a.members.keys() ^ b.members.keys():
            return False
        for mem_a, mem_b in zip(a.members.values(), b.members.values()):
            if mem_a.name != mem_b.name:
                return False
            elif mem_a.offset != mem_b.offset:
                return False
            elif not is_same_type(mem_a.type, mem_b.type, seen):
                return False
        return True
    else:
        return False


def replace_placeholder_type(in_type, placeholder, with_type, seen=()):
    if in_type in seen:
        return
    else:
        seen = {*seen, in_type}
    if isinstance(in_type, PointerType):
        if in_type.pointee is placeholder:
            in_type.pointee = with_type
        else:
            replace_placeholder_type(in_type.pointee, placeholder, with_type, seen)
    elif isinstance(in_type, FunctionType):
        raise NotImplementedError
    elif isinstance(in_type, StructType):
        for member in in_type.members.values():
            if member.type is placeholder:
                member.type = with_type
            else:
                replace_placeholder_type(member.type, placeholder, with_type, seen)


# Code generation


def global_name(name):
    return b"_%s" % name


def alignment_for_size(size):
    return 4 if size <= 4 else 8 if 4 < size <= 8 else 16


class ProgramContext:
    def __init__(self):
        self.global_scope = {}
        self.strings = []
        self.named_types = {
            "bool": BoolType(),
            "char": IntType(1, signed=False),  # FIXME: unicode?
            "i8": IntType(1, signed=True),
            "u8": IntType(1, signed=False),
            "i16": IntType(2, signed=True),
            "u16": IntType(2, signed=False),
            "i32": IntType(4, signed=True),
            "u32": IntType(4, signed=False),
            "i64": IntType(8, signed=True),
            "u64": IntType(8, signed=False),
            "void": VoidType(),
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
        self._asm_lines[-1] += b"	## " + message

    def commentln(self, message):
        self._asm_lines.append(b"	## " + message)

    def declare_global(self, name, type_, extern):
        var = GlobalVariable(name.encode("ascii"), type_, extern)
        self.global_scope[name] = var
        return var

    @property
    def asm(self):
        return b"\n".join(self._asm_lines)

    def declare_variable(self, name, type_):
        if self.func:
            ns = self.func._scope.maps[0]
        else:
            ns = self.global_scope

        if name in ns:
            raise XTypeError(f"Cannot redeclare variable {name!r}")
        if self.func:
            align_to = alignment_for_size(type_.size)
            self.func._next_stack_offset += type_.size
            self.func._next_stack_offset = align(self.func._next_stack_offset, align_to)
            var = LocalVariable(name, type_, self.func._next_stack_offset)
        else:
            raise NotImplementedError

        ns[name] = var
        return var

    def resolve_variable(self, name):
        try:
            if self.func:
                return self.func._scope[name]
            else:
                return self.global_scope[name]
        except KeyError:
            raise XTypeError(f"No variable named {name!r}")

    @contextmanager
    def spill(self, addr):
        assert self.func, "Cannot spill to stack outside of function"
        with self.func.stack_temp(IntType(addr.size, False)) as var:
            try:
                yield var.addr
            finally:
                self.emitln(var.load(addr))


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
    def __init__(self, name, type_):
        self.name = name
        self.type = type_

    def load(self, dest):
        raise NotImplementedError

    def store(self, source):
        raise NotImplementedError

    def load_addr(self, dest):
        raise NotImplementedError


class LocalVariable(Variable):
    def __init__(self, name, type_, stack_offset):
        super().__init__(name, type_)
        self.stack_offset = stack_offset

    @property
    def addr(self):
        return register.Address(
            register.bp,
            size=self.type.size,
            offset=-self.stack_offset,
        )

    def load(self, dest):
        if not dest.is_plain_reg():
            transient_reg = register.Address(register.a, self.type.size)
            return b"\n".join(
                [
                    b"	%s	%s, %s" % (mov(self.type.size), self.addr, transient_reg),
                    b"	%s	%s, %s" % (mov(dest.size), transient_reg, dest),
                ]
            )
        return b"	%s	%s, %s" % (mov(dest.size), self.addr, dest)

    def load_addr(self, dest):
        return b"	leaq	%s, %s" % (self.addr, dest)

    def store(self, source, op=mov):
        if not source.is_plain_reg():
            transient_reg = register.Address(register.a, self.type.size)
            return b"\n".join(
                [
                    b"	%s	%s, %s" % (mov(self.type.size), source, transient_reg),
                    b"	%s	%s, %s" % (op(self.type.size), transient_reg, self.addr),
                ]
            )
        return b"	%s	%s, %s" % (
            op(self.type.size),
            source.with_size(self.type.size),
            self.addr,
        )


class GlobalVariable(Variable):
    def __init__(self, name, type_, extern):
        super().__init__(name, type_)
        self.extern = extern

    def load(self, dest):
        if not self.extern:
            raise NotImplementedError
        return b"\n".join(
            [
                b"	movq	%s@GOTPCREL(%%rip), %s"
                % (global_name(self.name), register.Address(register.a, 8)),
                b"	%s	%s, %s"
                % (
                    mov(self.type.size),
                    register.Address(register.a, self.type.size, offset=b"0"),
                    dest,
                ),
            ]
        )

    def load_addr(self, dest):
        if not self.extern:
            raise NotImplementedError

        return b"	movq	%s@GOTPCREL(%%rip), %s" % (global_name(self.name), dest)

    def store(self, source, op=mov):
        if not self.extern:
            raise NotImplementedError
        return b"\n".join(
            [
                b"	movq	%s@GOTPCREL(%%rip), %s"
                % (global_name(self.name), register.Address(register.a, 8)),
                b"	%s	%s, %s"
                % (
                    op(self.type.size),
                    source,
                    register.Address(register.a, self.type.size, offset=b"0"),
                ),
            ]
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


class FuncMeta(Variable):
    def __init__(self, name, type_, end_label, ctx):
        super().__init__(name, type_)
        self.end_label = end_label
        self.ctx = ctx
        self._scope = ChainMap({}, ctx.global_scope)
        self._next_stack_offset = 0
        self._stack_temps = []
        self._next_stack_temp = 0

    def enter_scope(self):
        self._scope = self._scope.new_child()

    def exit_scope(self):
        self._scope = self._scope.parents

    def load(self, dest):
        return b"	leaq	%s(%rip), %s" % (self.name, dest)

    def store(self, source):
        raise XTypeError("Cannot assign to function")

    @contextmanager
    def stack_temp(self, type_):
        import bisect
        import operator

        key = operator.attrgetter("type.size")

        i = bisect.bisect_left(self._stack_temps, type_.size, key=key)
        if i != len(self._stack_temps):
            tmp = self._stack_temps.pop(i)
            tmp.type = type_
        else:
            tmp = self.ctx.declare_variable(
                b".tmp%s" % str(self._next_stack_temp).encode("ascii"), type_
            )
            self._next_stack_temp += 1

        try:
            yield tmp
        finally:
            bisect.insort(self._stack_temps, tmp, key=key)

    def __repr__(self):
        return f"FuncMeta(name={self.name!r}, type={self.type!r}, end_label={self.end_label!r})"  # noqa E501


def xcompile(decls):
    ctx = ProgramContext()

    ctx.emitln(b"	.section	__TEXT,__text")

    for decl in decls:
        if isinstance(decl, parse.FuncDecl):
            compile_func_decl(ctx, decl)
        elif isinstance(decl, parse.AbstractVarDecl):
            compile_global_var_decl(ctx, decl)
        elif isinstance(decl, parse.ExternTypeDecl):
            ctx.named_types[decl.name] = OpaqueType(decl.name)
        elif isinstance(decl, parse.TypeAliasDecl):
            placeholder = PlaceholderType(decl.name)
            ctx.named_types[decl.name] = placeholder
            ctx.named_types[decl.name] = compile_type(
                ctx, decl.alias_of, name=decl.name
            )
            replace_placeholder_type(
                ctx.named_types[decl.name], placeholder, ctx.named_types[decl.name]
            )
        else:
            raise NotImplementedError

    if ctx.strings:
        ctx.emitln(b"")
        ctx.emitln(b"	.section	__TEXT,__cstring")

    for label, string in ctx.strings:
        ctx.emitln(b'%s:	.asciz "%s"' % (label, string.encode("utf-8")))

    return opt.peephole_opt(ctx.asm)


def compile_global_var_decl(ctx, decl):
    if not isinstance(decl, parse.ExternVarDecl):
        raise NotImplementedError

    ctx.declare_global(decl.name, compile_type(ctx, decl.type), extern=True)


def compile_func_decl(ctx, decl):
    assert isinstance(decl, parse.FuncDecl)

    # Create type for function
    params = [
        compile_type(ctx, param.type, name=f"{decl.name}.{param.name}")
        for param in decl.params
    ]
    ret = (
        compile_type(ctx, decl.ret, name=f"{decl.name}.return")
        if decl.ret
        else VoidType()
    )
    func_ty = FunctionType(params, ret)
    end_label = None if decl.is_proto else Label.create()
    func_meta = FuncMeta(decl.name.encode("ascii"), func_ty, end_label, ctx)
    ctx.func = func_meta

    ctx.global_scope[decl.name] = func_meta

    # Just add the type
    if decl.is_proto:
        return

    symbol_name = global_name(func_meta.name)
    # TODO: Functions static by default
    ctx.emitln(b"")
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
        var = ctx.declare_variable(param_ast.name, param_ty)
        ctx.emitln(var.store(register.Address(arg_reg, size=param_ty.size)))
        ctx.comment(param_ast.name.encode("ascii"))

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


def compile_type(ctx, ast_ty, name=None):
    if isinstance(ast_ty, parse.NamedTypeExpr):
        try:
            return ctx.named_types[ast_ty.name]
        except KeyError:
            raise XTypeError(f"Unknown type {ast_ty.name!r}")
    elif isinstance(ast_ty, parse.PointerTypeExpr):
        return PointerType(compile_type(ctx, ast_ty.pointee))
    elif isinstance(ast_ty, parse.StructTypeExpr):
        next_member_offset = 0
        members = []

        for member_name, type_ast in ast_ty.members:
            member_ty = compile_type(ctx, type_ast, name=f"{name}.{member_name}")
            align_to = alignment_for_size(member_ty.size)
            next_member_offset = align(next_member_offset, align_to)
            members.append(StructMember(member_name, member_ty, next_member_offset))
            next_member_offset += member_ty.size

        return StructType(name, members, next_member_offset)
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
        var = ctx.declare_variable(stmt.name, ty)
        assert isinstance(var, LocalVariable)
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
    elif isinstance(stmt, parse.WhileLoop):
        top_label = Label.create()
        cond_label = Label.create()

        ctx.emitln(b"	jmp	%s" % cond_label)
        ctx.emitln(b"%s:" % top_label)
        ctx.func.enter_scope()
        for body_stmt in stmt.body:
            compile_statement(ctx, body_stmt)
        ctx.func.exit_scope()
        ctx.emitln(b"%s:" % cond_label)
        compile_logical_expr(ctx, stmt.cond, Jump(top_label, True))
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
    elif isinstance(expr, parse.NullExpr):
        if dest is None:
            return
        ctx.emitln(b"	movq	$0, %s" % dest)
        ctx.comment(b"null")
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
    elif isinstance(expr, parse.CharExpr):
        if dest is None:
            return
        ctx.emitln(
            b"	%s	%s, %s"
            % (
                mov(dest.size),
                register.Immediate(ord(expr.text)),
                dest,
            )
        )
        ctx.comment(repr(expr.text).encode("ascii"))
    elif isinstance(expr, parse.IdentExpr):
        var = ctx.resolve_variable(expr.name)
        # FIXME: There should be some kinda move/cast operation
        # FIXME: implicitly cast variable value to destination type if
        # applicable
        ctx.emitln(var.load(dest))
        ctx.comment(expr.name.encode("ascii"))
    elif isinstance(expr, parse.SizeofExpr):
        if isinstance(expr.operand, parse.TypeExpr):
            ty = compile_type(ctx, expr.operand)
        elif isinstance(expr.operand, parse.Expr):
            ty = analyze_type(expr.operand)
        else:
            raise NotImplementedError
        ctx.emitln(b"	%s	%s, %s" % (mov(dest.size), register.Immediate(ty.size), dest))
        ctx.comment(repr(expr).encode("ascii"))
    elif isinstance(expr, parse.CallExpr):
        # TODO: Function pointers
        if not isinstance(expr.target, parse.IdentExpr):
            raise NotImplementedError

        func = ctx.global_scope[expr.target.name]
        if not isinstance(func, FuncMeta):
            raise XTypeError("Cannot call non-function")

        if len(func.type.params) > len(register.argument_registers):
            raise XTypeError(
                f"Function {func.name.decode('ascii')} has too many parameters"
            )
        if len(func.type.params) != len(expr.args):
            raise XTypeError(
                f"Incorrect number of arguments for {func.name.decode('ascii')}"
            )

        for param_ty, arg_ast in zip(func.type.params, expr.args):
            arg_ty = analyze_type(ctx, arg_ast)
            if not is_assignable(arg_ty, param_ty):
                raise XTypeError(f"Cannot pass {arg_ty} as {param_ty}")

        # FIXME: implicitly cast arguments to parameter types if applicable
        reg_param_arg_triples = zip(
            register.argument_registers, func.type.params, expr.args
        )
        trivial_args, complex_args = partition(
            reg_param_arg_triples, lambda triple: is_trivial_expr(triple[2])
        )

        with ExitStack() as exit_stack:
            # compile and spill non-trivial arguments
            for i, (dest_reg, param, arg) in enumerate(complex_args):
                arg_ty = analyze_type(ctx, arg)
                arg_dest = register.Address(dest_reg, size=arg_ty.size)
                if i != len(complex_args) - 1:
                    arg_dest = exit_stack.enter_context(ctx.spill(arg_dest))
                compile_expr(ctx, arg, arg_dest)
                implicitly_cast(ctx, arg_dest, arg_ty, param)

        # compile trivial arguments
        for dest_reg, param, arg in trivial_args:
            arg_ty = analyze_type(ctx, arg)
            dest_addr = register.Address(dest_reg, size=arg_ty.size)
            compile_expr(ctx, arg, dest_addr)
            implicitly_cast(ctx, dest_addr, arg_ty, param)

        ctx.emitln(b"	call	%s" % global_name(func.name))
        if dest is not None:
            retval_addr = register.Address(register.a, size=func.type.ret.size)
            if dest != retval_addr:
                ctx.emitln(
                    b"	%s	%s, %s"
                    % (mov(dest.size), retval_addr.with_size(dest.size), dest)
                )
    elif isinstance(expr, parse.UnaryExpr) and expr.op is parse.UnaryOp.REFERENCE:
        if not is_assignment_target(expr.expr):
            raise XTypeError(f"Cannot take address of {expr.expr!r}")
        compile_addr_of(ctx, expr.expr, dest)
    elif isinstance(expr, parse.UnaryExpr) and expr.op is parse.UnaryOp.DEREFERENCE:
        result_ty = analyze_type(ctx, expr)
        result = dest.with_size(8)
        compile_expr(ctx, expr.expr, result)
        if not result.is_plain_reg():
            rax = register.Address(register.a, 8)
            ctx.emitln(b"	movq	%s, %s" % (result, rax))
            result = rax
        dereferenced = result.with_offset(0)
        if not dest.is_plain_reg():
            ctx.emitln(
                b"	%s	%s, %s"
                % (
                    mov(result_ty.size),
                    dereferenced,
                    dereferenced.with_offset(None).with_size(result_ty.size),
                )
            )
            dereferenced = dereferenced.with_offset(None).with_size(result_ty.size)
        ctx.emitln(b"	%s	%s, %s" % (mov(result_ty.size), dereferenced, dest))
    elif isinstance(expr, parse.IndexExpr):
        result_ty = analyze_type(ctx, expr)
        index_ty = analyze_type(ctx, expr.index)

        target = register.Address(register.a, 8)
        index = register.Address(register.c, index_ty.size)

        compile_expr(ctx, expr.target, target)

        with ExitStack() as exit_stack:
            if not is_trivial_expr(expr.index):
                tmp = exit_stack.enter_context(ctx.spill(target))
                ctx.emitln(b"	movq	%s, %s" % (target, tmp))

            compile_expr(ctx, expr.index, index)
        implicitly_cast(ctx, index, index_ty, IntType(8, signed=False))
        target = target.with_size(8)
        if dest:
            if not dest.is_plain_reg():
                dest2 = register.Address(register.d, result_ty.size)
            else:
                dest2 = dest
            ctx.emitln(
                b"	%s	%s, %s"
                % (
                    mov(result_ty.size),
                    compile_pointer_index(ctx, target.reg, index.reg, result_ty),
                    dest2,
                )
            )
            if not dest.is_plain_reg():
                ctx.emitln(b"	%s	%s, %s" % (mov(result_ty.size), dest2, dest))
    elif isinstance(expr, parse.DotExpr):
        expr_ty = analyze_type(ctx, expr)
        base_reg = dest.reg if dest and dest.is_plain_reg() else register.a
        addr = compile_reference(ctx, expr, base_reg)
        if dest:
            if not dest.is_plain_reg():
                addr2 = register.Address(base_reg, expr_ty.size)
                ctx.emitln(b"	%s	%s, %s" % (mov(expr_ty.size), addr, addr2))
                addr = addr2
            ctx.emitln(b"	%s	%s, %s" % (mov(expr_ty.size), addr, dest))
            ctx.comment(b".%s" % expr.member_name.encode("ascii"))
    elif isinstance(expr, parse.BinaryExpr) and expr.op in (
        parse.BinaryOp.ASSIGNMENT,
        parse.BinaryOp.ADDITION_ASSIGNMENT,
    ):
        if not is_assignment_target(expr.left):
            raise XTypeError(f"Cannot assign to {expr.left!r}")

        right_ty = analyze_type(ctx, expr.right)
        left_ty = analyze_type(ctx, expr.left)

        op = assignment_op(expr.op)

        # FIXME: "and op is mov" may not be safe?
        if not is_assignable(right_ty, left_ty) and op is mov:
            raise XTypeError(
                f"Cannot assign value of type {right_ty} to {expr.left}, which is {left_ty}"  # noqa E501
            )

        rhs = register.Address(
            register.c if dest and dest.reg is register.d else register.d, right_ty.size
        )
        compile_expr(ctx, expr.right, rhs)
        if right_ty.size < left_ty.size:
            implicitly_cast(ctx, rhs, right_ty, left_ty)

        compile_assignment(ctx, expr.left, rhs.with_size(left_ty.size), dest, op=op)
    elif isinstance(expr, parse.BinaryExpr) and expr.op is parse.BinaryOp.ADDITION:
        left_ty, right_ty = analyze_type(ctx, expr.left), analyze_type(ctx, expr.right)
        result_ty = analyze_type(ctx, expr)

        # Compile left into ax, right into cx, add to cx, move to dest
        ax = register.Address(register.a, result_ty.size)
        cx = (
            dest
            if dest.is_plain_reg()
            else register.Address(register.c, result_ty.size)
        )

        compile_expr(ctx, expr.left, ax)

        with ExitStack() as exit_stack:
            if not is_trivial_expr(expr.right):
                tmp = exit_stack.enter_context(ctx.spill(ax))
                ctx.emitln(b"	%s	%s, %s" % (mov(ax.size), ax, tmp))
            compile_expr(ctx, expr.right, cx)

        ctx.emitln(b"	%s	%s, %s" % (add(result_ty.size), ax, cx))
        if dest and dest != cx:
            ctx.emitln(b"	%s	%s, %s" % (mov(result_ty.size), cx, dest))
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


# TODO: Use this function for other stuff
def compile_reference(ctx, expr, base_reg):
    if isinstance(expr, parse.IdentExpr):
        var = ctx.resolve_variable(expr.name)
        addr = register.Address(base_reg, 8)
        ctx.emitln(var.load_addr(addr))
        return addr.with_offset(0)
    elif isinstance(expr, parse.IndexExpr):
        target_ty = analyze_type(ctx, expr.target)
        index_ty = analyze_type(ctx, expr.index)
        if not isinstance(target_ty, PointerType):
            raise XTypeError("Can only index pointers")
        target_dest = register.Address(base_reg, target_ty.size)
        compile_expr(ctx, expr.target, target_dest)

        if isinstance(expr.index, parse.IntExpr):
            offset = expr.index.value * target_ty.pointee.size
            return register.Address(base_reg, target_ty.pointee.size, offset)
        else:
            with ctx.spill(target_dest) as tmp:
                ctx.emitln(b"	%s	%s, %s" % (mov(target_dest.size), target_dest, tmp))
                index_addr = register.Address(
                    register.c if base_reg is register.d else register.d, index_ty.size
                )
                compile_expr(ctx, expr.index, index_addr)
                implicitly_cast(ctx, index_addr, index_ty, IntType(8, False))
            return compile_pointer_index(
                ctx, base_reg, index_addr.reg, target_ty.pointee
            )
    elif isinstance(expr, parse.DotExpr):
        target_ty = analyze_type(ctx, expr.target)

        if isinstance(target_ty, PointerType):
            if not isinstance(target_ty.pointee, StructType):
                raise XTypeError(
                    f"Can only access fields of structs, not {target_ty.pointee}"
                )
            compile_expr(ctx, expr.target, register.Address(base_reg, target_ty.size))
            mem = target_ty.pointee.get_member(expr.member_name)
            return register.Address(base_reg, mem.type.size, mem.offset)
        elif not isinstance(target_ty, StructType):
            raise XTypeError(f"Can only access fields of structs, not {target_ty}")
        else:
            addr = compile_reference(ctx, expr.target, base_reg)
            mem = target_ty.get_member(expr.member_name)
            return addr.with_offset(addr.offset + mem.offset).with_size(mem.type.size)
    else:
        raise NotImplementedError
        # expr_ty = analyze_type(ctx, expr)
        # assert isinstance(expr_ty, PointerType)
        # dest = register.Address(base_reg, expr_ty.size)
        # compile_expr(ctx, expr, dest)


def assignment_op(op):
    if op is parse.BinaryOp.ASSIGNMENT:
        return mov
    elif op is parse.BinaryOp.ADDITION_ASSIGNMENT:
        return add
    else:
        raise NotImplementedError(op)


def compile_assignment(ctx, expr, src, dest, op=mov):
    dest_ty = analyze_type(ctx, expr)

    if isinstance(expr, parse.IdentExpr):
        var = ctx.resolve_variable(expr.name)
        ctx.emitln(var.store(src, op=op))
        if dest:
            ctx.emitln(var.load(dest))
    elif isinstance(expr, parse.UnaryExpr) and expr.op is parse.UnaryOp.DEREFERENCE:
        with ExitStack() as exit_stack:
            if src.is_plain_reg() and not is_trivial_expr(expr.expr):
                saved_result = exit_stack.enter_context(
                    ctx.func.stack_temp(dest_ty)
                ).addr
                ctx.emitln(b"	%s	%s, %s" % (mov(dest_ty.size), src, saved_result))
            else:
                saved_result = src

            reg_a = register.Address(register.a, 8)
            compile_expr(ctx, expr.expr, reg_a)
            reg_c = register.Address(register.c, dest_ty.size)
            ctx.emitln(b"	%s	%s, %s" % (mov(dest_ty.size), saved_result, reg_c))
            ctx.emitln(b"	%s	%s, %s" % (op(dest_ty.size), reg_c, reg_a.with_offset(0)))

            if dest:
                if not dest.is_plain_reg():
                    ctx.emitln(
                        b"	%s	%s, %s"
                        % (
                            mov(dest_ty.size),
                            reg_a.with_offset(0),
                            reg_a.with_size(dest_ty.size),
                        )
                    )
                    ctx.emitln(
                        b"	%s	%s, %s"
                        % (mov(dest_ty.size), reg_a.with_size(dest_ty.size), dest)
                    )
                else:
                    ctx.emitln(
                        b"	%s	%s, %s"
                        % (
                            mov(dest_ty.size),
                            reg_a.with_offset(0),
                            dest,
                        )
                    )
    elif isinstance(expr, parse.IndexExpr):
        result_ty = analyze_type(ctx, expr)
        index_ty = analyze_type(ctx, expr.index)

        target = register.Address(register.a, 8)
        index = register.Address(register.c, index_ty.size)

        compile_expr(ctx, expr.target, target)

        with ExitStack() as exit_stack:
            if not is_trivial_expr(expr.index):
                tmp = exit_stack.enter_context(ctx.spill(target))
                ctx.emitln(b"	movq	%s, %s" % (target, tmp))

            compile_expr(ctx, expr.index, index)

        if not src.is_plain_reg():
            source = register.Address(register.d, result_ty.size)
            ctx.emitln(b"	%s	%s, %s" % (mov(result_ty.size), src, source))
        else:
            source = src
        ctx.emitln(
            b"	%s	%s, %s"
            % (
                op(result_ty.size),
                source,
                compile_pointer_index(ctx, target.reg, index.reg, result_ty),
            )
        )

        if dest:
            reg_d = register.Address(register.d, result_ty.size)
            if not dest.is_plain_reg():
                ctx.emitln(
                    b"	%s	%s, %s"
                    % (
                        mov(result_ty.size),
                        compile_pointer_index(ctx, target.reg, index.reg, result_ty),
                        reg_d,
                    )
                )
                ctx.emitln(b"	%s	%s, %s" % (mov(result_ty.size), reg_d, dest))
            else:
                ctx.emitln(
                    b"	%s	%s, %s"
                    % (
                        mov(result_ty.size),
                        compile_pointer_index(ctx, target.reg, index.reg, result_ty),
                        dest,
                    )
                )
    elif isinstance(expr, parse.DotExpr):
        base_reg = dest.reg if dest and dest.is_plain_reg() else register.a
        # with ctx.spill(src) as src_var:
        #     src_tmp = src
        #     if not src.is_plain_reg():
        #         src_tmp = register.Address(base_reg, src.size)
        #         ctx.emitln(b"	%s	%s, %s" % (mov(src.size), src, src_tmp))
        #     ctx.emitln(b"	%s	%s, %s" % (mov(src.size), src_tmp, src_var))
        #     addr = compile_reference(ctx, expr, base_reg)
        addr = compile_reference(ctx, expr, base_reg)

        if not src.is_plain_reg():
            newsrc = register.Address(register.c, src.size)
            ctx.emitln(b"	%s	%s, %s" % (mov(src.size), src, newsrc))
            src = newsrc

        ctx.emitln(b"	%s	%s, %s" % (op(src.size), src, addr))

        if dest:
            if dest.is_plain_reg():
                ctx.emitln(b"	%s	%s, %s" % (mov(src.size), addr, dest))
            else:
                tmp = register.Address(register.a, src.size)
                ctx.emitln(b"	%s	%s, %s" % (mov(src.size), addr, tmp))
                ctx.emitln(b"	%s	%s, %s" % (mov(src.size), tmp, addr))
    else:
        raise NotImplementedError


def compile_pointer_index(ctx, base_reg, index_reg, element_type):
    if element_type.size in (1, 2, 4, 8):
        return register.Address(
            base_reg,
            element_type.size,
            index_reg=index_reg,
            scale_factor=element_type.size,
        )
    else:
        import math

        assert element_type.size % 2 == 0
        shift = int(math.log(element_type.size) / math.log(2))
        # Scale factor too large, need to emit code to do the math
        base_addr = register.Address(base_reg, 8)
        index_addr = register.Address(index_reg, 8)
        ctx.emitln(b"	shlq	%s, %s" % (register.Immediate(shift), index_addr))
        ctx.emitln(b"	addq	%s, %s" % (index_addr, base_addr))
        return register.Address(base_reg, element_type.size, 0)


def is_trivial_expr(expr):
    return isinstance(
        expr,
        (
            parse.IdentExpr,
            parse.IntExpr,
            parse.StringExpr,
            parse.BoolExpr,
            parse.SizeofExpr,
        ),
    )


def partition(items, key):
    yes, no = [], []
    for item in items:
        if key(item):
            yes.append(item)
        else:
            no.append(item)
    return yes, no


def compile_addr_of(ctx, expr, dest):
    base_reg = dest.reg if dest and dest.is_plain_reg() else register.a
    addr = compile_reference(ctx, expr, base_reg)

    if dest.is_plain_reg():
        ctx.emitln(b"	leaq	%s, %s" % (addr, dest))
    else:
        ctx.emitln(b"	leaq	%s, %%rax" % addr)
        ctx.emitln(b"	movq	%%rax, %s" % dest)

    # if isinstance(expr, parse.IdentExpr):
    #     var = ctx.resolve_variable(expr.name)
    #     ctx.emitln(var.load_addr(dest))
    # else:
    #     raise NotImplementedError


def is_assignment_target(expr):
    return (
        isinstance(expr, parse.IdentExpr)
        or (isinstance(expr, parse.UnaryExpr) and expr.op is parse.UnaryOp.DEREFERENCE)
        or isinstance(expr, parse.IndexExpr)
        or isinstance(expr, parse.DotExpr)
    )


def is_logical_expr(expr):
    return (
        isinstance(expr, parse.UnaryExpr) and expr.op is parse.UnaryOp.LOGICAL_NOT
    ) or (
        isinstance(expr, parse.BinaryExpr)
        and expr.op
        in (
            parse.BinaryOp.EQUAL,
            parse.BinaryOp.INEQUAL,
            parse.BinaryOp.LOGICAL_OR,
            parse.BinaryOp.LOGICAL_AND,
        )
    )


class Jump:
    def __init__(self, dest, if_true):
        self.dest = dest
        self.if_true = if_true


def compile_logical_expr(ctx, expr, jump, invert=False):
    if isinstance(expr, parse.UnaryExpr) and expr.op is parse.UnaryOp.LOGICAL_NOT:
        compile_logical_expr(ctx, expr.expr, jump, invert=not invert)
    elif isinstance(expr, parse.BinaryExpr) and expr.op in (
        parse.BinaryOp.EQUAL,
        parse.BinaryOp.INEQUAL,
        parse.BinaryOp.LOGICAL_AND,
        parse.BinaryOp.LOGICAL_OR,
    ):
        # De Morgan's law, ask @macdja38
        if (expr.op is parse.BinaryOp.LOGICAL_AND and not invert) or (
            expr.op is parse.BinaryOp.LOGICAL_OR and invert
        ):
            compile_logical_expr(ctx, expr.left, jump, invert)
            compile_logical_expr(ctx, expr.right, jump, invert)
        elif (expr.op is parse.BinaryOp.LOGICAL_OR and not invert) or (
            expr.op is parse.BinaryOp.LOGICAL_AND and invert
        ):
            after_or = Label.create()
            compile_logical_expr(ctx, expr.left, Jump(after_or, True), invert)
            compile_logical_expr(ctx, expr.right, jump, invert)
            ctx.emitln(b"%s:" % after_or)
        elif expr.op in (parse.BinaryOp.EQUAL, parse.BinaryOp.INEQUAL):
            left_ty, right_ty = analyze_type(ctx, expr.left), analyze_type(
                ctx, expr.right
            )
            if not is_assignable(right_ty, left_ty) and not is_assignable(
                left_ty, right_ty
            ):
                raise XTypeError(f"Cannot compare {left_ty} with {right_ty}")

            size = max(left_ty.size, right_ty.size)

            with ctx.func.stack_temp(left_ty) as left_var:
                compile_expr(ctx, expr.left, left_var.addr)
                if left_ty.size < right_ty.size:
                    reg_a = register.Address(register.a, left_ty.size)
                    ctx.emitln(
                        b"	%s	%s, %s" % (mov(left_ty.size), left_var.addr, reg_a)
                    )
                    implicitly_cast(ctx, reg_a, left_ty, right_ty)
                    ctx.emitln(
                        b"	%s	%s, %s"
                        % (mov(size), reg_a.with_size(size), left_var.addr)
                    )
                reg_a = register.Address(register.a, right_ty.size)
                compile_expr(ctx, expr.right, reg_a)
                if right_ty.size < left_ty.size:
                    implicitly_cast(ctx, reg_a, right_ty, left_ty)
                ctx.emitln(
                    b"	%s	%s, %s"
                    % (cmp(size), reg_a.with_size(size), left_var.addr.with_size(size))
                )
                ctx.emitln(
                    b"	%s	%s"
                    % (
                        b"je"
                        if jump.if_true ^ invert ^ (expr.op is parse.BinaryOp.INEQUAL)
                        else b"jne",
                        jump.dest,
                    )
                )
        else:
            raise NotImplementedError
    else:
        expr_ty = analyze_type(ctx, expr)
        dest = register.Address(register.a, expr_ty.size)
        compile_expr(ctx, expr, dest)
        ctx.emitln(b"	%s	$0, %s" % (cmp(dest.size), dest))
        ctx.emitln(b"	%s	%s" % (b"jne" if jump.if_true ^ invert else b"je", jump.dest))


def implicitly_cast(ctx, addr, from_ty, to_ty):
    if is_same_type(from_ty, to_ty):
        return

    if isinstance(from_ty, PointerType) and isinstance(to_ty, PointerType):
        return  # pointers same size
    elif not isinstance(from_ty, IntType) or not isinstance(to_ty, IntType):
        raise NotImplementedError((str(from_ty), str(to_ty)))

    if from_ty.size == to_ty.size:
        return

    assert from_ty.size < to_ty.size
    if not from_ty.signed and from_ty.size > 2:
        return

    op = movs if from_ty.signed else movz

    assert addr.is_plain_reg()
    ctx.emitln(
        b"	%s	%s, %s"
        % (
            op(from_ty.size, to_ty.size),
            addr.with_size(from_ty.size),
            addr.with_size(to_ty.size),
        )
    )
