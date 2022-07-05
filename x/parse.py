# vim: et

from collections import defaultdict, namedtuple
from enum import Enum, auto

Token = namedtuple("Token", "type pos text")
_nothing = object()


class Peekable:
    def __init__(self, it):
        self._it = it
        self._peeked = _nothing

    def __next__(self):
        if self._peeked is not _nothing:
            val = self._peeked
            self._peeked = _nothing
        else:
            val = next(self._it)
        return val

    def peek(self):
        if self._peeked is _nothing:
            self._peeked = next(self._it)
        return self._peeked


class T(Enum):
    AMPERSAND = auto()
    ANDAND = auto()
    BANG = auto()
    COLON = auto()
    COMMA = auto()
    ELLIPSIS = auto()
    ELSE = auto()
    EXTERN = auto()
    EQ = auto()
    EQEQ = auto()
    FALSE = auto()
    FUNCTION = auto()
    IDENT = auto()
    IF = auto()
    INTEGER = auto()
    LBRACE = auto()
    LBRACKET = auto()
    LPAREN = auto()
    OROR = auto()
    PLUS = auto()
    RBRACE = auto()
    RBRACKET = auto()
    RETURN = auto()
    RPAREN = auto()
    SEMICOLON = auto()
    STAR = auto()
    STRING = auto()
    TRUE = auto()
    VAR = auto()
    WHILE = auto()


class ParseError(Exception):
    def __init__(self, pos, message):
        super().__init__(message)
        self.pos = pos


class UnexpectedToken(ParseError):
    def __init__(self, pos, token):
        super().__init__(pos, f"Unexpected token '{token}' at {pos}")


class UnexpectedEOF(ParseError):
    def __init__(self):
        super().__init__(-1, "Unexpected end of file")


class ParamsAfterEllipsis(ParseError):
    def __init__(self, pos):
        super().__init__(pos, f"Parameters after ... are not allowed at {pos}")


_one_char = {
    "!": T.BANG,
    "(": T.LPAREN,
    ")": T.RPAREN,
    "*": T.STAR,
    "+": T.PLUS,
    ",": T.COMMA,
    ":": T.COLON,
    ";": T.SEMICOLON,
    "[": T.LBRACKET,
    "]": T.RBRACKET,
    "{": T.LBRACE,
    "}": T.RBRACE,
}

_keywords = {
    "else": T.ELSE,
    "extern": T.EXTERN,
    "false": T.FALSE,
    "function": T.FUNCTION,
    "if": T.IF,
    "return": T.RETURN,
    "true": T.TRUE,
    "var": T.VAR,
    "while": T.WHILE,
}


def tokenize(text):
    chars = Peekable(iter(text))
    pos = 0

    def nextchar():
        nonlocal pos
        c = next(chars)
        pos += 1
        return c

    peekchar = chars.peek

    while True:
        start = pos

        try:
            c = nextchar()
            while c.isspace():
                c = nextchar()
        except StopIteration:
            return

        if c in _one_char:
            yield Token(_one_char[c], start, c)
        elif c == ".":
            c = peekchar()
            if c != ".":
                raise UnexpectedToken(start, ".")
            nextchar()
            c = peekchar()
            if c != ".":
                raise UnexpectedToken(start, "..")
            nextchar()
            yield Token(T.ELLIPSIS, start, "...")
        elif c == "=":
            if peekchar() == "=":
                c += nextchar()
                yield Token(T.EQEQ, start, c)
            else:
                yield Token(T.EQ, start, c)
        elif c == "&":
            if peekchar() != "&":
                yield Token(T.AMPERSAND, start, c)
            else:
                c += nextchar()
                yield Token(T.ANDAND, start, c)
        elif c == "|":
            if peekchar() != "|":
                raise UnexpectedToken(start, c)
            c += nextchar()
            yield Token(T.OROR, start, c)
        elif c == '"':
            while peekchar() != '"':
                c += nextchar()
            c += nextchar()
            yield Token(T.STRING, start, c)
        elif c.isalpha() or c == "_":
            while peekchar().isalnum() or peekchar() == "_":
                c += nextchar()
            yield Token(_keywords.get(c, T.IDENT), start, c)
        elif c.isdigit():
            while peekchar().isdigit():
                c += nextchar()
            yield Token(T.INTEGER, start, c)
        else:
            raise UnexpectedToken(pos, c)


class Decl:
    pass


class FuncDecl(Decl):
    def __init__(self, name, params, variadic, ret, body):
        super().__init__()
        self.name = name
        self.params = params
        self.variadic = variadic
        self.ret = ret
        self.body = body

    @property
    def is_proto(self):
        return self.body is None

    def __repr__(self):
        return f"FuncDecl({self.name!r}, {self.params}, {self.ret!r}, {self.body!r})"


class Stmt:
    pass


class AbstractVarDecl(Decl, Stmt):
    def __init__(self, name, type_):
        super().__init__()
        self.name = name
        self.type = type_

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, {self.type!r})"


class VarDecl(AbstractVarDecl):
    def __init__(self, name, type_, initializer):
        super().__init__(name, type_)
        self.initializer = initializer

    def __repr__(self):
        return f"VarDecl({self.name!r}, {self.type!r}, {self.initializer!r})"


class ExternVarDecl(AbstractVarDecl):
    pass


class ExprStmt(Stmt):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def __repr__(self):
        return f"ExprStmt({self.expr!r})"


class ReturnStmt(Stmt):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def __repr__(self):
        return f"ReturnStmt({self.expr!r})"


class IfStmt(Stmt):
    def __init__(self, cond, then_body, else_body):
        super().__init__()
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body

    def __repr__(self):
        return f"IfStmt({self.cond!r}, {self.then_body!r}, {self.else_body!r})"


class WhileLoop(Stmt):
    def __init__(self, cond, body):
        super().__init__()
        self.cond = cond
        self.body = body

    def __repr__(self):
        return f"WhileLoop({self.cond!r}, {self.body!r})"


class Expr:
    pass


class BinaryExpr(Expr):
    def __init__(self, left, op, right):
        super().__init__()
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryExpr({self.left!r}, {self.op!r}, {self.right!r})"


class UnaryExpr(Expr):
    def __init__(self, op, expr):
        super().__init__()
        self.op = op
        self.expr = expr

    def __repr__(self):
        return f"UnaryOp({self.op!r}, {self.expr!r})"


class CallExpr(Expr):
    def __init__(self, target, args):
        super().__init__()
        self.target = target
        self.args = args

    def __repr__(self):
        return f"CallExpr({self.target!r}, {self.args})"


class IndexExpr(Expr):
    def __init__(self, target, index):
        super().__init__()
        self.target = target
        self.index = index

    def __repr__(self):
        return f"IndexExpr({self.target!r}, {self.index!r})"


class IdentExpr(Expr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"IdentExpr({self.name!r})"


class BoolExpr(Expr):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"BoolExpr({self.value!r})"


class IntExpr(Expr):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f"IntExpr({self.value!r})"


class StringExpr(Expr):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def __repr__(self):
        return f"StringExpr({self.text!r})"


class TypeExpr:
    pass


class PointerTypeExpr(TypeExpr):
    def __init__(self, pointee):
        super().__init__()
        self.pointee = pointee

    def __repr__(self):
        return f"PointerTypeExpr({self.pointee!r})"


class NamedTypeExpr(TypeExpr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"NamedTypeExpr({self.name!r})"


def _expect(tok, type_):
    if tok.type is not type_:
        raise UnexpectedToken(tok.pos, tok.text)
    return tok


def parse(tokens):
    it = Peekable(iter(tokens))

    while True:
        try:
            tok = it.peek()
        except StopIteration:
            return

        try:
            if tok.type is T.EXTERN:
                yield from parse_extern_decl(it)
            elif tok.type is T.FUNCTION:
                yield parse_function_decl(it)
            elif tok.type is T.VAR:
                # TODO
                raise NotImplementedError("Global variables")
            else:
                raise UnexpectedToken(tok.pos, tok.text)
        except StopIteration:
            raise UnexpectedEOF()


def parse_extern_decl(it):
    _expect(next(it), T.EXTERN)
    ty = it.peek().type

    if ty is T.FUNCTION:
        yield parse_function_proto(it)
    elif ty is T.VAR:
        yield parse_extern_var_decl(it)
    else:
        raise UnexpectedToken(ty)


Param = namedtuple("Param", "name type")


def parse_function_sig(it):
    _expect(next(it), T.FUNCTION)
    name = _expect(next(it), T.IDENT).text
    _expect(next(it), T.LPAREN)

    params = []
    first = True
    variadic = False
    while it.peek().type is not T.RPAREN:
        if not first:
            _expect(next(it), T.COMMA)
        else:
            first = False

        if it.peek().type is T.ELLIPSIS:
            next(it)
            variadic = True
        elif variadic:
            raise ParamsAfterEllipsis(it.peek().pos)
        else:
            argname = _expect(next(it), T.IDENT).text
            _expect(next(it), T.COLON)
            ty = parse_type_expr(it)
            params.append(Param(argname, ty))

    _expect(next(it), T.RPAREN)

    if it.peek().type is T.COLON:
        next(it)
        ret = parse_type_expr(it)
    else:
        ret = None

    return FuncDecl(name, params, variadic, ret, None)


def parse_function_proto(it):
    decl = parse_function_sig(it)
    _expect(next(it), T.SEMICOLON)
    return decl


def parse_function_decl(it):
    decl = parse_function_sig(it)
    _expect(next(it), T.LBRACE)

    body = []
    while it.peek().type is not T.RBRACE:
        body.append(parse_statement(it))
    _expect(next(it), T.RBRACE)

    decl.body = body
    return decl


def parse_statement(it):
    if it.peek().type is T.RETURN:
        return parse_return_statement(it)
    elif it.peek().type is T.VAR:
        return parse_local_var_decl(it)
    elif it.peek().type is T.IF:
        return parse_if_statement(it)
    elif it.peek().type is T.WHILE:
        return parse_while_loop(it)
    else:
        return parse_expr_statement(it)


def parse_return_statement(it):
    _expect(next(it), T.RETURN)
    if it.peek().type is T.SEMICOLON:
        expr = None
    else:
        expr = parse_expression(it)
    _expect(next(it), T.SEMICOLON)
    return ReturnStmt(expr)


def parse_local_var_decl(it):
    _expect(next(it), T.VAR)
    name = _expect(next(it), T.IDENT).text
    if it.peek().type is T.COLON:
        next(it)
        type_ = parse_type_expr(it)
    else:
        type_ = None
    if it.peek().type is T.EQ:
        next(it)
        initializer = parse_expression(it)
    else:
        initializer = None
    _expect(next(it), T.SEMICOLON)
    if not type_ and not initializer:
        raise ParseError("Variable must have either type annotation or initializer")
    return VarDecl(name, type_, initializer)


def parse_extern_var_decl(it):
    _expect(next(it), T.VAR)
    name = _expect(next(it), T.IDENT).text
    _expect(next(it), T.COLON)
    type_ = parse_type_expr(it)
    _expect(next(it), T.SEMICOLON)
    return ExternVarDecl(name, type_)


def parse_if_statement(it):
    _expect(next(it), T.IF)

    cond = parse_expression(it)
    _expect(next(it), T.LBRACE)

    then_body = []
    while it.peek().type is not T.RBRACE:
        then_body.append(parse_statement(it))

    _expect(next(it), T.RBRACE)

    if it.peek().type is T.ELSE:
        next(it)
        if it.peek().type is T.IF:
            else_body = [parse_if_statement(it)]
        else:
            _expect(next(it), T.LBRACE)

            else_body = []
            while it.peek().type is not T.RBRACE:
                else_body.append(parse_statement(it))

        _expect(next(it), T.RBRACE)
    else:
        else_body = None

    return IfStmt(cond, then_body, else_body)


def parse_while_loop(it):
    _expect(next(it), T.WHILE)

    cond = parse_expression(it)
    _expect(next(it), T.LBRACE)

    body = []
    while it.peek().type is not T.RBRACE:
        body.append(parse_statement(it))

    _expect(next(it), T.RBRACE)
    return WhileLoop(cond, body)


def parse_expr_statement(it):
    expr = parse_expression(it)
    _expect(next(it), T.SEMICOLON)
    return ExprStmt(expr)


class UnaryOp(Enum):
    DEREFERENCE = T.STAR
    LOGICAL_NOT = T.BANG
    REFERENCE = T.AMPERSAND


class Assoc(Enum):
    LEFT = auto()
    RIGHT = auto()


class BinaryOp(Enum):
    ADDITION = T.PLUS
    ASSIGNMENT = T.EQ
    LOGICAL_AND = T.ANDAND
    LOGICAL_OR = T.OROR
    EQUAL = T.EQEQ


precedence = dict(
    (op, prec)
    for prec, op in enumerate(
        [
            BinaryOp.ADDITION,
            BinaryOp.EQUAL,
            BinaryOp.LOGICAL_AND,
            BinaryOp.LOGICAL_OR,
            BinaryOp.ASSIGNMENT,
        ]
    )
)

associativity = defaultdict(
    lambda: Assoc.LEFT,
    {
        BinaryOp.ASSIGNMENT: Assoc.RIGHT,
    },
)


def parse_expression(it):
    expr_stack = [parse_unary_op(it)]
    operator_stack = []

    def reduce():
        right, left = expr_stack.pop(), expr_stack.pop()
        expr_stack.append(BinaryExpr(left, operator_stack.pop(), right))

    while True:
        try:
            op = BinaryOp(it.peek().type)
        except ValueError:
            break
        else:
            next(it)

        if operator_stack:
            prev_op = operator_stack[-1]

            if (op is prev_op and associativity[op] is Assoc.LEFT) or precedence[
                prev_op
            ] < precedence[op]:
                reduce()

        operator_stack.append(op)
        expr_stack.append(parse_unary_op(it))

    while operator_stack:
        reduce()

    result = expr_stack.pop()
    assert not expr_stack
    return result


def parse_unary_op(it):
    tok = it.peek()
    try:
        UnaryOp(tok.type)
    except ValueError:
        expr = parse_atom(it)
        while it.peek().type in (T.LPAREN, T.LBRACKET):
            if it.peek().type is T.LPAREN:
                expr = parse_call_expr(it, expr)
            elif it.peek().type is T.LBRACKET:
                expr = parse_index_expr(it, expr)
        return expr
    else:
        return UnaryExpr(UnaryOp(next(it).type), parse_unary_op(it))


def parse_atom(it):
    ty = it.peek().type
    if ty is T.LPAREN:
        next(it)
        expr = parse_expression(it)
        _expect(next(it), T.RPAREN)
        return expr
    elif ty is T.INTEGER:
        return IntExpr(int(next(it).text))
    elif ty is T.IDENT:
        return IdentExpr(next(it).text)
    elif ty is T.STRING:
        return StringExpr(next(it).text[1:-1])
    elif ty in (T.TRUE, T.FALSE):
        return BoolExpr(next(it).type is T.TRUE)
    else:
        tok = next(it)
        raise UnexpectedToken(tok.pos, tok.text)


def parse_call_expr(it, target):
    _expect(next(it), T.LPAREN)
    args = []
    first = True
    while it.peek().type is not T.RPAREN:
        if first:
            first = False
        else:
            _expect(next(it), T.COMMA)
        args.append(parse_expression(it))
    _expect(next(it), T.RPAREN)
    return CallExpr(target, args)


def parse_index_expr(it, target):
    _expect(next(it), T.LBRACKET)
    index = parse_expression(it)
    _expect(next(it), T.RBRACKET)
    return IndexExpr(target, index)


def parse_type_expr(it):
    if it.peek().type is T.IDENT:
        return NamedTypeExpr(next(it).text)
    elif it.peek().type is T.STAR:
        next(it)
        pointee = parse_type_expr(it)
        return PointerTypeExpr(pointee)
    else:
        tok = next(it)
        raise UnexpectedToken(tok.pos, tok.text)
