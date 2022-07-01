# vim: et

from collections import namedtuple
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
    COLON = auto()
    COMMA = auto()
    DECLARE = auto()
    ELLIPSIS = auto()
    EQ = auto()
    FUNCTION = auto()
    IDENT = auto()
    INTEGER = auto()
    LBRACE = auto()
    LPAREN = auto()
    RBRACE = auto()
    RETURN = auto()
    RPAREN = auto()
    SEMICOLON = auto()
    STAR = auto()
    STRING = auto()
    VAR = auto()


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
    "(": T.LPAREN,
    ")": T.RPAREN,
    "*": T.STAR,
    ",": T.COMMA,
    ":": T.COLON,
    ";": T.SEMICOLON,
    "{": T.LBRACE,
    "}": T.RBRACE,
}

_keywords = {
    "declare": T.DECLARE,
    "function": T.FUNCTION,
    "return": T.RETURN,
    "var": T.VAR,
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
            yield Token(T.EQ, start, c)
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


class VarDecl(Decl, Stmt):
    def __init__(self, name, type_, initializer):
        super().__init__()
        self.name = name
        self.type = type_
        self.initializer = initializer

    def __repr__(self):
        return f"VarDecl({self.name!r}, {self.type!r}, {self.initializer!r})"


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


class Expr:
    pass


class CallExpr(Expr):
    def __init__(self, target, args):
        super().__init__()
        self.target = target
        self.args = args

    def __repr__(self):
        return f"CallExpr({self.target!r}, {self.args})"


class IdentExpr(Expr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"IdentExpr({self.name!r})"


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
            if tok.type is T.DECLARE:
                yield parse_function_proto(it)
            elif tok.type is T.FUNCTION:
                yield parse_function_decl(it)
            elif tok.type is T.VAR:
                # TODO
                raise NotImplementedError("Global variables")
            else:
                raise UnexpectedToken(tok.pos, tok.text)
        except StopIteration:
            raise UnexpectedEOF()


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
    _expect(next(it), T.DECLARE)
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
        return parse_var_decl(it, local=True)
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


def parse_var_decl(it, local):
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
    if not local and not type_:
        raise ParseError("Global variables must have explicit types")
    if not type_ and not initializer:
        raise ParseError("Variable must have either type annotation or initializer")
    return VarDecl(name, type_, initializer)


def parse_expr_statement(it):
    expr = parse_expression(it)
    _expect(next(it), T.SEMICOLON)
    return ExprStmt(expr)


def parse_expression(it):
    if it.peek().type is T.INTEGER:
        return IntExpr(int(next(it).text))
    elif it.peek().type is T.IDENT:
        expr = IdentExpr(next(it).text)
        if it.peek().type is T.LPAREN:
            next(it)
            args = []
            first = True
            while it.peek().type is not T.RPAREN:
                if first:
                    first = False
                else:
                    _expect(next(it), T.COMMA)
                args.append(parse_expression(it))
            _expect(next(it), T.RPAREN)
            expr = CallExpr(expr, args)
        return expr
    elif it.peek().type is T.STRING:
        return StringExpr(next(it).text[1:-1])
    else:
        tok = next(it)
        raise UnexpectedToken(tok.pos, tok.text)


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
