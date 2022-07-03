import sys
from argparse import ArgumentParser
from x.compile import xcompile
from x.parse import ParseError, parse, tokenize

argparser = ArgumentParser()
argparser.add_argument("file")
argparser.add_argument("-o", dest="out", default="-")
argparser.add_argument("--dump-ast", action="store_true")
args = argparser.parse_args()

with open(args.file, "r") as f:
    text = f.read()

try:
    ast = parse(tokenize(text))
    if args.dump_ast:
        print(repr(list(ast)))
        exit(0)
    assembly = xcompile(ast)
except ParseError as e:
    import traceback
    traceback.print_exc()
    print(repr(text[max(e.pos - 10, 0) : e.pos + 10]))  # noqa E203
    exit(1)

if args.out == "-":
    sys.stdout.buffer.write(assembly)
else:
    with open(args.out, "wb") as f:
        f.write(assembly)
