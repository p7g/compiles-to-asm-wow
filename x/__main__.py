import sys
from argparse import ArgumentParser
from x.compile import xcompile
from x.parse import ParseError, parse, tokenize

argparser = ArgumentParser()
argparser.add_argument("file")
argparser.add_argument("-o", dest="out", default="-")
args = argparser.parse_args()

with open(args.file, "r") as f:
    text = f.read()

try:
    assembly = xcompile(parse(tokenize(text)))
except ParseError as e:
    print(e, repr(text[max(e.pos - 10, 0) : e.pos + 10]))  # noqa E203
    exit(1)

if args.out == "-":
    sys.stdout.buffer.write(assembly)
else:
    with open(args.out, "wb") as f:
        f.write(assembly)
