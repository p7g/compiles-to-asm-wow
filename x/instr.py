_size_suffix = [None, b"b", b"w", None, b"l", None, None, None, b"q"]


def cmp(size):
    return b"cmp%s" % _size_suffix[size]


def mov(size):
    return b"mov%s" % _size_suffix[size]


def movs(src_size, dest_size):
    return b"movs%s%s" % (_size_suffix[src_size], _size_suffix[dest_size])


def sete(size):
    return b"sete%s" % _size_suffix[size]


def setne(size):
    return b"setne%s" % _size_suffix[size]
