_size_suffix = [None, b"b", b"w", None, b"l", None, None, None, b"q"]


def mov(size):
    return b"mov%s" % _size_suffix[size]


def movs(src_size, dest_size):
    return b"movs%s%s" % (_size_suffix[src_size], _size_suffix[dest_size])
