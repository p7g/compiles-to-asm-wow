extern function malloc(size: u64): *void;
extern function realloc(ptr: *void, new_size: u64): *void;
extern function free(ptr: *void);
extern function printf(fmt: *u8, i: i32);

type int = i32;

function mul(a: u64, b: u64): u64 {
    var acc: u64 = 0;

    while b {
        acc += a;
        b += -1;
    }

    return acc;
}

type intlist = {
    len: u32,
    cap: u32,
    items: *i32,
};

function intlist_init(list: *intlist) {
    list.cap = list.len = 0;
    list.items = null;
}

function intlist_deinit(list: *intlist) {
    if list.items {
        free(list.items);
    }
}

function intlist_append(list: *intlist, item: int) {
    if list.len == list.cap {
        if list.cap {
            list.cap += list.cap;
        } else {
            list.cap = 4;
        }
        list.items = realloc(list.items, mul(list.cap, sizeof(type int)));
    }

    list.items[list.len] = item;
    list.len += 1;
}

function main(): i32 {
    var l: intlist;

    intlist_init(&l);

    var i = 0;
    while i != 10 {
        intlist_append(&l, i);
        i += 1;
    }

    i = 0;
    while i != l.len {
        printf("%d\n", l.items[i]);
        i += 1;
    }

    return 0;
}
