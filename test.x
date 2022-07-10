extern function malloc(size: u64): *void;
extern function printf(fmt: *u8, i: i32);

function print_int(i: i32) {
    printf("%d\n", i);
}

type s = {b: i64, a: i32};

function make_s(): *s {
    return malloc(sizeof(type s));
}

function main(): i32 {
    var s = make_s();

    s.a = 0;
    print_int(s.a);

    s.a += 1;
    print_int(s.a);
    s.a += 1;
    print_int(s.a);

    return s.a;
}
