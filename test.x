extern function printf(fmt: *u8, i: i32, i2: i32);
extern function free(ptr: *void);
extern function malloc(size: u64): *void;

function mul(a: i32, b: i32): i32 {
    var acc = 0;
    var i = 0;
    while !(i == b) {
        acc += a;
        i += 1;
    }
    return acc;
}

function main(): i32 {
    var numints = 10;

    var ints: *i32 = malloc(mul(sizeof(type i32), numints));
    var i = -1;
    while i != 9 {
        ints[i] = (i += 1);
    }

    i = 0;
    while i != 10 {
        printf("%d%c", ints[i], 32);
        i += 1;
    }
    printf("%c%c", 32, 10);

    free(ints);

    return 0;
}
