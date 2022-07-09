extern function printf(fmt: *u8, i: i32, i2: i32);
extern function malloc(size: u64): *i32;

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
    var intsize = 4;
    var numints = 10;

    var ints = malloc(mul(intsize, numints));
    var i = 0;
    while !(i == 10) {
        ints[i] = i;
        i += 1;
    }

    i = 0;
    while !(i == 10) {
        printf("%d%c", ints[i], 32);
        i += 1;
    }
    printf("%c%c", 32, 10);

    return 0;
}
