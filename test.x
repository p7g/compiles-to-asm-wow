extern function printf(fmt: *u8, intval: i32);

function main(): i32 {
    var i = 0;

    while !(i == 10) {
        printf("%d", i);
        i = i + 1;
    }

    return i;
}
