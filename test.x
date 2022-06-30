declare function puts(str: *u8);
declare function strlen(str: *u8): u32;

function inner(): i32 {
    puts("Testing");
    return strlen("wowowowow");
}

function main(argc: i32, argv: **u8): i32 {
    puts("Hello, world!");
    return inner();
}
