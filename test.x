declare function puts(str: *u8);

function inner(): i32 {
    puts("Testing");
    return 0;
}

function main(argc: i32, argv: **u8): i32 {
    puts("Hello, world!");
    return inner();
}
