declare function puts(str: *u8);

function main(argc: i32, argv: **u8): i32 {
    puts("Hello, world!");
    return 1;
}
