declare function puts(string: **u8);

function x(): bool {
    puts("in x");
    return true;
}

function y(): bool {
    puts("in y");
    return true;
}

function z(): bool {
    puts("in z");
    return true;
}

function main(argc: i32, argv: **u8): i32 {
    var cond = !(x() && y()) || z();
    if cond {
        puts("true");
    } else {
        puts("false");
    }

    return 0;
}
