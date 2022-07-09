extern function fopen(name: *u8, mode: *u8): *void;
extern function fclose(fp: *void);
extern function fgetc(fp: *void): i32;
extern function fputc(c: i32, fp: *void);
extern function fprintf(fp: *void, fmt: *u8, prog: *u8, nl: i32);
extern function perror(msg: *u8);
extern var __stdoutp: *void;
extern var __stderrp: *void;

function main(argc: i32, argv: **u8): i32 {
    if argc != 2 {
        fprintf(__stderrp, "usage: %s FILE%c", argv[0], 10);
        return 1;
    }

    var f = fopen(argv[1], "r");
    if !f {
        perror("fopen");
        return 1;
    }

    var c: i32;
    while (c = fgetc(f)) != -1 {
        fputc(c, __stdoutp);
    }

    fclose(f);
    return 0;
}
