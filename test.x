extern type FILE;

extern function fopen(name: *u8, mode: *u8): *FILE;
extern function fclose(fp: *FILE);
extern function fgetc(fp: *FILE): i32;
extern function fputc(c: i32, fp: *FILE);
extern function fprintf(fp: *FILE, fmt: *u8, prog: *u8);
extern function perror(msg: *u8);
extern var __stdoutp: *FILE;
extern var __stderrp: *FILE;

function main(argc: i32, argv: **u8): i32 {
    if argc != 2 {
        fprintf(__stderrp, "usage: %s FILE\n", argv[0]);
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
