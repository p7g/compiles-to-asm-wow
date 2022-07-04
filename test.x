extern function fputs(string: *u8, file: *u64);
extern var __stdoutp: *u64;
extern var __stderrp: *u64;

function main(): i32 {
    fputs("Hello", __stdoutp);
    fputs("errororor", __stderrp);

    return 0;
}
