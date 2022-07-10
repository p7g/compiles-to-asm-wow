extern type FILE;

extern function calloc(num: u64, size: u64): *void;
extern function fclose(f: *FILE);
extern function fflush(f: *FILE);
extern function fgetc(f: *FILE): i32;
extern function fopen(name: *u8, mode: *u8): *FILE;
extern function fprintf(f: *FILE, fmt: *u8, progname: *u8);
extern function fputc(c: i32, f: *FILE);
extern function free(ptr: *void);
extern function perror(msg: *u8);
extern function realloc(data: *void, new_size: u64): *void;

extern var __stderrp: *FILE;
extern var __stdoutp: *FILE;

function mul(a: u64, b: u64): u64 {
    var acc: u64 = 0;

    while b {
        acc += a;
        b += -1;
    }

    return acc;
}

type Tape = {
    pos: u64,
    len: u64,
    cap: u64,
    data: *u8,
};

function Tape_init(tape: *Tape) {
    tape.pos = 0;
    tape.len = 1;
    tape.cap = 1;
    tape.data = calloc(1, sizeof(type u8));
}

function Tape_get(tape: *Tape): u8 {
    return tape.data[tape.pos];
}

function Tape_inc(tape: *Tape, amount: i32) {
    tape.data[tape.pos] += amount;
}

function Tape_move(tape: *Tape, amount: i32) {
    tape.pos += amount;

    if tape.pos == tape.cap {
        var newcap = tape.cap + tape.cap;
        tape.data = realloc(tape.data, newcap);
        var i = tape.cap;
        while i != newcap {
            tape.data[i] = 0;
            i += 1;
        }
        tape.cap = newcap;
    }
}

type Op = {
    type_: u8,
    value: i32,
    loop_ops: *Op,
};

function Op_deinit(ops: *Op, len: u64) {
    var i: u64 = 0;
    while i != len {
        var op = &ops[i];
        i += 1;
        if op.type_ == 4 {
            Op_deinit(op.loop_ops, op.value);
            free(op.loop_ops);
        }
    }
}

type OpList = {
    len: u32,
    cap: u32,
    data: *Op,
};

function OpList_init(l: *OpList) {
    l.cap = l.len = 0;
    l.data = null;
}

function OpList_append(l: *OpList, op: *Op) {
    if l.len == l.cap {
        if l.cap {
            l.cap += l.cap;
        } else {
            l.cap = 4;
        }
        l.data = realloc(l.data, mul(l.cap, sizeof(type Op)));
    }

    var dest = &l.data[l.len];
    dest.type_ = op.type_;
    dest.value = op.value;
    dest.loop_ops = op.loop_ops;
    l.len += 1;
}

function parse(f: *FILE, ops: *OpList) {
    while true {
        var op: Op;
        var c = fgetc(f);

        if c == -1 || c == ']' {
            return;
        } else if c == '+' {
            op.type_ = 1;
            op.value = 1;
            OpList_append(ops, &op);
        } else if c == '-' {
            op.type_ = 1;
            op.value = -1;
            OpList_append(ops, &op);
        } else if c == '>' {
            op.type_ = 2;
            op.value = 1;
            OpList_append(ops, &op);
        } else if c == '<' {
            op.type_ = 2;
            op.value = -1;
            OpList_append(ops, &op);
        } else if c == '.' {
            op.type_ = 3;
            OpList_append(ops, &op);
        } else if c == '[' {
            op.type_ = 4;

            var inner_ops: OpList;
            OpList_init(&inner_ops);
            parse(f, &inner_ops);
            op.value = inner_ops.len;
            op.loop_ops = inner_ops.data;
            OpList_append(ops, &op);
        }
    }
}

function run(tape: *Tape, ops: *Op, num_ops: u64) {
    var i = 0;

    while i != num_ops {
        var op = &ops[i];
        i += 1;

        if op.type_ == 1 {
            Tape_inc(tape, op.value);
        } else if op.type_ == 2 {
            Tape_move(tape, op.value);
        } else if op.type_ == 3 {
            fputc(Tape_get(tape), __stdoutp);
            fflush(__stdoutp);
        } else if op.type_ == 4 {
            while Tape_get(tape) != 0 {
                run(tape, op.loop_ops, op.value);
            }
        }
    }
}

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

    var ops: OpList;
    OpList_init(&ops);
    parse(f, &ops);
    fclose(f);

    var tape: Tape;
    Tape_init(&tape);
    run(&tape, ops.data, ops.len);

    Op_deinit(ops.data, ops.len);

    return 0;
}
