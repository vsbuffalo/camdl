(** SHA-256, as specified in FIPS 180-4, for recording which bytes a compiled
    model was built from.

    A model can read data files at compile time: the knots of a file-backed
    forcing, the levels of a dimension read from a file, the vocabulary file
    given with [--quantities]. The compiler records the hash of each such file
    beside its path, in a forcing's {!Ir.data_source} and in the file list of
    the declared-model section, so that a compiled model, and every fit built
    on it, says which contents it came from. {!hex_of_file} is what those
    records hold: the lowercase hexadecimal digest of the file's raw bytes,
    the same string [shasum -a 256 FILE] prints and the same digest the Rust
    command computes with the [sha2] crate. The expander calls it through
    [Expander.file_sha256], which hashes each distinct file once.

    The module is written out rather than taken from a library because the
    compiler depends only on [yojson] and [fmt], and the standard library's
    [Digest] is MD5, a different hash from the SHA-256 the runtime uses for
    every other provenance record. The tests in [ocaml/test/test_sha256.ml]
    pin it to the published FIPS 180-4 test vectors and to the messages whose
    lengths fall on the padding boundaries; a Rust test,
    [rust/crates/cli/tests/forcing_provenance.rs], checks a compiled model's
    recorded hash against [sha2].

    Arithmetic is on OCaml's native 63-bit [int], masked to 32 bits after
    every addition and shift, rather than on boxed [Int32]. *)

let mask = 0xFFFFFFFF

let rotr x n = ((x lsr n) lor (x lsl (32 - n))) land mask

(* FIPS 180-4 §4.2.2: the first 32 bits of the fractional parts of the cube
   roots of the first 64 primes. *)
let k =
  [| 0x428a2f98; 0x71374491; 0xb5c0fbcf; 0xe9b5dba5;
     0x3956c25b; 0x59f111f1; 0x923f82a4; 0xab1c5ed5;
     0xd807aa98; 0x12835b01; 0x243185be; 0x550c7dc3;
     0x72be5d74; 0x80deb1fe; 0x9bdc06a7; 0xc19bf174;
     0xe49b69c1; 0xefbe4786; 0x0fc19dc6; 0x240ca1cc;
     0x2de92c6f; 0x4a7484aa; 0x5cb0a9dc; 0x76f988da;
     0x983e5152; 0xa831c66d; 0xb00327c8; 0xbf597fc7;
     0xc6e00bf3; 0xd5a79147; 0x06ca6351; 0x14292967;
     0x27b70a85; 0x2e1b2138; 0x4d2c6dfc; 0x53380d13;
     0x650a7354; 0x766a0abb; 0x81c2c92e; 0x92722c85;
     0xa2bfe8a1; 0xa81a664b; 0xc24b8b70; 0xc76c51a3;
     0xd192e819; 0xd6990624; 0xf40e3585; 0x106aa070;
     0x19a4c116; 0x1e376c08; 0x2748774c; 0x34b0bcb5;
     0x391c0cb3; 0x4ed8aa4a; 0x5b9cca4f; 0x682e6ff3;
     0x748f82ee; 0x78a5636f; 0x84c87814; 0x8cc70208;
     0x90befffa; 0xa4506ceb; 0xbef9a3f7; 0xc67178f2 |]

(* FIPS 180-4 §5.3.3: the first 32 bits of the fractional parts of the square
   roots of the first 8 primes. *)
let init_h =
  [| 0x6a09e667; 0xbb67ae85; 0x3c6ef372; 0xa54ff53a;
     0x510e527f; 0x9b05688c; 0x1f83d9ab; 0x5be0cd19 |]

(** The state of one hash computation part way through its input. It is
    mutable and single-use: {!update} feeds it bytes, and
    [finalize] pads the message and reads the digest out, after which the
    state must not be used again. *)
type t = {
  h        : int array;
      (** The eight 32-bit working hash values, each held in an [int]. *)
  block    : Bytes.t;
      (** The 64-byte block being filled; compressed into [h] when full. *)
  mutable fill : int;
      (** How many bytes of [block] are filled, from 0 to 63 between calls. *)
  mutable len  : int;
      (** How many message bytes have been absorbed, padding excluded. *)
}

let create () =
  { h = Array.copy init_h; block = Bytes.create 64; fill = 0; len = 0 }

(* Compress the 64 bytes in [t.block] into [t.h]. FIPS 180-4 §6.2.2. *)
let compress t =
  let w = Array.make 64 0 in
  for i = 0 to 15 do
    let b j = Char.code (Bytes.unsafe_get t.block ((i * 4) + j)) in
    w.(i) <- (b 0 lsl 24) lor (b 1 lsl 16) lor (b 2 lsl 8) lor b 3
  done;
  for i = 16 to 63 do
    let x = w.(i - 15) and y = w.(i - 2) in
    let s0 = rotr x 7 lxor rotr x 18 lxor (x lsr 3) in
    let s1 = rotr y 17 lxor rotr y 19 lxor (y lsr 10) in
    w.(i) <- (s1 + w.(i - 7) + s0 + w.(i - 16)) land mask
  done;
  let a = ref t.h.(0) and b = ref t.h.(1) and c = ref t.h.(2) and d = ref t.h.(3)
  and e = ref t.h.(4) and f = ref t.h.(5) and g = ref t.h.(6) and hh = ref t.h.(7) in
  for i = 0 to 63 do
    let s1 = rotr !e 6 lxor rotr !e 11 lxor rotr !e 25 in
    let ch = (!e land !f) lxor (lnot !e land mask land !g) in
    let t1 = (!hh + s1 + ch + k.(i) + w.(i)) land mask in
    let s0 = rotr !a 2 lxor rotr !a 13 lxor rotr !a 22 in
    let maj = (!a land !b) lxor (!a land !c) lxor (!b land !c) in
    let t2 = (s0 + maj) land mask in
    hh := !g; g := !f; f := !e;
    e := (!d + t1) land mask;
    d := !c; c := !b; b := !a;
    a := (t1 + t2) land mask
  done;
  t.h.(0) <- (t.h.(0) + !a) land mask;
  t.h.(1) <- (t.h.(1) + !b) land mask;
  t.h.(2) <- (t.h.(2) + !c) land mask;
  t.h.(3) <- (t.h.(3) + !d) land mask;
  t.h.(4) <- (t.h.(4) + !e) land mask;
  t.h.(5) <- (t.h.(5) + !f) land mask;
  t.h.(6) <- (t.h.(6) + !g) land mask;
  t.h.(7) <- (t.h.(7) + !hh) land mask

(* Feed bytes through the block buffer without counting them as message
   content — [finalize]'s padding goes through here, the message through
   [update]. Keeping the length accounting out of the absorb loop is what
   stops the padding from being counted in its own length field. *)
let absorb t buf off n =
  let off = ref off and remaining = ref n in
  while !remaining > 0 do
    let take = min !remaining (64 - t.fill) in
    Bytes.blit buf !off t.block t.fill take;
    t.fill <- t.fill + take;
    off := !off + take;
    remaining := !remaining - take;
    if t.fill = 64 then (compress t; t.fill <- 0)
  done

let update t buf off n =
  t.len <- t.len + n;
  absorb t buf off n

(* FIPS 180-4 §5.1.1: append 0x80, pad with zeros to 56 mod 64, then the
   message length in bits as a 64-bit big-endian integer. A 63-bit OCaml int
   holds the bit length of any file up to 2^60 bytes (1 EiB). *)
let finalize t =
  let bitlen = t.len * 8 in
  let one = Bytes.make 1 '\x80' in
  absorb t one 0 1;
  let zeros = Bytes.make 64 '\x00' in
  (* 56 - fill mod 64, taken positively: fill = 56 needs a full 64 zeros
     (the length field spills to the next block). *)
  let n_zeros = ((56 - t.fill) + 64) mod 64 in
  absorb t zeros 0 n_zeros;
  let tail = Bytes.create 8 in
  for i = 0 to 7 do
    Bytes.set tail i (Char.chr ((bitlen lsr ((7 - i) * 8)) land 0xFF))
  done;
  absorb t tail 0 8;
  let buf = Buffer.create 64 in
  Array.iter (fun x -> Buffer.add_string buf (Printf.sprintf "%08x" x)) t.h;
  Buffer.contents buf

(** Lowercase 64-char hex SHA-256 of a string. *)
let hex_of_string (s : string) : string =
  let t = create () in
  update t (Bytes.unsafe_of_string s) 0 (String.length s);
  finalize t

(** Lowercase 64-char hex SHA-256 of a file's raw bytes — byte-for-byte what
    `shasum -a 256 FILE` reports. Streams in 64 KiB chunks so a large data file
    is never held in memory. Raises [Sys_error] if the file cannot be opened;
    callers hash only files they have already successfully read. *)
let hex_of_file (path : string) : string =
  let ic = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr ic) (fun () ->
    let t = create () in
    let buf = Bytes.create 65536 in
    let rec go () =
      let n = input ic buf 0 (Bytes.length buf) in
      if n > 0 then (update t buf 0 n; go ())
    in
    go ();
    finalize t)
