(* SHA-256 conformance for the compiler's hand-written digest
   (`ocaml/lib/compiler/sha256.ml`), which content-hashes the external data
   files a model reads at compile time.

   A hand-written hash is only trustworthy if it is pinned to the published
   vectors, so this suite is the whole justification for not taking a
   dependency. Two layers:

   1. **FIPS 180-4 published vectors** — the empty string, "abc", the
      56-byte and 112-byte multi-block examples, and the one-million-'a'
      stream (which exercises the chunked absorb path across ~15,625 blocks).

   2. **Block-boundary lengths** — 55/56/63/64/65/119/120 bytes. These are
      where a padding implementation goes wrong: 55 is the largest message
      whose 0x80 + length field still fit one block, 56 is the first that
      spills to a second block, 64 is an exact block, and 119/120 repeat the
      boundary one block later. Expectations are the reference digests
      (verified against both `shasum -a 256` and Python's hashlib).

   The OCaml↔Rust equivalence — that this digest agrees with `sha2::Sha256`,
   the hash every Rust-side provenance record already uses — is pinned
   separately over a real compiled model in
   `rust/crates/cli/tests/forcing_provenance.rs`. *)

let check_hex label expected input =
  Alcotest.(check string) label expected (Sha256.hex_of_string input)

(* FIPS 180-4, Appendix B / the NIST example set. *)
let fips_vectors () =
  check_hex "empty string"
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" "";
  check_hex "abc"
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" "abc";
  check_hex "two-block (56 bytes)"
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  check_hex "multi-block (112 bytes)"
    "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
    ("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
     ^ "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu")

(* One million 'a' — the NIST long-message vector. Absorbed in one call, so it
   also pins that a message far larger than the 64-byte block buffer streams
   correctly. *)
let fips_long_vector () =
  check_hex "one million 'a'"
    "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    (String.make 1_000_000 'a')

(* The lengths where padding logic breaks. *)
let block_boundaries () =
  List.iter (fun (n, expected) ->
    check_hex (Printf.sprintf "%d bytes of 'a'" n) expected (String.make n 'a'))
    [ 55,  "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318";
      56,  "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a";
      63,  "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34";
      64,  "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb";
      65,  "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0";
      119, "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb";
      120, "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c" ]

(* A digest that ignored some of its input would still pass a fixed-vector
   suite if the vectors happened to be short. Flipping one byte anywhere in a
   long message must move the digest. *)
let avalanche () =
  let base = String.make 1000 'x' in
  let flip i =
    let b = Bytes.of_string base in
    Bytes.set b i 'y';
    Sha256.hex_of_string (Bytes.to_string b)
  in
  let h = Sha256.hex_of_string base in
  List.iter (fun i ->
    Alcotest.(check bool) (Printf.sprintf "byte %d is read" i) true (flip i <> h))
    [ 0; 1; 63; 64; 500; 998; 999 ]

(* [hex_of_file] must agree with [hex_of_string] on the same bytes, including
   an empty file and one larger than the 64 KiB read chunk — the file path is
   what the compiler actually calls. Binary content (a NUL, a lone CR) pins
   that the read is binary, not text-mode line-translated. *)
let file_matches_string () =
  let with_temp content f =
    let path = Filename.temp_file "camdl_sha256" ".bin" in
    Fun.protect ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
      (fun () ->
         let oc = open_out_bin path in
         Fun.protect ~finally:(fun () -> close_out_noerr oc)
           (fun () -> output_string oc content);
         f path)
  in
  List.iter (fun (label, content) ->
    with_temp content (fun path ->
      Alcotest.(check string) (label ^ ": file digest = string digest")
        (Sha256.hex_of_string content) (Sha256.hex_of_file path)))
    [ "empty file",        "";
      "one line",          "t\tforce\n0\t1.0\n";
      "binary bytes",      "\x00\x01\r\n\xff\x80";
      "larger than chunk", String.make 200_000 'z' ]

let () =
  Alcotest.run "sha256"
    [ ("conformance",
       [ Alcotest.test_case "FIPS 180-4 vectors"    `Quick fips_vectors;
         Alcotest.test_case "FIPS 180-4 1M 'a'"     `Slow  fips_long_vector;
         Alcotest.test_case "block boundaries"      `Quick block_boundaries;
         Alcotest.test_case "every byte is read"    `Quick avalanche;
         Alcotest.test_case "hex_of_file = hex_of_string" `Quick file_matches_string ]) ]
