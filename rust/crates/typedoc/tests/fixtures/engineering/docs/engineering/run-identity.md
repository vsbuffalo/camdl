# Run identity

A run is named by the hash of what it was made from. The rest of this page
says which inputs those are.

A run starts from a [`Seed`](rust:demo::inputs::Seed), whose
[`value`](rust:demo::inputs::Seed::value) is made by
[`Seed::new`](rust:demo::inputs::Seed::new) and hashed by
[`run_id`](rust:demo::run_id), in [`demo::inputs`](rust:demo::inputs). The
compiler writes [`Ir.expr`](ocaml:Ir.expr), wraps it in
[`Ast.Located.t`](ocaml:Ast.Located.t) and calls through
[the `fn` field](ocaml:Ast.call.fn "a field").

<!-- decl: rust:demo::inputs::Seed -->

How a user sets the seed is in [the guide](../guide.md#seeds), and in its
[reference form][guide].

[guide]: ../guide.md

Written as code, `[not a link](rust:demo::Gone)` is left alone, and so is a
fenced block:

```text
[still not a link](rust:demo::Gone)
<!-- decl: rust:demo::Gone -->
```
