# A model's presentation as its own content

Status: superseded by
`docs/dev/proposals/2026-11-25-stored-models-and-invocations.md`

## The defect

A compiled model's run id is the digest of its model part (`camdl_store.rs:540`,
`CompiledParts::split`). The source file, `docs.json` (the `#'` documentation)
and `declared.json` are provenance files of that result, which are not hashed
and are kept from the first writer. Two model files that differ only in a `#'`
line compile to one compiled model, so every later reader is given the first
file's documentation: `stored_envelope` (`fit_store.rs:1576`) reads `docs.json`
back into the envelope that `fit summary`, `fit predict` and `compare` use.

Reproduced by
`fit_store::tests::a_reader_is_given_the_documentation_of_its_own_model_file`:
both files store as one compiled model (the first assertion passes), and the
reader of the second gets the first's documentation:

    left: Some("the first doc")
    right: Some("the second doc")

The fit problem has the same defect: it stores `model.camdl` as a provenance
file (`result_files/fit.rs:28`, `ProblemFile::ModelSource`), from whichever
invocation stored the problem first.

## The design

What a computation reads keeps its identity. What only presents the model
becomes one content artifact, keyed by its own bytes: the source file, the
documentation, the declared model, the rendered mathematics and the flow graph.
A computation does not depend on it, so editing a comment re-keys no fit. The
link from a computation to the presentation it was invoked with is a
per-invocation fact, so it lives in the invocation record, in the labels of the
compiled-model entry that `put` already registers in every invocation that
compiles a model.

```rust
/// How a model file is shown: everything one compile yields that no computation reads.
#[derive(store::Identity, Clone)]
#[identity(tag = "model_presentation", version = 1)]
pub(crate) struct PresentationIdentity {
    pub source: Hashed<SourceText>,              // source.camdl or source.ir.json
    pub docs: Hashed<DocsContent>,               // docs.json
    pub declared: Hashed<DeclaredContent>,       // declared.json
    pub render: Option<Hashed<RenderContent>>,   // model.render.json, .camdl sources only
    pub graph: Option<Hashed<GraphContent>>,     // model.graph.json, .camdl sources only
}
pub(crate) struct ModelPresentation;   // Artifact + Primary, KIND = "model_presentation", Atomic

/// The model's name, and the presentation this invocation compiled it from.
pub(crate) struct ModelLabels {
    pub name: String,
    pub presentation: Option<String>,   // run id (hex); None for a model that never had one
}
```

`CompiledParts`, before and after:

```rust
// before                                   // after
pub struct CompiledParts {                  pub struct CompiledParts {
    pub model: Hashed<CompiledModelContent>,    pub model: Hashed<CompiledModelContent>,
    pub quantities: Hashed<QuantitiesContent>,  pub quantities: Hashed<QuantitiesContent>,
    pub labels: ModelLabels,                    pub labels: ModelLabels,
    pub provenance: Provenance, // source,      pub presentation: PresentationIdentity,
        // docs, declared, description,         pub provenance: Provenance, // ir_version,
        // ir_version, validated_by, digests        // validated_by, digests
}                                           }
```

The `description` is written in the source and recorded today as a value of the
compiled model, so it has the same first-writer defect. It moves into the
presentation's `docs.json`. The one reader of `stored.recorded("description")`
(`fit_store.rs:1574`) reads it from there.

**Storing it generalizes `put`.** `Store::put` writes a content artifact with
one result file (`ContentArtifact::FILE`, `storage.rs:1433`). A presentation has
five, each in the format its readers use, and the result folder's manifest
already records each file's name and digest. `put` becomes the one write path
for content of one or more files: `ContentFiles::files(&identity)` returns each
file's name and bytes, `put` checks, as `finish` does, that the manifest's
digests equal `identity.held_content()`, and today's one-file content artifacts
are the one-file case. It takes no claim, because two writers of one content
write identical bytes. The claimed path (`begin`/`finish`) already writes
several files, but two fits of one model starting together would find the
presentation busy.

**Rendering is a shell-out.** camdlc is a separate OCaml program, so no library
call exists. Each command that compiles a `.camdl` runs
`camdlc render --format
json` and `--format graph` on the same source, with the
compile's settings (measured: under 10 ms each on `boarding_school_sir`). From
compiled IR no render is possible, and the command says so once, as a `RunNote`:
"model.ir.json is compiled IR, so its presentation has no rendered mathematics
or flow graph; pass the .camdl to store them." A render that fails on a `.camdl`
that compiled is a camdlc defect. The command warns once with camdlc's error and
stores the presentation without the render; the absence is in its identity, so
the next command renders again.

The render embeds the file stem (`"model": "model"`), a name inside content, so
two copies of one file under different names give two presentations. That costs
a few KB and re-keys no computation, so the field stays.

**A reader finds a result's presentation through its invocations.** One function
for every kind:

    model_presentation(store, handle, result) -> Option<Stored<ModelPresentation>>
      1. the invocation: @label → that one; an id or folder → the latest that lists `result`,
         with a notice when earlier invocations named a different presentation
      2. in it, the compiled_model entry whose run id is result's model dependency
      3. its labels.presentation → the stored presentation
      4. no such entry (a predict that compiled nothing) → repeat from the upstream
         result (the fit run), which did compile

`run_quantities` refuses an id whose invocations disagree, because quantities
change the values a prediction computes. A presentation changes nothing
computed, so the latest invocation answers instead.

`stored_envelope` takes the presentation and reads `docs.json` from it, and so
`fit summary`, `fit predict` and `compare` show the invoking file's
documentation. `camdl path` and `camdl cat` reach the files as
`model_presentation:render` (logical names `source`, `docs`, `declared`,
`render`, `graph`), resolved through the function above. camdl-scope calls
`camdl path <handle> model_presentation:render model_presentation:graph` instead
of looking in the run's folder.

**The result-file contract.** `model_presentation` gets a file enum in
`result_files/` and a contract section, which `make update-result-file-contract`
regenerates. `compiled_model` loses its provenance files and `fit_problem` loses
`model.camdl`, so both contract versions move. `CONTENT_CONTRACT` is one
constant for all three content kinds today, so bumping it also bumps
`quantities` and `data_table`. I would give each content kind its own version.

**No run id changes.** Provenance files are not hashed, so removing them moves
no compiled-model or fit-problem id. The red test's first assertion checks this
for the compiled model, and `only_what_the_engine_runs_keys_the_model_part`
checks it for the model part. For the fit problem it is inferred from
`ProblemFile::ModelSource` having `Role::Provenance`. A test that stores a
problem before and after the change would verify it, and I would add one. The
labels gain a field, and labels are not part of any id. No IR or golden changes.

**Rejected: documentation in the compiled model's identity.** Keying docs into
the compiled model would fix the reader, but a `#'` edit would then re-key every
simulation, problem and fit of the model and leave them orphaned. The proposal
rules this out ("What changes a result", run-store proposal, Part 2).

## Order

Each step is green on `make test`.

1. Delete the orphaned doc comment at `util.rs:1458`.
2. Land the red test together with the split: `PresentationIdentity`,
   `put_files`, the labels link and the reader.
3. Generate the render and the graph.
4. Update `path` and `cat`, the contract, the docs, and the note for scope.
