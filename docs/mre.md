# Minimal reproducible examples (`camdl mre`)

When a fit misbehaves and you want the maintainer to reproduce it, bundle
everything the run needs into a single file:

```sh
camdl mre fit fit.toml
```

This writes `<fit>.mre.tar.gz` and prints the exact command to reproduce it.
Send that one tarball — it is self-contained.

## Why a bundle (and not just the fit.toml)

A fit depends on files that are **named nowhere in `fit.toml`**: the covariate,
contact-matrix, and population tables the model `read()`s at _compile_ time.
These are baked into the compiled model, so a hand-assembled bug report
routinely forgets them and the maintainer can't reproduce. `camdl mre` asks the
compiler what it read and captures those files automatically, alongside the data
and config.

## What's in the bundle

- the model (`[model] camdl`) **and every file it `read()`s at compile time**
- observed data and holdout (`[data]`)
- fixed parameters (`[fixed] from_file`) and synthetic truth
  (`[synthetic] true_params`)
- the `fit.toml` itself, plus a `manifest.toml` (per-file inventory with sha256)
  and an auto-generated `README.md`

Paths inside the bundle keep their layout, so it relocates anywhere.

## How the maintainer reproduces it

```sh
tar xzf fit.mre.tar.gz
cd fit && camdl fit run fit.toml
```

## Bundling a forward simulation

The same closure logic packs a `simulate` reproduction — pass the model and the
exact simulate flags you ran:

```sh
camdl mre simulate model.camdl --params p.toml --seed 1
```

The bundle captures the model and its compile-time `read()` files, plus any
`--table` / `--params` / `--param-vec` / `--draws PATH` / `--fit` inputs. The
recorded reproduce command keeps your run-shaping flags (seed, backend, `dt`,
scenarios) but drops output destinations like `--obs` / `-o` — the run always
writes its content-addressed store leaf, and the maintainer chooses where to
mirror it. The root is the current directory, so run `mre simulate` from where
your input paths are relative.

## Sharing data

By default the bundle **includes your observed data** and prints a banner naming
the files and row counts — sharing data is never silent. If the data is
sensitive, use `--no-data` for a **structure-only** bundle (column names, row
counts, time range — no values):

```sh
camdl mre fit fit.toml --no-data
```

Structure-only is enough for crashes and structural bugs. It is **not** enough
for "wrong number" bugs (a biased posterior, an off trajectory), whose symptom
depends on the actual data values.

## Verify it reproduces before sending

Run the maintainer's command yourself, from a fresh unpack:

```sh
camdl mre fit fit.toml -b /tmp/bug.tar.gz
mkdir /tmp/iso && tar xzf /tmp/bug.tar.gz -C /tmp/iso
cd /tmp/iso/<bundle> && camdl fit run fit.toml --seed 1
```

The fit's `run_id` (the `fit-<hash>` leaf under `results/fits/`) is a content
hash of `(model, data, config)`. If the unpacked bundle yields the **same**
`fit-<hash>` as your in-place run, the bundle is a faithful, complete
reproduction. Run from the project directory both times — the id folds in the
`fit.toml`-relative path string, so addressing the same fit via a different
relative path changes the id.

## Requirements and current limits

- **Relative, contained paths.** Every input must live under the root — the
  `fit.toml`'s directory for `mre fit`, the current directory for
  `mre simulate`. Absolute or `../`-escaping paths are non-portable and error
  (move the file under the project, or make its path relative).
- **Self-contained init.** Fits whose chains seed from an upstream artifact
  (`init = "survey_top_k"` / `"from_mle"` / `"from_posterior"` /
  `"from_params"`) aren't bundled yet and error with guidance — switch to
  `init = "lhs"` / `"single"` / `"from_prior"` to make a self-contained MRE.
