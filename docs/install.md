# Installing camdl

camdl is two programs that install together: `camdlc`, the compiler that turns a
`.camdl` file into a flat model description, and `camdl`, which simulates and
fits it. Both build from source. A separate viewer, camdl 'scope, shows a fit's
traces, posteriors and predictive checks in the browser; you will want it from
the first fit onwards.

## camdl

The `install.sh` script at the root of the repository installs both toolchains
camdl is written in (OCaml ≥ 5.2 through opam, and Rust through rustup), fetches
the OCaml packages, builds, and copies `camdl` and `camdlc` to `~/.local/bin`:

```sh
git clone https://github.com/vsbuffalo/camdl
cd camdl
./install.sh
```

It runs on Linux and macOS, is safe to re-run, and never calls `sudo` or a
system package manager, so it works on machines where you have no root access
(an HPC login node, say). Everything lands in directories you own: opam under
`~/.local`, the OCaml switch under `~/.opam`, Rust under `~/.cargo` and
`~/.rustup`. It does expect a few base tools to be present already — `make`,
`git`, `curl`, `tar`, `unzip` and working C and C++ compilers — and if one is
missing it prints the command that installs it and stops.

Make sure `~/.local/bin` is on your `PATH`, then check the install:

```sh
camdl --version
camdl docs          # the guides, offline and matched to this version
```

### Updating

Pull and rebuild:

```sh
git pull
make build && make install
```

Run `make install` after every rebuild. `camdl` checks that the `camdlc` on your
`PATH` is the one it was built with and refuses to run otherwise, since a
mismatched compiler could produce a model description the runtime reads
differently. `make install` copies to `~/.local/bin`; if you installed somewhere
else with `PREFIX`, run `make install INSTALL_DIR=<dir>/bin` instead.

### Options

Set these in the environment when you run the script, for example
`PREFIX=$HOME/opt/camdl ./install.sh`.

| Variable                     | Effect                                                                                                      |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `PREFIX=<dir>`               | install to `<dir>/bin` instead of `~/.local/bin`, for example to keep a branch build beside the default one |
| `OCAML_SWITCH_VERSION=5.2.1` | use a different OCaml version (default 5.2.0)                                                               |
| `NO_SANDBOX=1`               | initialise opam without its build sandbox, when `bubblewrap` is unavailable on Linux                        |

`NO_SANDBOX=1` turns off a protection for every package opam builds in that
switch, so use it only when installing `bubblewrap` is not possible.

To set up the toolchains by hand instead, follow the manual steps in the
repository's
[README](https://github.com/vsbuffalo/camdl#prerequisites-manual-install).

## camdl 'scope

camdl 'scope reads the fits camdl writes under `results/fits/` and draws them:
per-chain traces, marginal posteriors, pair plots, posterior predictive bands
against the data, and camdl's convergence diagnostics. It refreshes while a fit
is still running, so you can watch chains mix rather than wait for the end.

It is a Python package installed with [uv](https://docs.astral.sh/uv/). Its
install builds the browser interface from source, so Node.js (which provides
`npm`) must also be on your `PATH`:

```sh
uv tool install git+https://github.com/vsbuffalo/camdl-scope
```

It installs the `camdl-scope` command. Run it from the project directory — the
one holding `results/` — and open the address it prints:

```sh
camdl-scope --port 8800                  # http://127.0.0.1:8800
camdl-scope --port 8800 --host 0.0.0.0   # reachable from other devices on your network
```

The viewer only reads the store, so it is safe to leave running while fits write
to it.

## uv, for the guide pages

Some guide pages draw their figures with short Python scripts that declare their
own dependencies, and run them with `uv run`. To repeat those steps, install
[uv](https://docs.astral.sh/uv/getting-started/installation/); nothing else is
needed, since `uv run` fetches each script's packages on first use. camdl
'scope's install above already needs uv, so you may have it.
