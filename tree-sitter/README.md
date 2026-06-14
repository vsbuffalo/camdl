# tree-sitter-camdl

Tree-sitter grammar for the [camdl](../) compartmental modelling DSL.

## Coverage

Last refreshed against the language spec on **2026-05-26**. The grammar parses
every model in `ocaml/golden/`, every test fixture under
`rust/crates/sim/tests/fixtures/`, and every worked example in
`docs/dev/proposals/fixtures/`. Sibling tooling for static document rendering
(the KDE/Pandoc syntax XML at `camdl-book/_extensions/camdl/camdl.xml`) covers
the same surface; the two are maintained in parallel.

Surface covered:

- All 22 top-level block kinds — `time_unit`, `description`, `origin`,
  `dimensions`, `compartments`, `parameters`, `tables`, `functions`, `forcing`,
  `transitions`, `observations`, `interventions`, `events`, `ode`, `output`,
  `simulate`, `init`, `timepoints`, `stratify`, `let`, `scenarios`, `balance`.
- `#[lineage]` transition attribute (and the lexer rule that requires `#[` to be
  one token with no intervening space).
- Calendar-time surface: `origin = date("YYYY-MM-DD")`, `date()` /
  `add_calendar_*` / `date_range` builtins, `instant` / `duration` parameter
  kinds.
- Multi-source stoichiometry (`A + B --> C`) and branching destinations
  (`--> { D1 : w1, D2 : w2 } @ rate`).
- Tier-3 dimension annotations on parameter declarations — both unit literals
  (`tau : positive 'ratio`) and bracket forms (`mu : positive
  [P*T^-1]`).
- Range literals inside list expressions (`on = [7 'days : 100 'days]`).
- Multi-name shared table declarations (`pop, init_sus : patch = read(...)`).
- Indexed scenario overrides (`R0[north] = 2.5` in a `set = { ... }` block).
- Prior assignment (`beta : rate in [0, 1] ~ log_normal(mu = 0, sigma = 1)`)
  with optional hierarchical pooling (`| age`).

If you hit a model the grammar doesn't parse, file an issue with the offending
fragment.

## Building

```bash
cd tree-sitter
npm install
npx tree-sitter generate   # produces src/parser.c
```

To test against a model file:

```bash
npx tree-sitter parse ../ocaml/golden/sir_basic.camdl
```

## Editor integration

### Neovim (nvim-treesitter)

Add to your Neovim config:

```lua
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.camdl = {
  install_info = {
    url = "https://github.com/your-org/camdl",   -- or local path
    files = { "tree-sitter/src/parser.c" },
    branch = "main",
    generate_requires_npm = false,
    requires_generate_from_grammar = false,
  },
  filetype = "camdl",
}

-- Associate .camdl files
vim.filetype.add({ extension = { camdl = "camdl" } })
```

Then copy the queries into the Neovim runtime:

```bash
mkdir -p ~/.config/nvim/queries/camdl
cp tree-sitter/queries/highlights.scm ~/.config/nvim/queries/camdl/
cp tree-sitter/queries/locals.scm     ~/.config/nvim/queries/camdl/
```

Run `:TSInstall camdl` (or `:TSInstallFromGrammar camdl` for a local path).

### Helix

Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "camdl"
scope = "source.camdl"
file-types = ["camdl"]
comment-token = "#"
indent = { tab-width = 2, unit = "  " }

[language.grammar]
name = "camdl"
source = { path = "/path/to/camdl/tree-sitter" }
```

Copy queries:

```bash
mkdir -p ~/.config/helix/runtime/queries/camdl
cp tree-sitter/queries/highlights.scm ~/.config/helix/runtime/queries/camdl/
cp tree-sitter/queries/locals.scm     ~/.config/helix/runtime/queries/camdl/
```

### Zed

Add to `~/.config/zed/settings.json` once the grammar is published to the
tree-sitter registry, or use the
[local grammar extension API](https://zed.dev/docs/extensions/languages).

### VS Code

VS Code consumes tree-sitter grammars via an extension. The simplest local path
is to wrap this grammar in a minimal extension using
[tree-sitter-vscode](https://github.com/tree-sitter/tree-sitter-vscode) or a
custom extension that registers `.camdl` as a language and points at
`tree-sitter/src/parser.c`. A maintained extension may be published to the
marketplace later; until then, the local-extension path is the supported route.

## Refreshing after a DSL change

When the DSL grammar changes (a new keyword, a new block, a new operator):

1. Edit `grammar.js` to add the production.
2. Update `queries/highlights.scm` for any new keyword or syntactic class.
3. Update `queries/locals.scm` if the change introduces a new scope or binding
   form.
4. Run `tree-sitter generate` to regenerate `src/parser.c`.
5. Verify by parsing the golden fixtures:
   ```bash
   for g in ../ocaml/golden/*.camdl; do tree-sitter parse --quiet "$g"; done
   ```
6. Update the "Last refreshed" date at the top of this README.

The OCaml `parser.mly` is the source of truth for syntax decisions; if in doubt,
mirror its productions.
