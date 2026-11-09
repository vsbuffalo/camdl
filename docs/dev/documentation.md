# How the documentation is built

One set of markdown files in `docs/` serves three readers: someone running
`camdl docs` in a terminal, someone browsing the repository on GitHub, and
someone reading the website at <https://vsbuffalo.github.io/camdl/>. Wherever
the three need something different, the difference is made at build time from
the same file. There is never a second copy of a page to keep in step. The
documented examples are also checked against the code, so a page that drifts
from the software fails a test instead of misleading a reader.

This page describes how that works and what to do when adding to it. It is for
anyone editing the docs, human or agent.

## Each reader gets the same file, treated differently

| Reader   | Gets the text from                                                                                     | Sees a figure marker as                    |
| -------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| Terminal | the binary: `rust/crates/cli/src/docs.rs` embeds each topic with `include_str!` when camdl is compiled | one line linking to the figure on the site |
| GitHub   | the `.md` file, rendered by GitHub                                                                     | nothing (GitHub hides HTML comments)       |
| Website  | Quarto, rendering `docs/` with the filters in `docs/_web/`                                             | the drawing                                |

`camdl docs <topic>` therefore always matches the installed binary and needs no
network. The website follows the latest `main`, so it can be newer than an
installed binary.

## Which files reach which reader

**Embedded topics** are the `TOPICS` table in `rust/crates/cli/src/docs.rs`.
Each entry names its file with `include_str!` and, if the website publishes that
file, gives its path under `docs/` as `page`. A unit test holds `page` to the
embedded file and checks that the file is in the site's render list, so a
renamed or unpublished page fails the build rather than printing a dead link.

**Published pages** are the render list in `docs/_quarto.yml`, written once as
the YAML anchor `&published` and reused as `camdl-published`. A file not on the
list is not rendered. A link from a published page to an unpublished `.md` file,
such as anything under `docs/dev/`, is rewritten to that file on GitHub by
`_web/unpublished-links.lua`; otherwise Quarto would copy the raw markdown into
the site. The site's CI job fails if any `.md` file reaches `_site/`.

**`.qmd` files are website-only and are never embedded.** There are two kinds:
the landing page `docs/index.qmd`, and the tutorials under `docs/guide/`, whose
commands run at build time (see
[Executed tutorials](#executed-tutorials-show-real-output)). Quarto renders
`.md` and `.qmd` files directly; nothing converts one to the other. A page that
should also be readable in the terminal must stay `.md`.

## The website build

```sh
make docs-site            # render to docs/_site/
make docs-site-preview    # serve with live reload on port 4321 (DOCS_SITE_PORT)
```

Both targets build camdl first and put this checkout's `camdl` and `camdlc`
first on `PATH`, because the tutorials run them. The preview listens on all
interfaces, so it can be read from another device at `http://<host>:4321/`.

`docs/_quarto.yml` reads markdown with `commonmark_x`. Pandoc's default reader
misparses fences whose info string carries attributes after the language
(`` ```camdl preamble=sir ``, `` ```tsv data-example=hole ``), which the
language and data specs rely on. The filters run in this order, before Quarto's
own:

| Filter                       | What it does                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| `_web/unpublished-links.lua` | rewrites links to unpublished `.md` files to GitHub                                  |
| `_web/page-title.lua`        | takes the browser title from the page's first heading                                |
| `_web/spec-banner.lua`       | adds a "Reading this page" note to the top of the language spec, on the website only |
| `_web/camdl-run.lua`         | includes model files and runs commands in the tutorials                              |
| `_web/figures.lua`           | replaces figure markers with the drawings                                            |

Styling is `_web/theme.scss` (both themes) and `_web/dark.scss` (dark-mode
defaults). Code is highlighted with the camdl syntax definition
`_web/camdl.xml`.

`.github/workflows/docs-site.yml` renders the site for every change to `docs/`,
`ocaml/`, `rust/` or the `Makefile`, and deploys it to GitHub Pages from `main`.
A pull request is rendered but not deployed, so a broken page fails the check
before it lands.

## Figures are SVG on the website and text elsewhere

A figure is a hand-drawn SVG in `docs/figures/<name>.svg`, placed in a page by
an HTML comment on its own line. There are two forms.

**Adding a figure.** The figure appears on the website and nothing is removed:

```markdown
Read each transition as: "from [source] to [destination] at total rate
[propensity]."

<!-- figure: transition-anatomy -->
```

**Replacing a text diagram.** Put the diagram between the marker and a closing
marker. The website shows the drawing in its place; the terminal and GitHub keep
the text diagram:

````markdown
<!-- figure: date-axis -->

```
 origin
   │
───┼──────┬──────┬──────▶ t (days)
   0      31     59
```

<!-- /figure -->
````

Either form takes a caption, written in markdown, as the last thing in the
marker:

```markdown
<!-- figure: architecture, caption: The modeler is responsible for the model. -->
```

In the terminal, `camdl docs` prints each opening marker as a link to the
figure's anchor and drops the closing marker:

```
[Figure on the website: https://vsbuffalo.github.io/camdl/intro.html#fig-transition-anatomy]
```

Every figure on the website is a `<figure id="fig-<name>">`, so
`page.html#fig-<name>` links straight to it, from the terminal or anywhere else.

When a drawing replaces a text diagram, the two must say the same thing: the
text diagram is the only version terminal and GitHub readers see. Change both in
the same commit.

### What counts as a marker

The website (`_web/figures.lua`) and `camdl docs` (`marker` in
`rust/crates/cli/src/docs.rs`) read markers by the same rule, and a change to
one must be made to the other:

- A marker is a whole line: `<!--`, optional space, `figure:`, a name of
  letters, digits, `-` and `_`, then anything up to a `-->` that ends the line.
  `<!--figure: a-->` is a marker; `<!-- figure: a --> note` is not, and both
  readers leave it alone.
- The closing marker is `<!--`, `/figure` and `-->`, with optional space
  between.
- An opening marker may run over several lines (a long caption) until its `-->`.
- A line inside fenced code is never a marker, so a page can show the syntax, as
  this one does.
- Markers do not nest, and a closing marker must be in the same block as its
  opening one: a figure opened inside a list item must close inside it.

### Drawing a figure

The SVG carries class names, never colours; the colours come from variables in
`_web/theme.scss`, so one drawing serves light and dark mode. The filter inlines
the file rather than linking it as an image, which is what lets the page's
styles reach it. The classes are:

- **Text:** `t-label` (16px), `t-small` (14px), `t-cap` (small capitals for
  column headings), `t-b` (bold), and for code `t-code`, `t-code-label`,
  `t-code-cell` and `t-mono-label`.
- **Fill:** `fig-ink`, `fig-muted`, `fig-accent`, `fig-warm`, `fig-bad`,
  `fig-panel`, `fig-card`, `fig-code-bg`, `fig-accent-soft`, and `fig-ink-box`
  with `fig-inverse`/`fig-inverse-muted` for text on a dark box.
- **Stroke:** `s-line`, `s-muted`, `s-accent`, `s-warm`, and `s-arrow` with an
  arrowhead marker filled `fig-arrow`. Arrows are a solid colour, so a head
  drawn over its own line does not show the overlap.
- **Code tokens:** `tk-kw` (keyword), `tk-pp` (directive), `tk-cf` (control
  word), `tk-fu` (transition label), `tk-op` (operator), `tk-dv` (number),
  `tk-bu` (builtin) and `tk-dt` (type) on `<tspan>`s give code in a figure the
  colours the site's code blocks use. The colours are copied from the
  `github`/`github-dark` highlight styles in `_quarto.yml`; a change of
  highlight style must update them in `theme.scss` too.

Draw at 640 units wide. The figure scales to the text column, so at that width
16px text stays readable on a tablet. Give the SVG a `<title id="t">` and a
`<desc id="d">` that says in words what the drawing shows; the filter makes the
ids unique per figure.

A marker naming a missing SVG fails the render, and so does a closing marker
with no figure open.

## Executed tutorials show real output

The `.qmd` tutorials use two code-block forms from `_web/camdl-run.lua`, both
resolved relative to the page:

````markdown
```{.camdl include="dimensions/seir_erlang_via.camdl" lines="15-19"}
```

```{.camdl-run}
camdl inspect dimensions/seir_erlang_via.camdl --transitions
```
````

The first shows the model file itself, so the model on the page is the one the
commands run. The second runs the command when the site is built and shows its
output. A non-zero exit fails the render; a block written to show an error is
marked `expect="fail"`, and then a zero exit fails it instead. A block runs
under `bash -e -o pipefail`, so a failure anywhere in a multi-line block or a
pipeline counts, and every command's error output is shown. A command that
cannot run at all (not found, exit 127) fails the render even under
`expect="fail"`, because that is a broken build, not the error the block means
to show. An `include` whose `lines` range runs past the end of the file fails
too, since the excerpt no longer matches what it was written against. Model
files live next to the page, in `docs/guide/<page>/`.

A tutorial whose commands write files, such as a fit, declares a scratch
directory in its front matter:

```yaml
camdl-run-scratch: first-fit
```

Every block on that page then runs in a fresh copy of `docs/guide/first-fit/`,
made once per render and deleted afterwards. The commands read as a user types
them inside their own project (`camdl fit run fit.toml`), the checkout never
collects a `results/` directory, and a fit `--label`, which can never be reused,
is new on every render. `include` paths stay relative to the page.

A block that draws a plot names the SVG it writes with `figure="name.svg"` (and
an optional `caption="..."`), and the filter inlines that file under the block's
output; a missing file fails the render. The plots are `uv run` scripts beside
the page, with their dependencies declared inline, so a site build needs `uv` on
PATH (the CI workflow installs it). A script that draws in `currentColor`, as
`guide/first-fit/plot_prior.py` does, follows the light and dark themes.

Screenshots of camdl 'scope cannot be made at render time, since they need a
running viewer. They are committed images, regenerated by a named target
(`make docs-first-fit-screenshots`) that reruns the page's fits with the same
seeds and drives headless Chrome. Rerun it when the page's fits or camdl 'scope
change.

Because the output is regenerated on every build, a tutorial cannot show output
the current code no longer produces. The prose around it can still drift: when a
command's output changes, reread the paragraph that describes it.

## Doc tests check the examples against the code

| Target                                 | What it checks                                                                                                                          | Docs                             | CI workflow                                      |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------ |
| `make test-docs`                       | every `` ```camdl `` block compiles, or is classified as a fragment (`camdlc doctest --gate`)                                           | `DOCTEST_DOCS` in the `Makefile` | `doctest.yml`                                    |
| `make test-cli-docs`                   | every documented `camdl …` command parses against the real CLI                                                                          | `CLI_DOCS` in the `Makefile`     | `cli-docs.yml`                                   |
| `make test-data-spec`                  | every observation-file example loads through the reader `fit run` uses                                                                  | `camdl-data-spec.md`             | `data-spec.yml`                                  |
| `cargo test -p cli --bin camdl docs::` | the embedded topics exist and have unique names; each topic's page is the embedded file and is published; figure markers print as links | every embedded topic             | `docs-site.yml` (and `ci.yml` when code changes) |
| `make docs-site`                       | every tutorial command exits as marked, and every figure marker names an SVG                                                            | the published pages              | `docs-site.yml`                                  |

A `` ```camdl `` example that only makes sense after declarations elsewhere in
the section borrows them from a hidden preamble, itself an HTML comment:

````markdown
<!-- camdl-doctest-preamble: sir
compartments { S, I, R }
parameters { gamma : rate }
-->

```camdl preamble=sir
transitions { recovery : I --> R @ gamma * I }
```
````

A block with `preamble=` is asserted to compile, so any error in it is a failure
rather than a skip. The closing `-->` must be on a line of its own, because
camdl's own `-->` arrow would otherwise end the comment early. Data a block
reads can travel the same way, as `<!-- camdl-doctest-data: data/pop.tsv`. The
full rules are in the header comment of `ocaml/lib/compiler/doctest.ml`.

When adding a page with camdl examples or commands, add it to `DOCTEST_DOCS` or
`CLI_DOCS` and to the matching workflow's `paths:`, so a change to the page
alone reruns its check.

## Formatting

Format any `.md` file you touch with `dprint fmt` (`mdfmt`) as the last step,
with one exception: never format `docs/camdl-language-spec.md`. Its doc-test
preambles do not survive dprint's reflow, and passing doc tests turn into
failures. Edit it by hand. `docs/camdl-data-spec.md` is safe to format, because
its example markers are in the fence's info string, which dprint leaves alone.

## Recipes

**Add a figure to a page.** Draw `docs/figures/<name>.svg` with the classes
above, put `<!-- figure: <name> -->` where it belongs, and run `make docs-site`.
If the page is embedded, `camdl docs <topic>` now prints the link in its place.

**Replace a text diagram with a drawing.** Draw the SVG from the diagram, then
wrap the diagram's code block in `<!-- figure: <name> -->` and
`<!-- /figure -->`. Check both the website and `camdl docs <topic>`: the first
should show the drawing, the second the text diagram under a link.

**Add a page to the website.** Add it to the `&published` list and to a sidebar
section in `docs/_quarto.yml`. If it has camdl examples or commands, put it
under a doc test as described above.

**Add a page to `camdl docs`.** Add a `Topic` to `TOPICS` in
`rust/crates/cli/src/docs.rs`, with `page` set if the website publishes it, and
run `cargo test -p cli --bin camdl docs::`.

**Add a tutorial.** Write `docs/guide/<page>.qmd`, keep its models in
`docs/guide/<page>/`, use `{.camdl include=…}` and `{.camdl-run}` for every
model and command, and publish it as above.
