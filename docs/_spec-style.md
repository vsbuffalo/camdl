A specification (`docs/camdl-language-spec.md`, `docs/camdl-run-spec.md`,
`docs/camdl-data-spec.md`, `docs/camdl-inference-spec.md`,
`docs/compartmental-ir-spec.md`) states what the language and its files mean.
Its readers look things up: a modeler checking what a construct accepts, an
agent writing a model, a developer checking that the code does what the
specification says. The prose rules of section 2 apply in full. The register
differs from a guide: a specification states rules in the present tense ("is
refused with E272"), gives the minimal example that shows each rule, and leaves
motivation and worked examples to the guide pages it links to.

### What a section contains

A section describes one construct, in this order:

1. **Purpose.** One or two sentences saying what the construct is for, with a
   link to the guide page that explains why a modeler would use it.
2. **Syntax.** Every accepted form, each with a minimal example.
3. **Meaning.** What each form does when the model is compiled and when it is
   run, including units and time conventions.
4. **What is refused.** Every error and warning code the construct can raise,
   with the condition that raises it.
5. **Related sections**, by number.

**Lists.** Use a short list where the material is parallel: the two sides of a
pair, such as simulation and inference, or a few cases of one rule. Keep it to
two to four items, each a sentence or two. Reasoning that connects one point to
the next stays in paragraphs.

### What a specification leaves out

- **History.** A removed form appears only as the error it now raises and the
  form that replaces it: "A bare `every` field is refused with E272; write
  `emit_schedule = every N 'unit`." Never "the pre-gh#171 form".
- **Promises.** A feature that does not exist is stated as unsupported. "Once X
  is implemented" goes stale without anyone noticing.
- **Internal names.** Function names, code paths and lists of the algorithms
  that consume a construct describe the implementation, not the language. Name
  only what a user can write, run or observe. A comparison with another
  project's design is different: it stays, cited, where it shows that camdl
  follows established practice (section 2).
- **Rules owned by another section.** Link to the section that states a rule
  rather than restating it, so the two copies cannot disagree.

Examples in a specification are executed by the doc-test harness, and
`docs/camdl-language-spec.md` is never run through `dprint`, which would break
the doc-test markers. Edit it by hand, and check every example compiles.

### Example

**Instead of:**

> ### 12.4 Sampling vs Scoring
>
> The `observations {}` block is evaluated at runtime in both directions.
>
> - **Forward simulation** (`camdl simulate`): the runtime evaluates each
>   stream's projection on its `emit_schedule` and **samples** from the declared
>   likelihood family to produce synthetic observations. …
> - **Inference** (`camdl fit` and friends): the runtime **scores** observed
>   data against the same likelihood family, producing log p(y | θ). PGAS, IF2,
>   particle filtering, and PMMH all consume the `observations {}` declarations
>   via the compiled `dmeasure` / `rmeasure` paths. …
>
> A bare `every`/`at` field at the top of an observation block is the removed
> pre-gh#171 form and is rejected with **E272**. Monthly incidence can be
> obtained natively by setting `emit_schedule = every 30 'days` (or
> `every 1 'months` once time-unit arithmetic is implemented).

**Write:**

> ### 12.4 Simulation and inference
>
> Each observation stream is read by both simulation and inference.
>
> - **Simulation** (`camdl simulate`). At each time in the stream's
>   `emit_schedule`, the runtime evaluates the stream's projection and draws an
>   observation from the declared distribution at that value. The draws are
>   written only when `--obs`, `--obs-dir` or `--obs-only` is passed (§21). To
>   simulate on the times of a data set instead, pass `--design-from fit.toml`
>   (run spec §4.6).
> - **Inference** (`camdl fit`). At each time in the data, the runtime evaluates
>   the same projection and adds the log density of the observed value under the
>   same distribution to the log-likelihood, log p(y | θ). The data's times
>   replace the `emit_schedule` (§12.1.1).
>
> Because only simulation uses the emission times, they are set by
> `emit_schedule`, in one of two forms:
>
> ```
> emit_schedule = every 7 'days                       # a recurring cadence
> emit_schedule = at [7 'days, 14 'days, 28 'days]    # listed times, with the unit on each entry
> ```
>
> A bare `every` or `at` field is refused with E272.

The first version names internal code paths and inference algorithms, uses a
metaphor ("in both directions"), records history, and makes a promise about a
unit the language already supports. The second keeps its two-sided structure,
states what each command does with the block and where the times come from, and
gives each form of the schedule.
