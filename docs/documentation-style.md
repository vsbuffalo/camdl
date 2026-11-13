# Writing the camdl documentation

_An editorial style guide for human and AI authors_

Status: adopted, November 2026

camdl documentation should read like a good scientific computing book: precise
enough to trust, clear enough to learn from, and pleasant enough to keep
reading. It should explain both how to use the software and how to think about
the models being built with it. Readers should finish a page better equipped to
make scientific decisions, not merely able to reproduce its commands.

Our model is an experienced computational scientist writing for a capable
colleague. The prose should be conversational without becoming chatty, rigorous
without sounding like a specification, and confident without overselling what a
model or diagnostic can establish.

The most important editorial rule is this: **put the central scientific ideas in
the main text and use carefully named, color-coded collapsible boxes for the
material a reader may reasonably postpone.** A likelihood equation may be
essential. The derivation of a particle filter estimator usually is not. Hiding
both is an error, as is putting both in the opening paragraph.

## 1. Write for the reader, not the implementation

The typical reader can program and has some quantitative or scientific
background. They may know SIR models, probability, and Bayesian inference; they
do not necessarily know camdl's syntax, fit configuration, run store, or
particle algorithms. Never explain familiar concepts automatically, and never
assume familiarity with a camdl-specific concept simply because its name sounds
technical.

Before drafting, write down three things for yourself:

1. **Reader:** Who is this page for, and what do they already know?
2. **Promise:** What will they be able to do or understand after reading it?
3. **Scope:** What belongs on this page, and what should be linked elsewhere?

These notes do not usually belong in the published page. They are an editorial
check against writing everything the software happens to support.

An installation page should help readers install and verify camdl. Its opening
paragraph is not the place to explain OCaml switches or the intermediate
representation. A first-fit tutorial should get a reader through a meaningful
fit and help them interpret it. It need not also teach every convergence
diagnostic, every observation distribution, and every model-comparison score.

## 2. A natural, disciplined voice

Write in the voice of an experienced scientist who enjoys explaining the
subject. Use _we_ while working through an example and _you_ for actions readers
perform. _We_ means the author and the reader working through the example
together, never the authors' own behind-the-scenes runs: a claim that rests on
runs the page does not show either gets its evidence on the page, or on a page
it links to, or is cut. Prefer strong verbs and concrete subjects. Let
paragraphs develop an idea, rather than turning every point into a list.

**Instead of:**

> The `observations` block provides the functionality required to map the
> underlying latent state to the observed data, enabling likelihood computation.

**Write:**

> The model describes infections, but our data contain reported cases. The
> `observations` block connects the two: it says what the model predicts we
> would measure, and how much the measurements can vary around that prediction.

The second version gives a reason for the feature before naming its
responsibility. It also introduces the statistical distinction readers will need
later.

**Instead of:**

> In this section, we will explore how to configure Bayesian inference using
> camdl's flexible and powerful inference framework.

**Write:**

> We'll fit the model to the daily counts, then check whether the resulting
> posterior makes reasonable predictions.

Avoid puffery, stock transitions, long strings of parenthetical definitions, and
claims that something is “important” without saying why. But do not edit all
personality out of the writing. An occasional concrete comparison, well-chosen
observation, or candid sentence about a method's limitations can make technical
prose memorable.

Prefer US English for consistency. Use sentence-case headings. Write `camdl` and
`camdlc` exactly as the programs are named. Use `camdl ’scope` only where that
is the project's intended product name; use `camdl-scope` for the executable.

## 3. Choose a document's primary purpose

Different pages require different writing.

- **Tutorials** teach through a complete, deliberately sequenced experience.
  Prefer one realistic example and introduce ideas when the reader needs them.
- **How-to guides** solve a particular problem. Start with the task and the
  shortest reliable path through it; move alternatives and caveats to the right
  point in the procedure.
- **Conceptual explanations** develop an idea: what a stochastic likelihood
  means, why observability differs from identifiability, or how backends relate
  to model assumptions. Equations are welcome when they clarify the idea.
- **Reference pages** define syntax, commands, options, defaults, constraints,
  and behavior comprehensively. Their prose should be economical, and their
  structure should support lookup.

A page may contain some of each, but should have one primary job. If a tutorial
has become a reference manual, split it. If a reference page contains a
compelling worked example, consider giving that example its own guide.

A useful progression for a tutorial is **problem → model → command → result →
interpretation → next decision**. Not every page needs these exact headings.
They describe the intellectual sequence, not a template to reproduce
mechanically.

## 4. Keep essential mathematics in the main text

Do not treat equations as advanced decoration to be tucked away. Scientific
readers often understand an equation faster than a paragraph of approximations.
Put an equation in the main body when it names the object being computed,
exposes a modeling assumption, or supplies intuition needed to interpret the
next step. Explain each term in prose, including what varies, what is
conditioned on, and any important units.

**Worked example: explaining the likelihood of a stochastic model**

A stochastic epidemic model predicts a distribution over epidemic histories, not
a single trajectory. Let $x_{0:T}$ denote the unobserved epidemic history,
$y_{1:T}$ the observations, and $\theta$ the parameters. The likelihood of the
observed data is

$$
p(y_{1:T}\mid\theta)
=\int p(x_{0:T}\mid\theta)\,
       p(y_{1:T}\mid x_{0:T},\theta)\,dx_{0:T}.
$$

This separates two sources of variation. The **process model**,
$p(x_{0:T}\mid\theta)$, describes which epidemics can occur under the
parameters. The **observation model**, $p(y_{1:T}\mid x_{0:T},\theta)$,
describes what we are likely to record if one of those epidemics occurs. The
likelihood averages the observation model over epidemic histories we did not
observe.

If observations are conditionally independent given the latent epidemic and
parameters, the second term becomes

$$
p(y_{1:T}\mid x_{0:T},\theta)
=\prod_{t=1}^{T}p(y_t\mid x_{0:T},\theta).
$$

For example, suppose $z_t(x_{0:T})$ is the number of infections during week $t$,
$\rho$ is the reporting fraction, and weekly cases follow a negative-binomial
observation model:

$$
y_t\mid x_{0:T},\theta
\sim\operatorname{NegBin}\!\left(\text{mean}=\rho z_t(x_{0:T}),\,\text{dispersion}=k\right).
$$

This distinction is essential to understanding what camdl fits. Changing the
transmission model changes the distribution of epidemic histories. Changing the
reporting fraction or dispersion changes how histories become data. Both can
change the likelihood. The integral is a sum for discrete latent histories; for
deterministic dynamics it reduces to evaluating the observation likelihood along
the model's trajectory.

The equations belong here, in the main text. A novice to particle filtering can
understand this decomposition before learning how camdl approximates the
integral. Reserve the estimation algorithm for an optional orange box
immediately afterward.

Do not confuse the marginal likelihood $p(y\mid\theta)$, which integrates out
latent histories, with Bayesian model evidence $p(y)$, which also integrates
over a prior on $\theta$. Use names and conditioning arguments carefully.

## 5. Collapsible boxes: depth without interruption

Use collapsible boxes to let readers choose how deeply to pursue an idea without
interrupting the main narrative. A reader should be able to ignore every closed
box and still complete the main task and understand the scientific claims being
made. Another reader should be able to expand the boxes and discover
substantial, worthwhile explanations—not leftovers or trivia.

**Use a box when:** a derivation, algorithmic detail, modeling judgment,
troubleshooting alternative, or practical aside is useful but not required
immediately.

**Do not use a box when:** it contains an essential command, a central
definition or equation, a crucial modeling assumption, the interpretation of the
result just shown, or a warning that changes whether the reader should trust the
result. Those belong in the main text.

### Color is a promise about the kind of depth

camdl's documentation has five kinds of box, each with its own color:

| Color      | Kind                                  | Typical label            | What readers should expect                                                                                             |
| ---------- | ------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Green**  | Practical guidance                    | **In practice: ...**     | Workflow tips, useful alternatives, performance choices, or troubleshooting that help with a task                      |
| **Yellow** | Modeling judgment                     | **Modeling choice: ...** | Assumptions, prior selection, identifiability, scientific tradeoffs, and decisions that do not have a universal answer |
| **Orange** | Advanced mechanics                    | **Under the hood: ...**  | Particle filters, sampler mechanics, compiler behavior, numerical approximations, and derivations                      |
| **Red**    | Critical warning, **not collapsible** | **Warning: ...**         | Data loss, invalid inference, a dangerous default, or a condition under which a result should not be trusted           |
| **Slate**  | Notes for AI agents                   | **For AI agents: ...**   | What an agent running camdl for someone should do differently from a human at this step (section 5.1)                  |

The light-theme tokens are green `#2A7A51` on `#F0FDF4`, yellow `#946200` on
`#FFFBEB`, orange `#B45309` on `#FFF7ED`, red `#B42318` on `#FEF3F2`, and slate
`#475569` on `#F1F5F9`; the dark theme uses lighter accents on dark tints of the
same hues (`docs/_web/theme.scss`). Every title and body color meets the WCAG AA
contrast ratio for body text, 4.5:1, against its panel in both themes, and any
change to the palette should be checked against that bar. The boxes use a
subdued background, a visible left border, and a clear disclosure chevron. Color
alone must never convey meaning. The label—_In practice_, _Modeling choice_, or
_Under the hood_—must remain visible in text and accessible to screen readers.

A box's title is an editorial sentence, not a category tag. **Good:** “Modeling
choice: How should we choose a prior for $R_0$?” **Weak:** “Note,” “Details,”
“Advanced,” “Important information,” or “Reading this output.”

Default states: yellow and orange are closed unless a page expressly depends on
their contents; green may start closed when longer than a few sentences, or
remain as a short visible aside if immediately useful. A short, crucial caveat
should generally be plain prose, not a box. Avoid nested collapsibles and long
runs of boxes with nothing between them.

### Writing a box

A box is a fenced div whose class names its kind and whose `title` is the
editorial sentence:

```markdown
::: {.practice title="Checking a fit without reading its full log"}

Start with `camdl fit summary @r0`. ...

:::

::: {.modeling title="How should we choose a prior for $R_0$?"}

A prior should express what is plausible before using the observations being
fitted. ...

:::

::: {.internals title="How a particle filter estimates the likelihood"}

In a stochastic model the likelihood integrates over possible epidemic
histories. ...

:::

::: {.warning title="Do not report an unconverged fit as a posterior estimate"}

If chains are stuck or have substantially different distributions, ...

:::
```

The site's filter (`docs/_web/boxes.lua`) writes the label at the start of the
title (_In practice:_, _Modeling choice:_, _Under the hood:_, _Warning:_ or _For
AI agents:_), unless the title already begins with it, so the kind of box is in
its text and not only its color. Green, yellow, orange and slate boxes are
collapsed by default; `collapse="false"` shows one open, and the reader can
still close it. A red warning is never collapsible, and a `collapse` attribute
on it is ignored. A box without a title, or a box inside another box, fails the
render. Leave a blank line after the opening `:::` line and before the closing
one: without it, `dprint` reflows the fences into the box's text, and the box
stops being a box (the filter fails the render if it finds one). Write a box
with these classes, never with hand-coded colors or a raw Quarto callout.

`camdl docs` prints the same pages in a terminal, where nothing collapses: each
box's opening fence becomes its titled label in bold and the closing fence is
dropped (`rust/crates/cli/src/docs.rs`). An agent reading a page there therefore
sees every box, the agent notes among them.

The examples below are live boxes.

### Example: a yellow modeling box

::: {.modeling title="How should we choose a prior for $R_0$?"}

A prior should express what is plausible before using the observations being
fitted. For a school influenza outbreak, one might start with previous estimates
from comparable settings, then examine the epidemics implied by those values. A
log-normal prior is convenient because it keeps $R_0$ positive, but its
parameters should be chosen by their consequences, not because they look diffuse
on the log scale.

For example, a log-normal prior with log mean 1.0 and log standard deviation 0.5
has median $e^1\approx2.7$, with roughly 90% between 1.2 and 6.2. A prior
predictive check shows what outbreaks that range produces.

The outbreak's existence is itself information. If the same outbreak data are
used to set a prior that rules out non-outbreaks and then used again for
fitting, explain the choice and whether ascertainment has been accounted for. Do
not present a convenient parameterization as independent prior evidence.

:::

This box helps a reader who wants to understand the choice without forcing
everyone to study prior elicitation before running the tutorial.

### Example: an orange algorithmic box

::: {.internals title="How a particle filter estimates the likelihood"}

In a stochastic model the likelihood integrates over possible epidemic
histories, an integral generally too large to calculate directly. A particle
filter approximates it by propagating an ensemble of simulated histories,
scoring them against each new observation, and resampling histories according to
those weights.

For a standard bootstrap particle filter, the estimated likelihood can be
expressed as a product of average incremental weights:

$$\widehat p(y_{1:T}\mid\theta)=\prod_{t=1}^T\left(\frac{1}{N}\sum_{i=1}^N w_t^{(i)}\right).$$

Under standard conditions this is an unbiased estimator of the likelihood,
although its logarithm is not an unbiased estimator of the log likelihood. PMMH
can use that likelihood estimator in an MCMC acceptance ratio and still target
the correct posterior under its usual assumptions. In practice, the estimator's
variance affects how readily proposals are accepted and how well chains mix.

Link here to the full particle-filter and PMMH explanation rather than
reproducing the entire algorithm in every fitting tutorial.

:::

The main body should already have explained _what likelihood is being
estimated_. The box explains _how_ the estimate is obtained.

### Example: a green practical box

::: {.practice title="Checking a fit without reading its full log"}

Start with `camdl fit summary @r0`. Look first at chain agreement and effective
sample size, then examine traces and posterior predictive checks in camdl
’scope. A clean convergence summary is a reason to continue checking the model,
not a substitute for checking it.

:::

### Critical warnings are visible

::: {.warning title="Do not report an unconverged fit as a posterior estimate"}

If chains are stuck or have substantially different distributions, the pooled
numerical summary may not represent the intended posterior.

:::

This warning belongs beside the result, in the main reading path—not in a closed
orange box.

### 5.1 Notes for AI agents

Many readers of a camdl page are AI agents running camdl on someone's behalf,
and they read it through `camdl docs`. Most of what a page says applies to them
as it does to a person. A slate **For AI agents** box holds the rest: what an
agent should do differently at this step, because it is unreliable at something
a person does well or because the decision belongs to the person it works for.

Write one only when an agent would get the step wrong without it. Three kinds
qualify:

- **A judgment an agent makes badly.** Reading the geometry of a scatter plot, a
  posterior ribbon or a pair plot is the standing example: say what to rely on
  instead (the numerical table beside the figure) and what to hand to the person
  (the figure's path).
- **A decision that belongs to the person.** Choosing a prior, loosening a
  convergence threshold, reaching for an escape-hatch flag, publishing a result.
  Name the decision and say to stop and ask before making it.
- **An action an agent is tempted to take.** Hand-rolling a pipeline that a
  camdl verb already owns, or tuning a fit until it matches a published number.
  Name the verb or the action to take instead.

The box says what to do, in one to three sentences, and links to the longer
reasoning in [`camdl docs agents`](agents.md) rather than repeating it. The
title is the instruction itself: "Ask before choosing a prior shape", not
"Priors". A rule an agent must follow on every page belongs in `agents.md`,
once, and not in a box on each page.

Never use the box to hide something from human readers. A caveat that changes
whether a result can be trusted is a red warning or plain text, whoever is
reading. If a person would need the point too, it belongs in the main text, and
the box is the wrong place for it.

::: {.agents title="Surface the survey plot rather than reading its geometry"}

Rank points by the log-likelihoods in the survey's `landscape.tsv`, not by your
reading of the scatter in its HTML page. Give the person the HTML path and your
one-line reading, and let them confirm before you start a fit from it.

:::

## 6. Show readers what matters in code and output

Prefer examples from genuine scientific workflows. Introduce the question before
showing the implementation, and interpret the result afterward. Do not explain
every line of code if its purpose is apparent from a short introduction.

Use `bash` for shell commands, `toml` for fit files, and `text` for terminal
output. Never label a shell snippet `css`, `sql`, or `go` merely to get
highlighting. Use executable commands and verified output. Put placeholders in
obvious notation, not in a form that looks safe to paste unmodified.

Large terminal transcripts are almost never useful in the narrative. Select the
lines that support the point being made, mark omissions with `...`, and provide
a link to the full transcript when appropriate. Keep command output faithful:
never silently change numbers, warnings, file paths, or diagnostic decisions to
make an example look tidy.

**Too much:** hundreds of lines of chain progress, particle-filter diagnostics,
and storage paths before telling the reader what happened.

**Better:** run the command on the page and show only the lines the paragraph
discusses. On this site, examples run when the page is rendered, so trim the
output with an option on the `{.camdl-run}` block rather than copying it into a
`text` block. A hand-copied transcript drifts from what camdl prints, and
nothing tells the reader when it has.

````markdown
```{.camdl-run show-match="NOT converged|max R̂|acceptance"}
camdl fit summary @r0
```
````

`show-match` keeps the lines containing any of the `|`-separated texts
(`show-context="2"` adds two lines either side of each), and
`show-lines="1-6,40-52"` keeps lines by number instead. Either way the block
puts a `...` line wherever it cut, still runs the whole command, and still
checks its exit status. A range past the end of the output, or a text no line
contains, fails the render, so an excerpt cannot quietly drift onto other lines
when camdl's output changes. For a fit whose chains were stuck, the excerpt
reads something like this (shown as text here, because this page does not run a
fit):

```text
✗ NOT converged — the chains are in different places
max R̂ 3.325 (gamma)
chain 0 acceptance: 0.050%
chain 1 acceptance: 0.10%
...
```

> Two chains barely moved. Their acceptance rates were below 0.1%, and the
> chains disagreed strongly about the posterior, with a maximum $\hat R$ of
> 3.325. This fit is not usable. The trace plots show where the chains became
> stuck; the next step is to understand why.

The output gives evidence. The prose gives interpretation and a reason to keep
reading.

The block below is `show-lines="4-6"` on a real command, run when this page was
built: the staged latent transitions of a model from
[Dimensions and stratification](guide/dimensions.qmd).

```{.camdl-run show-lines="4-6"}
camdl inspect guide/dimensions/seir_erlang.camdl --transitions
```

## 7. Explain scientific checks without overclaiming

camdl is a tool for scientific inference. Its documentation must be unusually
careful about the difference between a successful computation and a reliable
scientific conclusion.

- A model that compiles may still have incorrect assumptions or invalid
  observation mappings.
- Good $\hat R$ and effective sample size support a convergence assessment; they
  do not demonstrate model adequacy.
- Recovering known parameters from one synthetic dataset is a useful check, not
  a guarantee of identifiability or calibrated uncertainty across repeated
  datasets.
- A posterior predictive check tests particular consequences of a fitted model;
  it does not prove the model is correct.
- A more flexible observation model may cover unusual data without explaining
  the underlying epidemiological process.
- In-sample predictive scores are not automatically estimates of out-of-sample
  forecasting performance.

State the relevant distinction where readers encounter it. Avoid turning each
caveat into a ceremonial disclaimer. The aim is clear scientific judgment, not
defensive prose.

**Instead of:** “The fit converged successfully, validating the model.”

**Write:** “The chains now agree, so we can examine the posterior. Whether the
model describes the school outbreak is a separate question, which we'll test
against the observed counts and the historical attack rate.”

## 8. Structure pages so they can be read at two speeds

A good camdl page works as a continuous chapter and as a page a reader skims
during work.

Open with a brief statement of the problem or task. Organize the main text into
sections with specific, descriptive headings. Put the central commands,
equations, interpretation, and decisions in the visible path. Put worthwhile
detours in labeled boxes close to the text that motivates them. Link to
reference material when a detail is needed for lookup rather than understanding.

Do not open a page with nine bullet points previewing nine sections unless the
length truly justifies a roadmap. Do not place a collapsible box after every
paragraph. When many boxes accumulate, the page may need to be split into a
tutorial and a deeper explanation.

For the boarding-school example, a productive division is:

1. **Your first Bayesian fit:** one model, one fit, convergence assessment, and
   a posterior predictive check.
2. **When a fit goes wrong:** the prior predictive failure, stuck chains,
   likelihood noise, and reparameterization.
3. **Criticizing an epidemic model:** competing observation models, predictive
   scores, and the mismatch with the historical attack rate.

The second and third pages can be scientifically substantial. They need not be
disguised as first-run instructions.

The boarding-school tutorial now on the site,
[Your first fit](guide/first-fit.qmd), predates this guide and does not yet
follow it.

## 9. Drafting and editing are separate jobs

An agent should not treat its first complete draft as the finished document.

**First, plan.** Identify the reader, objective, document type, main example,
conceptual dependencies, and material that could be deferred. Check nearby pages
to avoid duplication.

**Then, verify.** Read the current implementation, CLI help, and tests before
asserting syntax or behavior. Run commands when practical; record exact
versions, seeds, and outputs when they matter. Do not manufacture a clean result
or quietly remove scientifically relevant warnings.

**Then, write.** Draft the visible narrative before the collapsible boxes. If
the page makes no sense with the boxes closed, the main narrative is incomplete.
Add the boxes only after identifying genuine optional depth.

**Then, edit twice.** First review pedagogy and structure: is material in the
right order and at the right level? Then review prose line by line: awkward
rhythm, repetition, bureaucratic phrases, gratuitous definitions, and abrupt
transitions. Read the result aloud or as continuous prose. It should sound
authored, not assembled.

**Finally, audit the science.** Check equations, conditioning statements, units,
statistical interpretations, limitations, and the difference between evidence
and convenience. Correctness has priority over elegance.

## 10. Final editorial checklist

Before submitting documentation, an agent should be able to answer _yes_ to
these questions:

- Can the intended reader tell why the page exists within its first few
  sentences?
- Is its main reading path sufficient without expanding any boxes?
- Are important equations visible and explained, rather than hidden because they
  seem technical?
- Are optional explanations properly categorized as green practical guidance,
  yellow modeling judgment, or orange mechanics?
- Does every collapsible title tell readers what question it answers?
- Are serious warnings and validity limits visible at the point where they
  matter?
- Does code match current camdl behavior, and is shown output verified?
- Are mathematical symbols, conditioning statements, units, and likelihood
  terminology correct?
- Are paragraphs doing explanatory work rather than narrating obvious code or
  repeating headings?
- Would this page feel at home in a carefully edited scientific computing book?

Good documentation should make the sophisticated parts of camdl accessible
without pretending they are simple. Teach the scientific ideas, give readers a
trustworthy path through the software, and make deeper explanations inviting
rather than compulsory.
