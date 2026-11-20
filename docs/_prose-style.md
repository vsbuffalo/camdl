### Patterns to avoid

Write technical documentation in direct, restrained prose, in the manner of a
good methods section or a well-written engineering design document. Explain
mechanisms, assumptions, evidence and consequences without rhetorical
embellishment. Aim for information-dense, natural technical writing: the prose
of an engineer or scientist explaining something they understand.

Language models fall into a recognizable set of patterns, and so do hurried
human writers. Each one below is named by what it does to the reader, because
banning a phrase only invites its synonym: forbid "importantly" and "crucially"
takes its place.

1. **Negation and reframing.** "X is not merely Y; it is Z" and "this isn't
   about X, it's about Y" make the reader dismiss a claim nobody made before
   hearing the real one. State the positive claim. Use contrast only to correct
   a misconception the reader is likely to hold.
2. **Manufactured contrast.** "Rather than X, we do Y" invents an alternative to
   make the choice look considered. Mention an alternative only when it matters
   to the technical argument.
3. **Inflated significance.** "Critical architectural boundary", "fundamental
   insight", "powerful abstraction" and "key design principle" assert importance
   without showing it. Show the consequence instead, and the reader will judge
   the importance.
4. **Attention words.** "Importantly", "notably", "it's worth noting", "the key
   insight is" and "this highlights" ask for attention without adding
   information; delete them. Connectives that state logic ("thus",
   "consequently", "because", "so") are different: they tell the reader how one
   claim follows from another, and they stay.
5. **The colon reveal.** "The block connects the two: it says…" and "The answer
   is simple: …" stage a small reveal. Write the sentence that says the thing.
6. **Repetition.** Make each point once. Do not close a paragraph with a
   sentence that restates it ("In short", "This means that"), and do not
   introduce a concept twice to make the prose feel complete. An example should
   add information the text did not already give.
7. **Artificial parallelism.** Rhetorical triads, runs of identically built
   sentences, and lists of three abstract adjectives ("clean, robust and
   consistent") come from rhythm, not content. Use the structure the material
   requires.
8. **Excess formatting.** Connected ideas belong in connected paragraphs.
   Reserve lists for items that are genuinely separate and enumerable, and do
   not give every list item a bold lead-in.
9. **Anthropomorphism.** Software computes, checks, stores, refuses and records.
   It does not "want", "know", "understand", "see" or "recognize". A
   surveillance system records reported cases; a design does not recognize
   anything.
10. **Self-congratulation.** Calling a design safe, clean, robust, elegant or
    intentional says nothing a reader can check. Describe the invariant, the
    mechanism, the failure it prevents or the trade-off it makes.
11. **Decoration.** Em-dash asides used for rhythm, stacked hedges ("may
    potentially"), rhetorical questions that open a section, and sentences that
    add emphasis but no information. Remove them before finishing.

Some habits to keep. Prefer concrete nouns and active verbs. Make causal claims
precise, and distinguish what is guaranteed, observed, inferred and assumed.
Explain why a design choice matters when the reason is not obvious. Keep genuine
qualifications, uncertainty and important contrasts. Do not remove useful
technical detail in pursuit of brevity.

Credit the work camdl builds on. When camdl follows another project's design or
borrows its idea (such as the transform Stan applies to a bounded parameter),
name the project and cite a primary source: its documentation, with the version
and section, or its paper. A comparison of this kind tells the reader where
camdl follows established practice and where to read more. Describe the other
project accurately and specifically.

### Testing prose

A language model asked to review prose is reliable at finding factual errors,
undefined terms and sentences an agent could extract as a wrong fact. It is
unreliable at judging voice: model judges favour text that resembles their own
output (the "self-enhancement bias" of
[Zheng et al. 2023](https://arxiv.org/abs/2306.05685);
[Panickssery et al. 2024](https://arxiv.org/abs/2404.13076)). Use a model to
check what a text claims. Judge how it reads yourself.
