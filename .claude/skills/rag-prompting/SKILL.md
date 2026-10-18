---
name: rag-prompting
description: Prompt design, context assembly, citations and abstention for generation with a local LLM (`ask` tutor and `draft` writing modes). Use it ALWAYS when working in src/biblio/generate/, writing or changing a prompt, adjusting how many fragments go into the context, implementing citation validation or map-reduce, or evaluating the quality of generated answers.
---

# Grounded generation with small models

3–8B models follow instructions worse than large ones, get distracted by long context and tend to fill
in with their own knowledge when the evidence is weak. The design compensates with structure, not with
longer prompts.

## Context assembly

- Order the fragments by reranker score and number them `[1]..[n]`. Each one carries a header with the
  breadcrumb and the path: `[3] estadistica/fisher.md › Información de Fisher › Cota de Cramér-Rao`.
- Respect `context_budget_tokens` counting with the LLM's tokenizer (not with characters). If it does not
  fit, remove whole fragments from the end; never truncate a fragment in the middle.
- Prefer 4–6 good fragments to 12 mediocre ones. More context makes small models worse.
- Contiguous fragments from the same document are merged into one with their line range.

## Tutor prompt (structure)

1. Role and main rule, in a few lines: answer only with the given evidence; cite every claim with `[n]`;
   if the evidence is not enough, say so and do not fill in with your own knowledge.
2. The numbered evidence.
3. The question.
4. Expected output format, briefly.

Write the prompt in the language of the queries (Spanish). Keep the instructions short and concrete:
with small models, every extra instruction dilutes the others. Store prompts in versioned text files in
`src/biblio/generate/prompts/`, never as strings scattered through the code.

## Post-validation (not optional)

- Extract the `[n]`s from the answer. A citation to a non-existent index is an error; flag the answer.
- An answer without any citation to a question that is not `absent` is shown flagged as "ungrounded".
- Show at the end the sources actually cited, with the path and the line range.

## Map-reduce for synthesis questions

If the relevant evidence exceeds the budget or comes from 3+ different guides:

1. **Map:** for each group of fragments (grouped by document), ask for a short summary oriented to the
   question that preserves the original citations.
2. **Reduce:** synthesize the summaries into the final answer, preserving the citations.

It costs more calls and latency. Enable it only when needed (criterion in config) and report it in the CLI.

## Writing mode (`draft`)

- Two steps with human intervention: first an outline of sections from the evidence and from 1–2 sibling
  guides as a structural model; the user approves or edits it; then each section is written in turn.
- Each section retrieves its own evidence (query = topic + section title).
- The final draft ends with the list of sources used.
- Honest expectation (ADR-008): with this hardware, the result is a skeleton to edit, not a finished guide.
- Implementation (`generate/draft.py`): the outline is a Markdown file of `## ` lines the user edits;
  sibling guides contribute their H2/H3 structure only; each section's local `[n]` is renumbered to
  a global source list; the guide's own title and table-of-contents sections are dropped from the
  outline, and a heading repeating the section title is stripped (small models add both despite
  the prompt). Sections without evidence carry the fixed `draft_missing` sentence.

## Evaluating answers

Measure three things on a small set (10–15 questions) with human review:

- **Faithfulness:** every claim is backed by the fragment it cites.
- **Coverage:** the answer uses the relevant evidence available.
- **Abstention:** on `absent` questions, the system says it is not in the library.

A new prompt is compared against the previous one on the same set before replacing it.
