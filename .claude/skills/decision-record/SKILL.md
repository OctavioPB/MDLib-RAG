---
name: decision-record
description: How to propose, write and update architecture decisions (ADR) in DECISIONS.md. Use it ALWAYS when choosing between technical alternatives (model, library, schema, algorithm, threshold with impact), when wanting to revert or change something already decided, when closing an open decision, or when the user asks why something was done a certain way.
---

# Decision record

DECISIONS.md is the memory of the why. It exists so that a future session (human or Claude) does not
reopen a settled discussion without knowing what was weighed.

## When an ADR is needed

- Choosing between two or more reasonable alternatives (model, library, format, algorithm).
- Fixing a value with architectural impact (embedding dimension, LLM context, chunker version).
- Accepting a known limitation (quality ceiling, latency).
- Changing or reverting an accepted decision. This is **always** a new ADR that supersedes the old one;
  the content of an accepted ADR is never edited.

They are not needed for implementation details that can be reverted at no cost (variable names, the
internal structure of a module).

## Format

```markdown
## ADR-NNN — Title that states the decision, not the topic

**Date:** YYYY-MM-DD · **State:** Proposed | Accepted | Superseded by ADR-MMM | Rejected

**Context.** The problem and the relevant constraints. What forces the decision now.

**Options.** Each serious alternative, with its main advantage and cost. Include "do nothing" if viable.

**Decision.** What was chosen, in one or two sentences.

**Consequences.** (+) what is gained. (−) what is paid or assumed. Risks and the condition that would force revisiting the decision.
```

## Rules

1. Sequential numbering, never reusing numbers. Update the index table at the top of the file.
2. A decision backed by measured evidence (evaluation, memory, latency) cites the numbers and the run in `eval/results/`.
3. When superseding: the old ADR only changes its State field to "Superseded by ADR-NNN"; the new one
   explains what changed in the context or in the evidence.
4. If the decision depends on missing information, it stays **Proposed** with a "Pending" section that
   says exactly which piece of data closes it.
5. Propose non-trivial decisions to the user before implementing them: present the options with their
   cost and your justified recommendation, and wait for their choice. Trivial decisions within an already
   accepted ADR do not require consultation.
6. If while working you notice that the code contradicts an accepted ADR, point it out; do not silently
   fix it in either direction.
