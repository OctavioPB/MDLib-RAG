---
name: memory-budget
description: Calculation and verification of the memory budget (< 8 GB) for local models — quantized weights, KV cache, buffers and the coexistence of embeddings, reranker and LLM. Use it ALWAYS when a model (LLM, embeddings, reranker) is chosen, added or changed, the quantization changes, context_budget_tokens or the context window is adjusted, the LLM runtime is configured, or OOM errors or extreme slowness appear.
---

# Memory budget

Hard constraint: under 8 GB for models (ADR-008). The most common mistake is to count only the weights
and forget the KV cache, which grows linearly with the context.

## Formulas

**Quantized weights.** `bytes ≈ n_params × bits_per_weight / 8`. Approximate effective bits in GGUF:
Q4_K_M ≈ 4.8, Q5_K_M ≈ 5.7, Q8_0 ≈ 8.5, F16 = 16.

**KV cache.** `bytes = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element × n_ctx`

- Use `n_kv_heads`, not `n_heads`: models with GQA have far fewer KV heads.
- `bytes_per_element`: 2 in f16, ~1 in q8_0, ~0.5 in q4_0.
- Take `n_layers`, `n_kv_heads` and `head_dim` from the `config.json` or the GGUF metadata of the specific
  model. Do not assume them: they vary between families and versions.

**Compute and runtime buffers.** Reserve 0.5–1 GB.

**Margin.** Leave ≥ 10% free. The operating system, the desktop and the browser also use VRAM, and on
Apple Silicon the unified memory is shared with the whole system.

## Worked example (Llama-style 8B architecture: 32 layers, 8 KV heads, head_dim 128)

- Q4_K_M weights: 8.0e9 × 4.8 / 8 ≈ 4.8 GB
- KV per token in f16: 2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KiB
- KV with 8k context: 128 KiB × 8192 = 1 GiB in f16, ~0.5 GiB in q8_0
- Total with buffers: ≈ 6.3–6.8 GB. It fits in 8 GB only if the GPU is almost free. With less than ~7 GB
  usable, drop to a 3–4B model or reduce the context.

## Procedure for any model or context change

1. Get the real architecture of the model (layers, KV heads, head_dim, maximum context) and the license.
2. Compute weights + KV at the configured `context_budget_tokens + max_answer_tokens` + buffers.
3. Compare it with the usable memory reported by `biblio doctor` (ADR-011).
4. If it does not fit, consider in this order: quantize the KV to q8_0 → reduce the context → smaller model
   → partial offload to RAM (llama.cpp `--n-gpu-layers` < total). Offloading multiplies latency;
   note it in the ADR.
5. Verify empirically: start the runtime and observe the real memory with a query at maximum context.
   Record the measured number next to the computed one.
6. Record the choice in an ADR with the calculation.

## Coexistence

- Query embeddings (bge-m3, ~2.3 GB fp32) run on **CPU** and use system RAM, not VRAM.
- The reranker (bge-reranker-v2-m3, ~2.3 GB fp32, ~1.1 GB fp16) runs on the **GPU** for `search` while
  no LLM server is running; when one is, `search` skips it (ADR-014). Phase 5 decides whether it can
  share the GPU with the LLM during `ask`: count it in that budget.
- Batch indexing embeds on the **GPU** (ADR-012): measured peak 2.71 GiB VRAM in fp32 (1.38 GiB in
  fp16) at batch 16 and `max_seq_length` 1024. Raising `batch_size` or `max_seq_length` raises that
  peak, so re-measure.
- At query time, the reranker and the LLM coexist: add the reranker's RAM to the RAM the LLM uses if there is offload.
- On Apple Silicon everything shares the same pool. Add the three models against the total unified memory
  minus what the system uses.
- Bulk indexing and the LLM server never run at the same time (ADR-012); the indexer refuses to use the
  GPU while the server answers on `llm.base_url`.

## Warning signs

- OOM on the first long token → the KV cache was not counted or the real context exceeds the configured one.
- Very slow first token and then normal → loading from disk or swap. Check that the model fits without swap.
- Persistently very low tokens/s → unintended offload to CPU. Check the layers on GPU in the runtime log.
