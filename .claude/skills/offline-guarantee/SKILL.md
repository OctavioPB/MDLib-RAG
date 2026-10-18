---
name: offline-guarantee
description: Guarantees and verification that the system makes no network calls at runtime (operational independence, ADR-002). Use it ALWAYS when a dependency is added or updated, model loading is touched (sentence-transformers, huggingface-hub, transformers), code that uses HTTP is written, the LLM client is configured, or `biblio models pull` or `biblio doctor` is modified.
---

# Offline guarantee

The core requirement of the project is that not a single byte leaves the machine at runtime. ML
libraries make implicit network calls (version checks, lazy downloads, telemetry) that are not visible
in our own code. That is why the guarantee cannot rest on "I did not write any call".

## Layers of defense

1. **Environment variables at startup**, before importing any ML library. In `cli.py`, at process start,
   if `runtime.offline = true`:
   `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
   `DO_NOT_TRACK=1`. Setting them before the imports matters: some libraries read these variables at
   import time. The list lives in `biblio.offline.OFFLINE_ENV`.
2. **Load models by local path**, never by repository name: `SentenceTransformer(str(models_dir / "bge-m3"), local_files_only=True)`
   (or whatever equivalent the installed version accepts). If the directory does not exist, raise a clear
   error suggesting `biblio models pull`; never fall back to downloading. Resolve the path with
   `biblio.models.local_model_path`, which also checks the pinned revision against `models/manifest.json`.
3. **LLM client restricted to loopback.** When building `LLMClient`, validate that the host of `base_url`
   is `127.0.0.1`, `::1` or `localhost` (`biblio.offline.require_loopback`). Do not resolve DNS to check it:
   resolution itself is a network call. Any other host is a configuration error, not a warning.
4. **Tests with the network blocked.** `pytest-socket` in `addopts` with only localhost allowed. A test
   that fails because of a blocked socket reveals a hidden network call: investigate it; do not unblock it.
5. **`biblio doctor`** checks the variables, the presence of the models and that `base_url` is loopback,
   and reports any deviation in red.

## `biblio models pull`

It is the only exception. It downloads with `huggingface_hub.snapshot_download(repo_id, revision=<hash>, local_dir=...)`.
Always pin the revision by commit hash, never by branch: a branch can change the weights under your feet.
It stores the repo, the revision, the date and the declared license in `models/manifest.json`.
To move to a newer revision, change the hash in `config/default.toml` (after reviewing the license and,
for the embedding model, accepting that it forces a full reindex) and run `pull` again.

## When adding a dependency

1. Does it make network calls on import or on use (telemetry, updates, lazy downloads)? Check its documentation and search its code for `requests`/`httpx`/`urllib` if in doubt.
2. Does it have an environment variable or an option to disable them? Add it to layer 1.
3. Run the suite with the network blocked and a real command (`biblio search`) with the system network disabled.
4. If the dependency cannot work offline, it does not get in. Look for an alternative or write an ADR.

## Periodic manual verification

With the system network disabled (airplane mode or firewall), run `doctor`, `index`, `search` and `ask`.
All four must work the same as with the network on.
