# Starts the tutor's LLM server with the settings of ADR-015 (memory budget measured with them).
# Loopback only, flash attention (required for a quantized KV cache), KV cache q8_0, 8k context.
# One model loaded at a time: the tutor's 4B and the «Razonar» model take turns (ADR-057).
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
$env:OLLAMA_CONTEXT_LENGTH = "8192"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
# Ollama 0.40 runs llama-server, whose prompt cache keeps up to 8 GiB of past prompts in RAM by
# default: with Gemma 4 it grew 0.3-0.4 GB per request until the PC swapped. Off, it stays flat.
$env:LLAMA_ARG_CACHE_RAM = "0"
ollama serve
