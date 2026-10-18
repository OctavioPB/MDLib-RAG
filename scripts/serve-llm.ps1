# Starts the tutor's LLM server with the settings of ADR-015 (memory budget measured with them).
# Loopback only, flash attention (required for a quantized KV cache), KV cache q8_0, 8k context.
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
$env:OLLAMA_CONTEXT_LENGTH = "8192"
ollama serve
