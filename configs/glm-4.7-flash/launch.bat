@echo off
REM ============================================================
REM  GLM-4.7-Flash (Unsloth UD-Q5_K_XL) — Speed Daily Driver
REM  Engine: llama.cpp b8720+ (native Windows)
REM  GPU:    Single RTX 3090 within the dual-rig (no PCIe penalty)
REM  Port:   8000 | Context: 32K | KV cache: f16 (MLA requires it)
REM  Flash Attention: ENABLED
REM
REM  Architecture notes:
REM    - 30B MoE, ~3-3.6B active params per token (~9x faster than dense 27B)
REM    - MLA attention (all 47 layers) — no recurrent state, no reprocessing bug
REM    - KV cache MUST be f16 — MLA's compressed representation cannot be further
REM      quantized without disproportionate quality loss.
REM    - Do NOT use --grammar (open issue #19068: infinite loop with tool calling).
REM
REM  Adjust the C:\ paths to match your install locations.
REM ============================================================

set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM ATTN_ROT_DISABLE is irrelevant for MLA — clear it.
set LLAMA_ATTN_ROT_DISABLE=

C:\llama-cpp\build\bin\Release\llama-server.exe ^
  -m "C:\models\glm-4.7-flash-gguf\GLM-4.7-Flash-UD-Q5_K_XL.gguf" ^
  -ngl 99 ^
  --main-gpu 0 ^
  -c 32768 ^
  -n -1 ^
  --no-context-shift ^
  --flash-attn on ^
  --cache-type-k f16 ^
  --cache-type-v f16 ^
  --jinja ^
  --temp 0.7 ^
  --top-p 1.0 ^
  --min-p 0.01 ^
  --repeat-penalty 1.0 ^
  --parallel 1 ^
  --host 0.0.0.0 ^
  --port 8000
