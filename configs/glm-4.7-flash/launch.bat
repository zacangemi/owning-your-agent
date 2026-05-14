@echo off
REM ============================================================
REM  GLM-4.7-Flash (Unsloth UD-Q5_K_XL) — Speed Daily Driver
REM  Engine: llama.cpp b8720+ (native Windows)
REM  GPU:    2x RTX 3090 (layer split, dual GPU)
REM  Port:   8000 | Context: 96K | KV cache: f16 (MLA requires it)
REM  Flash Attention: ENABLED
REM
REM  Architecture notes:
REM    - 30B MoE, ~3-3.6B active params per token (~9x cheaper per-token
REM      compute than Qwen3.5-27B dense; that's why fresh-context speed is
REM      ~124 t/s vs Qwen's ~36 t/s).
REM    - MLA attention (all 47 layers) — no recurrent state, no reprocessing bug.
REM    - KV cache MUST be f16 — MLA's compressed representation cannot be further
REM      quantized without disproportionate quality loss.
REM    - Do NOT use --grammar (open issue #19068: infinite loop with tool calling).
REM    - Dual GPU layer split: ~10-20% PCIe penalty for MoE (vs 30-50% for
REM      dense). Acceptable cost in exchange for matching Qwen's 96K context
REM      window — same launch flags, same Hermes config, /model swaps cleanly.
REM
REM  Adjust the C:\ paths to match your install locations.
REM ============================================================

set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM ATTN_ROT_DISABLE is irrelevant for MLA — clear it.
set LLAMA_ATTN_ROT_DISABLE=

C:\llama-cpp\build\bin\Release\llama-server.exe ^
  -m "C:\models\glm-4.7-flash-gguf\GLM-4.7-Flash-UD-Q5_K_XL.gguf" ^
  -ngl 99 ^
  -sm layer ^
  --tensor-split 1,1 ^
  -c 98304 ^
  -n -1 ^
  --no-context-shift ^
  --flash-attn on ^
  --cache-type-k f16 ^
  --cache-type-v f16 ^
  --jinja ^
  --reasoning-format deepseek ^
  --temp 0.7 ^
  --top-p 1.0 ^
  --min-p 0.01 ^
  --repeat-penalty 1.0 ^
  --parallel 1 ^
  --host 0.0.0.0 ^
  --port 8000
