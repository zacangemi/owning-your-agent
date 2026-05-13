@echo off
REM ============================================================
REM  Qwen3.5-27B (Q4_K_XL) — Single GPU Variant (Accessibility)
REM  Engine: llama.cpp b8720+ (native Windows)
REM  GPU:    Single RTX 3090 (24 GB VRAM)
REM  Port:   8000 | Context: 32K | KV cache: q8_0 (quantized)
REM  Flash Attention: ENABLED (single GPU avoids multi-GPU FA crash path)
REM
REM  Tradeoffs vs the dual-GPU UD-Q5 config:
REM    - Q4_K_XL (14-16 GB) instead of UD-Q5_K_XL (18.8 GB)
REM    - Lower quality on edge cases (~95%+ tool-calling, vs 97% with Q5)
REM    - q8_0 KV cache (lossless on this model per #20035) for VRAM headroom
REM    - 32K context (vs 96K) due to single-GPU VRAM budget
REM
REM  Adjust the C:\ paths to match your install locations.
REM ============================================================

set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM Safety: disables an experimental rotation transform that crashes on
REM RTX 3090 + agentic prompt patterns (open issue #21383). Required.
set LLAMA_ATTN_ROT_DISABLE=1

C:\llama-cpp\build\bin\Release\llama-server.exe ^
  -m "C:\models\qwen3.5-27b-gguf\Qwen3.5-27B-Q4_K_XL.gguf" ^
  -ngl 99 ^
  --main-gpu 0 ^
  -c 32768 ^
  -n -1 ^
  --no-context-shift ^
  --flash-attn on ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  --jinja ^
  --reasoning-format deepseek ^
  --temp 0.6 ^
  --top-p 0.95 ^
  --top-k 20 ^
  --min-p 0 ^
  --repeat-penalty 1.0 ^
  --parallel 1 ^
  --host 0.0.0.0 ^
  --port 8000
