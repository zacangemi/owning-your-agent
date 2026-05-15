@echo off
REM ============================================================
REM  Qwen3.5-27B (Unsloth UD-Q5_K_XL) — Accuracy Heavy Hitter
REM  Engine: llama.cpp b8720+ (native Windows)
REM  GPU:    2x RTX 3090 (layer split, dual GPU)
REM  Port:   8000 | Context: 96K | KV cache: f16
REM  Flash Attention: ENABLED (verified safe with safety layers)
REM
REM  Adjust the C:\ paths to match your install locations.
REM ============================================================

set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM Safety: disables an experimental rotation transform that crashes on
REM RTX 3090 + agentic prompt patterns (open issue #21383). Required.
set LLAMA_ATTN_ROT_DISABLE=1

C:\llama-cpp\build\bin\Release\llama-server.exe ^
  -m "C:\models\qwen3.5-27b-gguf\Qwen3.5-27B-UD-Q5_K_XL.gguf" ^
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
  --temp 0.6 ^
  --top-p 0.95 ^
  --top-k 20 ^
  --min-p 0 ^
  --repeat-penalty 1.0 ^
  --parallel 1 ^
  --host 0.0.0.0 ^
  --port 8000
