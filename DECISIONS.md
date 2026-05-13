# Decisions Log

Every meaningful technical decision made while building this local agentic stack. One line of reasoning per row; the longer narrative is in the [blog post](https://blog.zacharycangemi.com). Each decision is tagged `general` (applies to any local agentic setup) or `hardware-specific` (specific to this 2× RTX 3090 rig).

Read time: ~5 minutes. If you want the full story behind any single decision, the source for each is named in the "Source" column.

---

## Hardware / Platform

| # | Decision | Choice | Why (one line) | Tag | Source |
|---|---|---|---|---|---|
| 1 | Inference platform | Windows 11 + llama.cpp native | Avoids WSL2 overhead and virtualization tax; full GPU control | hardware-specific | blog §Act I |
| 2 | Agent client OS | macOS (separate from inference) | Mac is the daily driver; Hermes runs comfortably; Tailscale connects to the inference server | general | blog §Act I |
| 3 | Network between agent + inference | Tailscale mesh VPN | Zero port forwarding, encrypted, works on any network, no public IP exposure | general | blog §Act I |

## Model Selection

| # | Decision | Choice | Why (one line) | Tag | Source |
|---|---|---|---|---|---|
| 4 | Primary model (accuracy) | Qwen3.5-27B (Unsloth UD-Q5_K_XL) | 100% ToolCall-15 at temp 0.6; thinking traces act as a built-in planning layer | general | benchmarks/toolcall15_leaderboard.md |
| 5 | Speed model (alternative) | GLM-4.7-Flash (Unsloth UD-Q5_K_XL) | 2.3–3.4× faster than Qwen; 93% ToolCall-15; no reprocessing bug (pure MLA attention) | general | benchmarks/glm_vs_qwen.md |
| 6 | Live model switching | Hermes `/model` command | Same Hermes config, same port, hot-swap underlying model based on task needs | general | blog §two-model dance |
| 7 | 27B-dense is the sweet spot | (framing) | Smaller models lack capacity for simultaneous tool decision; larger models' priors override tools | general | blog §Act I |
| 8 | Why Qwen3-Coder-Next 80B was ruled out | (eliminated) | 77% on ToolCall-15 vs Qwen3.5-27B's 97% — prior-strength caused tool avoidance, MoE diluted signal | general | benchmarks/toolcall15_leaderboard.md |
| 9 | Why Hermes-4.3-36B was ruled out | (eliminated) | 35→15 t/s degradation across context — dealbreaker for agentic sessions that grow large | general | hermes_progress_log §Research Round 7 |
| 10 | Why Qwen3-Coder-30B-A3B was ruled out | (eliminated) | Too weak for Hermes — confirmed by Hermes maintainer; small models default to curl/Python fallbacks | general | hermes_progress_log §Day 1 |
| 11 | Why SGLang was ruled out as backend | (eliminated) | Worse Qwen3.5 support than vLLM; no native tool-call parser | general | hermes_progress_log §Day 1 |
| 12 | Why Claude-distilled models were ruled out | (eliminated) | Storage/quality tradeoff not worth it on consumer hardware | general | hermes_progress_log §Day 1 |
| 13 | GLM was initially ruled out, then added back | (reversal) | Lower coding SWE-Bench (59.2%) — but exceptional tool calling (79.5% tau2-bench); kept as speed alternative | general | hermes_progress_log §Day 5 |

## Inference Engine

| # | Decision | Choice | Why (one line) | Tag | Source |
|---|---|---|---|---|---|
| 14 | Inference engine | llama.cpp (native Windows, built from source, CUDA 12.6) | 1.8× faster generation, 4× faster prompt eval vs vLLM on this hardware | hardware-specific | blog §Act I |
| 15 | Why vLLM was ruled out (for this setup) | (eliminated for consumer rig) | 7 compounding penalties on Ampere + Windows + WSL2; BF16 doesn't fit in 48 GB; FP8 W8A16 suboptimal on Ampere | hardware-specific | hermes_progress_log §Day 2-4 |
| 16 | llama.cpp build version | b8720 → b8800+ during experiment | PR #20970 (Qwen3.5 thinking + tool-call) and PR #19866 (FA multi-GPU fix) both critical for this workload | general | benchmarks/flash_attention_canary.md |

## Quantization / KV Cache

| # | Decision | Choice | Why (one line) | Tag | Source |
|---|---|---|---|---|---|
| 17 | Model quantization | Unsloth Dynamic Q5_K_XL | UD preserves router (f32) and attention precision; only FFN compressed — exactly where tool calling needs precision | general | blog §Act I |
| 18 | Why not Q8_0 | (alternative considered) | Q8 99.9% vs UD-Q5 99.4% of BF16 — unmeasurable gap; UD-Q5 saves 9.5 GB and fits single GPU | general | hermes_progress_log §Day 4 |
| 19 | KV cache quantization | f16 (no quantization) | 21+ GB VRAM headroom at 96K context; q8_0 saves ~2 GB for nothing; q4_0 documented to break arithmetic on some models | hardware-specific | blog §Act I |
| 20 | CPU offloading | None (full GPU) | Dense 27B models bottleneck on every token if any layers offloaded; only MoE benefits from CPU offload | general | hermes_progress_log §Day 4 |

## Performance / Stability

| # | Decision | Choice | Why (one line) | Tag | Source |
|---|---|---|---|---|---|
| 21 | Flash Attention | Enabled (with safety layers) | 3.5× flatter degradation curve; at 84K context, 28.6 t/s vs 11.0 t/s without FA | general | benchmarks/flash_attention_canary.md |
| 22 | FA safety layers | f16 KV + `LLAMA_ATTN_ROT_DISABLE=1` + `-sm layer` | Mitigates open issue #21383 (RTX 3090 + agentic patterns crash path) | hardware-specific | benchmarks/flash_attention_canary.md |
| 23 | GPU split — Qwen | Both GPUs, layer split | 20.5 GB model doesn't fit on a single 24 GB 3090 | hardware-specific | configs/qwen3.5-27b_dual/launch.bat |
| 24 | GPU split — GLM | Single GPU (`--main-gpu 0`) | 20.2 GB fits on one 3090; avoids 30-50% dense PCIe x8 bifurcation penalty | hardware-specific | configs/glm-4.7-flash/launch.bat |
| 25 | Sampling temperature (Qwen) | temp=0.6 | Qwen-recommended for thinking mode; greedy decoding (temp=0) makes thinking models over-cautious | general | benchmarks/toolcall15_leaderboard.md |
| 26 | Sampling temperature (GLM) | temp=0.7, top-p 1.0, min-p 0.01 | GLM-recommended sampling profile; thinking via deepseek reasoning format | general | configs/glm-4.7-flash/launch.bat |
| 27 | Context window | 96K (98,304 tokens) | Fits f16 KV cache in VRAM budget; ~12K system prompt overhead leaves ~86K working space | hardware-specific | blog §Act I |
| 28 | TDR (Timeout Detection and Recovery) | Disabled (`TdrLevel=0`) | Default 2-second timeout causes Windows-killing cascade on multi-GPU NCCL cleanup | hardware-specific | hermes_progress_log §Day 2-4 |

## Agent Setup

| # | Decision | Choice | Why (one line) | Tag | Source |
|---|---|---|---|---|---|
| 29 | System prompt philosophy | Karpathy approach — shape thinking, not tools | SOUL.md = how to think; skills = how to use specific tools; model = which tool to pick | general | blog §SOUL.md |
| 30 | Web search backend | Tavily (native Hermes integration) | DuckDuckGo rate-limited + low quality; browser hits CAPTCHAs; Tavily free tier (1,000/mo) works reliably | general | blog §web search |
| 31 | Mobile access | Telegram gateway (bot via @BotFather) | 2-minute setup; full Hermes access from anywhere; sovereignty extends past the desk | general | blog §Telegram |
| 32 | Approval timeout | 300 seconds (was 60 default) | Long enough to review complex terminal commands without auto-timeout; mode stays manual (no auto-approve) | general | configs/hermes/config.yaml |
| 33 | Streaming | Enabled | UX improvement only — no speed change; visible token output during generation | general | configs/hermes/config.yaml |

## Things explicitly NOT changed

| # | Decision | Choice | Why (one line) | Tag |
|---|---|---|---|---|
| 34 | No `--cache-reuse` flag | (rejected) | Hybrid DeltaNet architecture cannot do partial KV cache reuse; llama.cpp logs warning and ignores | general |
| 35 | No `--grammar` constraints on GLM | (rejected) | Open issue #19068 — infinite loop with tool calling; use `--jinja` autoparser instead | general |
| 36 | No reasoning budget cap | (rejected) | Thinking traces enable tool-call accuracy; capping them hurts the primary use case | general |
| 37 | No Q4 model quantization | (rejected for the 27B dual-GPU path) | 10-15% speed gain not worth tool-call accuracy risk on the accuracy daily driver | general |

---

## How to read this log

- **`general` rows** — port to any local agentic setup. Reasoning applies regardless of your hardware.
- **`hardware-specific` rows** — calibrated for 2× RTX 3090 / Ryzen 9 7950X3D / 96 GB DDR5-6000. If your hardware differs, the *direction* of the decision usually still applies; the exact numbers don't.

For decisions tagged `hardware-specific`, the matching transferable question to ask on your own rig: *"What does the same reasoning produce given my VRAM, my PCIe topology, my GPU architecture?"*

---

## Decisions that didn't make this cut

A few decisions were considered and either deferred, untested, or determined not to matter at the scale of this experiment:

- **Native Linux on Threadripper** — would eliminate most vLLM penalties (no WSL2, no WDDM tax, native FP8 paths). Future build.
- **NVLink bridge** — would help training significantly; minimal impact on inference at our scale. Not pursued.
- **Speculative decoding** — not supported for Qwen3.5's hybrid DeltaNet architecture.
- **Single-GPU Qwen3.5 (Q4 + q8_0 KV)** — would save the second GPU for other work, but tight VRAM headroom and increased risk for marginal performance gain.

---

## License

This decision log, and the rest of this repo, is MIT-licensed. See `LICENSE`. Use any of these decisions in your own setup, write up your own findings, and link back if it helped.
