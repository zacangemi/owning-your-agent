# Decisions Log

Every meaningful technical decision made while building this local agentic stack. One line of reasoning per row; the longer narrative is in the [blog post](https://blog.zacharycangemi.com).

**The 27B-dense sweet spot.** This entire stack is built around the empirical finding that 27B-class dense models are the right choice for local agentic tool-calling work. Smaller models (<14B) lack the parameter budget for simultaneous intent understanding + tool selection + parameter generation + restraint, and default to curl/Python fallbacks. Larger models (70B+) have strong priors that override tool-call signals — they "know" the answer and skip the tools. 27B-dense gives you enough capacity for multi-step reasoning without overriding tool use, and it fits on consumer hardware (1-2 RTX 3090s) without aggressive quantization that costs accuracy. Most decisions below follow from this anchor.

**How to read this log.** These decisions are calibrated for our specific rig (2× RTX 3090 / Ryzen 9 7950X3D / 96 GB DDR5-6000). On different hardware, the *direction* of most decisions still applies — the exact numbers don't. The transferable question to ask on your own setup: *"What does the same reasoning produce given my VRAM, my PCIe topology, my GPU architecture?"*

Read time: ~5 minutes.

---

## Hardware / Platform

| # | Decision | Choice | Why (one line) |
|---|---|---|---|
| 1 | Inference platform | Windows 11 + llama.cpp native | Avoids WSL2 overhead and virtualization tax; full GPU control |
| 2 | Agent client OS | macOS (separate from inference) | Mac is the daily driver; Hermes runs comfortably; Tailscale connects to the inference server |
| 3 | Network between agent + inference | Tailscale mesh VPN | Zero port forwarding, encrypted, works on any network, no public IP exposure |

## Model Selection

| # | Decision | Choice | Why (one line) |
|---|---|---|---|
| 4 | Primary model (accuracy) | Qwen3.5-27B (Unsloth UD-Q5_K_XL) | 100% ToolCall-15 at temp 0.6; thinking traces act as a built-in planning layer |
| 5 | Speed model (alternative) | GLM-4.7-Flash (Unsloth UD-Q5_K_XL) | 2.3–3.4× faster than Qwen; 93% ToolCall-15; no reprocessing bug (pure MLA attention) |
| 6 | Why Qwen3-Coder-Next 80B was ruled out | (eliminated) | 77% on ToolCall-15 vs Qwen3.5-27B's 97% — large model's strong priors caused tool *avoidance* (the model "knows" the answer and skips the tools), and MoE routing diluted the tool-call signal further. Bigger ≠ better for agentic work. |
| 7 | Why Claude-distilled fine-tunes were ruled out | (eliminated) | Hobbyist Claude-distilled models on HuggingFace (e.g., Jackrong's Qwen3.5-Claude-4.6-Opus-Reasoning-Distilled series) are fine-tunes of base Qwen/Llama on Claude reasoning outputs — optimized for reasoning *brevity* (shorter CoT chains), not for tool calling. For agentic work, the base Qwen3.5-27B preserves the careful step-by-step tool-call reasoning the model was originally trained for. **Important distinction:** these hobbyist fine-tunes ≠ full lab-distillation pipelines (Z.ai's GLM family, Moonshot's Kimi K2, DeepSeek's training process), which use teacher-student training plus serious RLHF/RLAIF and produce competitive frontier-grade models. We avoided the hobbyist variants on consumer hardware. |

### Models we evaluated and ruled out

We seriously considered models from the Qwen, Hermes-4 / Hermes-4.3, Devstral, Mistral Small, GPT-OSS, and GLM families. Some we tested directly; others we ruled out at the research stage. Honest accounting below.

**Researched and ruled out (not tested in our setup):**

| Candidate | Why ruled out |
|---|---|
| Hermes-4.3-36B | 35→15 t/s degradation across context (verified by community testing — @sudoingX). Dealbreaker for agentic sessions that grow large. |
| Hermes-4-14B | 14B sits below the documented 27B-dense sweet spot for agentic work. Hermes maintainer confirmed small models default to curl/Python fallbacks under tool-call pressure. Native Hermes tool format was the appeal but didn't outweigh capacity limits. |
| Devstral Small 2 24B | Mistral's purpose-built agentic coding model — 24B dense, 68% SWE-Bench, native FP8. Required vLLM (which we ruled out for our consumer Ampere + Windows setup — see row 9). No Hermes community working examples at evaluation time. Lower SWE-Bench than Qwen3.5-27B (72.4%). Worth revisiting on native Linux. |
| Mistral Small (3.x family) | Smaller Mistral models sit in the 14-24B range. Public agentic-tool-calling benchmarks show Qwen3.5-27B scoring higher (e.g., on tau2-bench-class evals). Without a clear differentiator, didn't justify another evaluation cycle on an already-stretched scope. |
| GPT-OSS variants (20B / 120B) | OpenAI's mid-2025 open-source releases. Two reasons we passed: (1) GPT-OSS-20B sits below the 27B-dense sweet spot, and public agentic-tool-calling benchmarks (tau2-bench, ToolCall-15-class) show weaker performance than Qwen3.5-27B and GLM-4.7-Flash — the GPT-OSS series wasn't trained with the same agentic-tool-calling emphasis as the newer Chinese open-source releases; (2) GPT-OSS-120B requires hardware beyond our dual-3090 envelope. Not selected for either hardware path. |

**Tested but ruled out for production:**

| Candidate | Status | Why ruled out |
|---|---|---|
| Qwen3-Coder-30B-A3B | Tested in our prior OpenCode validation: 62 t/s, 100% on 16 basic tool calls, junior-mid code quality | Maintainer-confirmed too weak for harder Hermes-class agentic workloads. Small Coder models default to curl/Python fallbacks under tool-call pressure. Wouldn't have come close to Qwen3.5-27B's 97% or GLM-4.7-Flash's 93% on ToolCall-15-class benchmarks. |
| Qwen3.5-27B FP8 on vLLM | Tested fully on ToolCall-15: 100% at temp=0 | Same model as our production primary, but 15.5 t/s vs 28.5 t/s on llama.cpp UD-Q5_K_XL — 1.8× slower, plus WSL2 + SSH tunnel + tmux infrastructure complexity. Ruled out for production. See engine row 9 for the full vLLM-vs-llama.cpp story. |

## Inference Engine

| # | Decision | Choice | Why (one line) |
|---|---|---|---|
| 8 | Inference engine | llama.cpp (native Windows, built from source, CUDA 12.6) | 1.8× faster generation, 4× faster prompt eval vs vLLM on this hardware |
| 9 | Why vLLM was ruled out (for this setup) | (eliminated for consumer Ampere + Windows) | The cleanest framing: **vLLM's FP8 path uses Marlin kernels** (software FP8 dequant on Ampere — RTX 3090 has no native FP8 tensor cores, so the dequant happens in software at ~10-15% throughput cost). On top of that: **WSL2 passthrough overhead** (`pin_memory=False`, GPU virtualization layer), **no GPU peer-to-peer** (TP communication crosses PCIe every layer), **forced `--enforce-eager`** (CUDA graphs crash on Ampere + FP8 + Qwen3.5 hybrid), **WDDM driver tax** (~1.4 GB per GPU reserved for Windows display), **PCIe x8 bifurcation** (vs x16), **no torch.compile**. Seven compounding penalties. Net result: vLLM at 15.5 t/s on FP8 vs llama.cpp at 28.5 t/s on Unsloth UD-Q5_K_XL. **Better consumer-hardware practice: llama.cpp + Unsloth UD quants on native Windows. vLLM's design assumes Ada/Hopper datacenter on native Linux** — not what we have. |
| 10 | Why SGLang was ruled out (for this setup) | (eliminated for this setup) | SGLang is a strong inference engine in general — RadixAttention prefix caching, structured generation, mature for production scaling. For our specific use case it had two specific blockers at evaluation time (March 2026): **(1) Qwen3.5 support was less mature than vLLM's** — fewer working configurations and community examples, and **(2) no native tool-call parser** — meaning we'd be stuck doing client-side parsing or hacking templates, which Hermes explicitly leaves to the inference server. Combined with the same Ampere-on-Windows penalty stack vLLM has (WSL2, no native FP8, etc.), SGLang offered no advantages over vLLM and meaningful disadvantages vs llama.cpp. Worth revisiting on native Linux with newer model support. |
| 11 | llama.cpp build version | b8720 → b8800+ during experiment | PR #20970 (Qwen3.5 thinking + tool-call) and PR #19866 (FA multi-GPU fix) both critical for this workload |

## Quantization

| # | Decision | Choice | Why (one line) |
|---|---|---|---|
| 12 | Model quantization | Unsloth Dynamic Q5_K_XL | UD preserves router (f32) and attention precision; only FFN compressed — exactly where tool calling needs precision |
| 13 | Why not Q8_0 | (alternative considered) | Q8 99.9% vs UD-Q5 99.4% of BF16 — unmeasurable gap; UD-Q5 saves 9.5 GB of VRAM that we use for context window instead |
| 14 | Why not UD-Q4_K_XL | (alternative considered) | UD-Q4_K_XL would save another ~4-6 GB and give an estimated 10-15% speed gain — but we didn't need either: UD-Q5_K_XL already fits comfortably with full attention/router precision preserved (UD keeps router at f32 and attention at high precision at every UD level), and our 21+ GB VRAM headroom at 96K context means we don't need the extra room. Q4 is also documented to potentially affect arithmetic and structured-output reliability in some models — not worth the risk on the accuracy daily driver. Note: For single-GPU setups (which we did not test in this experiment), Q4_K_XL is the documented community-recommended quant — the 24 GB VRAM budget makes the more aggressive quantization necessary. |

## KV Cache

| # | Decision | Choice | Why (one line) |
|---|---|---|---|
| 15 | KV cache quantization | f16 (no quantization) | **No need to quantize — both models on dual GPUs leave 21+ GB VRAM free at 96K context.** This is the dual-card advantage: on a single 3090 you'd need q8_0 KV cache to fit (per documented community patterns; we didn't test single-GPU). Architecture also keeps the KV cache naturally small: **Qwen3.5-27B is hybrid** — only **16 of 64 layers** use full attention (the other 48 are Gated DeltaNet with fixed-size recurrent state, no KV cache); within those 16 attention layers, GQA gives 4 KV heads vs 24 query heads (6× reduction). **GLM-4.7-Flash uses MLA** (Multi-head Latent Attention), which natively compresses K+V into a small latent representation, plus an MLA V-less optimization (K stored, V derived) that further shrinks the cache. **MLA also requires f16** — it can't be quantized further without disproportionate quality loss. Combination of model architecture + inference engine + dual-card VRAM budget means quantizing the KV cache would have saved ~2 GB we didn't need, in exchange for a quality variable we didn't want. |
| 16 | CPU offloading | None (full GPU) | **Plenty of VRAM headroom on dual GPUs (21+ GB free at 96K context) — no reason to push layers to CPU.** Architecture also matters here: dense 27B models bottleneck on every token if any layers are CPU-offloaded (every token activates every layer); MoE benefits from CPU offload only when active params per token are small enough that the bottleneck doesn't dominate. With our setup, full GPU was the obvious choice — we had the VRAM. |

## Performance / Stability

| # | Decision | Choice | Why (one line) |
|---|---|---|---|
| 17 | Flash Attention | Enabled (with safety layers) | 3.5× flatter degradation curve; at 84K context, 28.6 t/s vs 11.0 t/s without FA |
| 18 | FA safety layers | f16 KV + `LLAMA_ATTN_ROT_DISABLE=1` + `-sm layer` | Mitigates open issue #21383 (RTX 3090 + agentic patterns crash path) |
| 19 | GPU split — Qwen | Both GPUs, layer split | 20.5 GB model doesn't fit on a single 24 GB 3090 |
| 20 | GPU split — GLM | Dual GPU, layer split (`-sm layer --tensor-split 1,1`) | MoE PCIe penalty is small (~10-20% vs 30-50% for dense); accepted in exchange for matching Qwen's 96K context window for clean swapping between accuracy and speed without changing context budget |
| 21 | Sampling temperature (Qwen) | temp=0.6 | Qwen-recommended for thinking mode; greedy decoding (temp=0) makes thinking models over-cautious |
| 22 | Sampling temperature (GLM) | temp=0.7, top-p 1.0, min-p 0.01 | GLM-recommended sampling profile; thinking via deepseek reasoning format |
| 23 | Context window | 96K (98,304 tokens) | Fits f16 KV cache in VRAM budget. After ~12K Hermes system prompt overhead and the default 85% compaction trigger (~83.5K — see row 24), the working window per session is roughly ~70K tokens before Hermes summarizes middle turns. |
| 24 | Hermes auto-compaction threshold | 0.85 (Hermes default — kept) | Hermes auto-compacts at 85% of `context_length` by default. With our 96K window that fires at 98,304 × 0.85 = 83,558 tokens. Hermes then summarizes middle turns into the `summary_model` (we point this at the local model itself — no cloud round-trip), drops context back to a smaller working set, and continues. Configurable via `compression.threshold` in `~/.hermes/config.yaml`; we kept the default. The ~15% headroom between the trigger and the hard context limit gives the agent room to finish its current tool-call round without truncation before compaction kicks in. |
| 25 | TDR (Timeout Detection and Recovery) | Disabled (`TdrLevel=0`) | Default 2-second timeout causes Windows-killing cascade on multi-GPU NCCL cleanup |

## Agent Setup

| # | Decision | Choice | Why (one line) |
|---|---|---|---|
| 26 | System prompt philosophy | Karpathy approach — shape thinking, not tools | SOUL.md = how to think; skills = how to use specific tools; model = which tool to pick |
| 27 | Web search backend | Tavily (native Hermes integration) | DuckDuckGo rate-limited + low quality; browser hits CAPTCHAs; Tavily free tier (1,000/mo) works reliably |
| 28 | Mobile access | Telegram gateway (bot via @BotFather) | 2-minute setup; full Hermes access from anywhere; sovereignty extends past the desk |
| 29 | Approval timeout | 300 seconds (was 60 default) | Long enough to review complex terminal commands without auto-timeout; mode stays manual (no auto-approve) |
| 30 | Streaming | Enabled | UX improvement only — no speed change; visible token output during generation |

## Things we did NOT do

| # | Decision | Choice | Why (one line) |
|---|---|---|---|
| 31 | No `--cache-reuse` flag | (rejected) | Hybrid DeltaNet architecture cannot do partial KV cache reuse; llama.cpp logs warning and ignores |
| 32 | No `--grammar` constraints on GLM | (rejected) | Open issue #19068 — infinite loop with tool calling; use `--jinja` autoparser instead |
| 33 | No reasoning budget cap | (rejected) | Thinking traces enable tool-call accuracy; capping them hurts the primary use case |
| 34 | No live `/model` switching during sessions | (offered by Hermes, not used) | Hermes supports live `/model` swapping mid-session; we never used it in production. We restart sessions with the appropriate model for the task instead. |

---

## Decisions that didn't make this cut

A few decisions were considered and either deferred, untested, or determined not to matter at the scale of this experiment:

- **Native Linux on Threadripper** — would eliminate most vLLM penalties (no WSL2, no WDDM tax, native FP8 paths). Future build.
- **Speculative decoding** — not supported for Qwen3.5's hybrid DeltaNet architecture.
- **Single-GPU Qwen3.5 (Q4 + q8_0 KV)** — would save the second GPU for other work, but tight VRAM headroom and increased risk for marginal performance gain.

---

## License

This decision log, and the rest of this repo, is MIT-licensed. See `LICENSE`. Use any of these decisions in your own setup, write up your own findings, and link back if it helped.
