# Decisions Log

Every meaningful technical decision made while building this local agentic stack. One line of reasoning per row; the longer narrative is in the [blog post](https://blog.zacharycangemi.com). Each decision is tagged `general` (applies to any local agentic setup) or `hardware-specific` (specific to this 2× RTX 3090 rig).

**The 27B-dense sweet spot.** This entire stack is built around the empirical finding that 27B-class dense models are the right choice for local agentic tool-calling work. Smaller models (<14B) lack the parameter budget for simultaneous intent understanding + tool selection + parameter generation + restraint, and default to curl/Python fallbacks. Larger models (70B+) have strong priors that override tool-call signals — they "know" the answer and skip the tools. 27B-dense gives you enough capacity for multi-step reasoning without overriding tool use, and it fits on consumer hardware (1-2 RTX 3090s) without aggressive quantization that costs accuracy. Most decisions below follow from this anchor.

Read time: ~5 minutes.

---

## Hardware / Platform

| # | Decision | Choice | Why (one line) | Tag |
|---|---|---|---|---|
| 1 | Inference platform | Windows 11 + llama.cpp native | Avoids WSL2 overhead and virtualization tax; full GPU control | hardware-specific |
| 2 | Agent client OS | macOS (separate from inference) | Mac is the daily driver; Hermes runs comfortably; Tailscale connects to the inference server | general |
| 3 | Network between agent + inference | Tailscale mesh VPN | Zero port forwarding, encrypted, works on any network, no public IP exposure | general |

## Model Selection

| # | Decision | Choice | Why (one line) | Tag |
|---|---|---|---|---|
| 4 | Primary model (accuracy) | Qwen3.5-27B (Unsloth UD-Q5_K_XL) | 100% ToolCall-15 at temp 0.6; thinking traces act as a built-in planning layer | general |
| 5 | Speed model (alternative) | GLM-4.7-Flash (Unsloth UD-Q5_K_XL) | 2.3–3.4× faster than Qwen; 93% ToolCall-15; no reprocessing bug (pure MLA attention) | general |
| 6 | Why Qwen3-Coder-Next 80B was ruled out | (eliminated — counterintuitive teaching moment) | 77% on ToolCall-15 vs Qwen3.5-27B's 97% — large model's strong priors caused tool *avoidance* (the model "knows" the answer and skips the tools), and MoE routing diluted the tool-call signal further. Bigger ≠ better for agentic work. | general |
| 7 | Why Claude-distilled fine-tunes were ruled out | (eliminated — common newbie download) | Hobbyist Claude-distilled models on HuggingFace (e.g., Jackrong's Qwen3.5-Claude-4.6-Opus-Reasoning-Distilled series) are fine-tunes of base Qwen/Llama on Claude reasoning outputs — optimized for reasoning *brevity* (shorter CoT chains), not for tool calling. For agentic work, the base Qwen3.5-27B preserves the careful step-by-step tool-call reasoning the model was originally trained for. **Important distinction:** these hobbyist fine-tunes ≠ full lab-distillation pipelines (Z.ai's GLM family, Moonshot's Kimi K2, DeepSeek's training process), which use teacher-student training plus serious RLHF/RLAIF and produce competitive frontier-grade models. We avoided the hobbyist variants on consumer hardware. | general |

### Models we evaluated and ruled out

We evaluated 7+ models across Qwen, Hermes-4.3, Devstral, Mistral Small, GPT-OSS, and GLM families. After narrowing, three additional candidates were eliminated for the reasons below (the two we *kept* and the two we explicitly called out above are detailed in rows 4-7):

| Candidate | Why ruled out |
|---|---|
| Hermes-4.3-36B | 35→15 t/s degradation across context — dealbreaker for agentic sessions that grow large |
| Qwen3-Coder-30B-A3B | Too weak for Hermes (confirmed by Hermes maintainer); small models default to curl/Python fallbacks |
| SGLang as backend | Worse Qwen3.5 support than vLLM; no native tool-call parser |

## Inference Engine

| # | Decision | Choice | Why (one line) | Tag |
|---|---|---|---|---|
| 8 | Inference engine | llama.cpp (native Windows, built from source, CUDA 12.6) | 1.8× faster generation, 4× faster prompt eval vs vLLM on this hardware | hardware-specific |
| 9 | Why vLLM was ruled out (for this setup) | (eliminated for consumer Ampere + Windows) | The cleanest framing: **vLLM's FP8 path uses Marlin kernels** (software FP8 dequant on Ampere — RTX 3090 has no native FP8 tensor cores, so the dequant happens in software at ~10-15% throughput cost). On top of that: **WSL2 passthrough overhead** (`pin_memory=False`, GPU virtualization layer), **no GPU peer-to-peer** (TP communication crosses PCIe every layer), **forced `--enforce-eager`** (CUDA graphs crash on Ampere + FP8 + Qwen3.5 hybrid), **WDDM driver tax** (~1.4 GB per GPU reserved for Windows display), **PCIe x8 bifurcation** (vs x16), **no torch.compile**. Seven compounding penalties. Net result: vLLM at 15.5 t/s on FP8 vs llama.cpp at 28.5 t/s on Unsloth UD-Q5_K_XL. **Better consumer-hardware practice: llama.cpp + Unsloth UD quants on native Windows. vLLM's design assumes Ada/Hopper datacenter on native Linux** — not what we have. | hardware-specific |
| 10 | llama.cpp build version | b8720 → b8800+ during experiment | PR #20970 (Qwen3.5 thinking + tool-call) and PR #19866 (FA multi-GPU fix) both critical for this workload | general |

## Quantization / KV Cache

| # | Decision | Choice | Why (one line) | Tag |
|---|---|---|---|---|
| 11 | Model quantization | Unsloth Dynamic Q5_K_XL | UD preserves router (f32) and attention precision; only FFN compressed — exactly where tool calling needs precision | general |
| 12 | Why not Q8_0 | (alternative considered) | Q8 99.9% vs UD-Q5 99.4% of BF16 — unmeasurable gap; UD-Q5 saves 9.5 GB | general |
| 13 | KV cache quantization | f16 (no quantization) | **No need to quantize: both models on dual GPUs leave 21+ GB VRAM free at 96K context.** Architecture also keeps the KV cache naturally small: Qwen3.5-27B is hybrid — only **16 of 64 layers** use full attention (the other 48 are Gated DeltaNet with fixed-size recurrent state, no KV cache); within those 16 attention layers, GQA gives 4 KV heads vs 24 query heads (6× reduction). GLM-4.7-Flash uses MLA (Multi-head Latent Attention), which natively compresses K+V into a small latent representation — and **MLA requires f16** because it can't be quantized further without disproportionate quality loss. Quantizing the KV cache would have saved ~2 GB of headroom we didn't need, in exchange for a quality variable we didn't want. | hardware-specific |
| 14 | CPU offloading | None (full GPU) | **Plenty of VRAM headroom on dual GPUs (21+ GB free at 96K context) — no reason to push layers to CPU.** Architecture also matters here: dense 27B models bottleneck on every token if any layers are CPU-offloaded (every token activates every layer); MoE benefits from CPU offload only when active params per token are small enough that the bottleneck doesn't dominate. With our setup, full GPU was the obvious choice — we had the VRAM. | general |

## Performance / Stability

| # | Decision | Choice | Why (one line) | Tag |
|---|---|---|---|---|
| 15 | Flash Attention | Enabled (with safety layers) | 3.5× flatter degradation curve; at 84K context, 28.6 t/s vs 11.0 t/s without FA | general |
| 16 | FA safety layers | f16 KV + `LLAMA_ATTN_ROT_DISABLE=1` + `-sm layer` | Mitigates open issue #21383 (RTX 3090 + agentic patterns crash path) | hardware-specific |
| 17 | GPU split — Qwen | Both GPUs, layer split | 20.5 GB model doesn't fit on a single 24 GB 3090 | hardware-specific |
| 18 | GPU split — GLM | Dual GPU, layer split (`-sm layer --tensor-split 1,1`) | MoE PCIe penalty is small (~10-20% vs 30-50% for dense); accepted in exchange for matching Qwen's 96K context window for clean swapping between accuracy and speed without changing context budget | hardware-specific |
| 19 | Sampling temperature (Qwen) | temp=0.6 | Qwen-recommended for thinking mode; greedy decoding (temp=0) makes thinking models over-cautious | general |
| 20 | Sampling temperature (GLM) | temp=0.7, top-p 1.0, min-p 0.01 | GLM-recommended sampling profile; thinking via deepseek reasoning format | general |
| 21 | Context window | 96K (98,304 tokens) | Fits f16 KV cache in VRAM budget; ~12K Hermes system prompt overhead leaves ~86K working space | hardware-specific |
| 22 | TDR (Timeout Detection and Recovery) | Disabled (`TdrLevel=0`) | Default 2-second timeout causes Windows-killing cascade on multi-GPU NCCL cleanup | hardware-specific |

## Agent Setup

| # | Decision | Choice | Why (one line) | Tag |
|---|---|---|---|---|
| 23 | System prompt philosophy | Karpathy approach — shape thinking, not tools | SOUL.md = how to think; skills = how to use specific tools; model = which tool to pick | general |
| 24 | Web search backend | Tavily (native Hermes integration) | DuckDuckGo rate-limited + low quality; browser hits CAPTCHAs; Tavily free tier (1,000/mo) works reliably | general |
| 25 | Mobile access | Telegram gateway (bot via @BotFather) | 2-minute setup; full Hermes access from anywhere; sovereignty extends past the desk | general |
| 26 | Approval timeout | 300 seconds (was 60 default) | Long enough to review complex terminal commands without auto-timeout; mode stays manual (no auto-approve) | general |
| 27 | Streaming | Enabled | UX improvement only — no speed change; visible token output during generation | general |

## Things we did NOT do (and why)

| # | Decision | Choice | Why (one line) | Tag |
|---|---|---|---|---|
| 28 | No `--cache-reuse` flag | (rejected) | Hybrid DeltaNet architecture cannot do partial KV cache reuse; llama.cpp logs warning and ignores | general |
| 29 | No `--grammar` constraints on GLM | (rejected) | Open issue #19068 — infinite loop with tool calling; use `--jinja` autoparser instead | general |
| 30 | No reasoning budget cap | (rejected) | Thinking traces enable tool-call accuracy; capping them hurts the primary use case | general |
| 31 | No Q4 model quantization on the dual-GPU path | (rejected for accuracy daily driver) | 10-15% speed gain not worth tool-call accuracy risk on the accuracy daily driver. Q4_K_XL only used on the single-GPU accessibility variant where the VRAM constraint forces it. | general |
| 32 | No live `/model` switching during sessions | (offered by Hermes, not used) | Hermes supports live `/model` swapping mid-session; we never used it in production. We restart sessions with the appropriate model for the task instead. Listed for honesty — readers should know this is available even though we don't lean on it. | general |

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
