# Owning Your Agent

A local agentic coding stack on dual RTX 3090s. [Hermes Agent](https://github.com/NousResearch/hermes-agent) (NousResearch, MIT) running on a MacBook, connected over Tailscale to a llama.cpp inference server hosting Qwen3.5-27B and GLM-4.7-Flash. Daily-driver agentic work — no cloud, no metering, no rate caps.

- **Tool-calling accuracy:** 97% (Qwen3.5-27B) / 93% (GLM-4.7-Flash) on ToolCall-15.
- **Generation speed:** 28–54 tok/s at production context depth.
- **Hardware envelope:** dual RTX 3090, $4,282 total rig — also runs on a single 3090 with adjustments.

**Narrative writeup:** [blog post on blog.zacharycangemi.com](https://blog.zacharycangemi.com).
**Full decision reasoning:** [`DECISIONS.md`](DECISIONS.md) — 37 decisions, ~5-minute scan.
**Where this stack falls short:** [`LIMITATIONS.md`](LIMITATIONS.md).
**How to reproduce:** [`METHODOLOGY.md`](METHODOLOGY.md).

---

## The hardware envelope — three real paths

| Path | Hardware | Model | Context | Speed |
|---|---|---|---|---|
| **Accessible** | Single RTX 3090 (~$1,050 used) | Qwen3.5-27B Q4_K_XL + q8_0 KV | 32K | ~35–40 t/s |
| **What we built** | Dual RTX 3090 (~$4,282 full rig) | Qwen3.5-27B + GLM-4.7-Flash (UD-Q5_K_XL) | 96K | 28–54 t/s |
| **Frontier-class local** | 4× A6000 / multi-GPU clusters ($20K+) | Kimi K2, GLM-5, large MoE | varies | varies |

The accessible path is what makes this matter for most readers. A 27B-class agent on a single consumer GPU is the actual Overton-window shift — local agentic work is no longer hobbyist-only.

---

## What's in this repo

```
.
├── DECISIONS.md              — every meaningful technical decision + one-line reasoning
├── LIMITATIONS.md            — where this stack falls short of frontier products
├── METHODOLOGY.md            — how to reproduce the benchmarks
├── benchmarks/
│   ├── flash_attention_canary.md       — 3.5× flatter degradation curve, full regression
│   ├── glm_vs_qwen.md                  — head-to-head: speed, reprocessing, tool calling
│   ├── flash_attention_comparison.png
│   └── glm_vs_qwen_comparison.png
├── configs/
│   ├── qwen3.5-27b_dual/launch.bat     — primary accuracy daily driver
│   ├── qwen3.5-27b_single/launch.bat   — accessible single-3090 variant
│   ├── glm-4.7-flash/launch.bat        — speed alternative
│   └── hermes/
│       ├── SOUL.md           — Karpathy-style agent identity (shape thinking, not tools)
│       ├── config.yaml       — Hermes config template
│       └── .env.example      — required env vars (Tavily, Telegram, etc.)
└── LICENSE                   — MIT
```

---

## Quick start

1. **Build llama.cpp from source** with CUDA support. You want build b8720 or newer for the Qwen3.5 thinking-mode tool-call fix (PR #20970) and the multi-GPU Flash Attention fix (PR #19866). Latest stable is fine.
2. **Download the GGUF models** from Unsloth's Hugging Face:
   - `Qwen3.5-27B-UD-Q5_K_XL.gguf` — dual-GPU primary
   - `GLM-4.7-Flash-UD-Q5_K_XL.gguf` — speed alternative
   - `Qwen3.5-27B-Q4_K_XL.gguf` — optional, for the single-GPU variant
3. **Adjust the `C:\` paths** in `configs/*/launch.bat` to match your install locations (the scripts assume `C:\llama-cpp\` and `C:\models\` — change to wherever you put llama.cpp and your GGUF files).
4. **Install Hermes Agent** on your client machine:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   ```
5. **Copy the Hermes config files:**
   - `configs/hermes/.env.example` → `~/.hermes/.env` (fill in your values — at minimum `TAVILY_API_KEY` and update `OPENAI_BASE_URL`)
   - `configs/hermes/config.yaml` → `~/.hermes/config.yaml`
   - `configs/hermes/SOUL.md` → `~/.hermes/SOUL.md`
6. **Launch your inference server** (run the relevant `.bat` from `configs/`).
7. **Run `hermes`** from a terminal on your client machine.

For Hermes setup details beyond what's here, see the [official Hermes docs](https://hermes-agent.nousresearch.com/docs/).

---

## A note on the trajectory

The numbers in this repo are a snapshot from spring 2026. The local stack improved measurably between when this experiment ran and when this writeup published — Qwen 3.6 came out, Hermes shipped major versions, llama.cpp jumped 200+ builds. By the time you read this, the conservative version of every benchmark above is probably tighter than what's documented here.

That's the point. The trajectory matters more than the snapshot. If you replicate this on your own hardware at your own snapshot — write it up. The community needs more honest build logs and fewer demos.

---

## License

MIT. Use anything in this repo. If it helps your work, link back. If you build something interesting, let me know.

**Author:** Zachary Cangemi · [blog](https://blog.zacharycangemi.com) · [portfolio](https://zacharycangemi.com) · [GitHub](https://github.com/zacangemi)
