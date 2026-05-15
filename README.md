# Owning Your Agent

A local agentic coding stack on dual RTX 3090s. [Hermes Agent](https://github.com/NousResearch/hermes-agent) (NousResearch, MIT) running on a MacBook, connected over Tailscale to a llama.cpp inference server hosting Qwen3.5-27B and GLM-4.7-Flash. Daily-driver agentic work — no cloud, unlimited tokens, no API caps!

- **Tool-calling accuracy:** 97% (Qwen3.5-27B) / 93% (GLM-4.7-Flash) on ToolCall-15.
- **Generation speed (Flash Attention enabled):** Qwen3.5-27B 28–36 tok/s · GLM-4.7-Flash 63–124 tok/s · measured at production context depth.
- **Hardware envelope:** dual RTX 3090, $4,282 total rig.

**Narrative writeup:** [blog post on blog.zacharycangemi.com](https://blog.zacharycangemi.com).
**Full decision reasoning:** [`DECISIONS.md`](DECISIONS.md) — 34 decisions, ~5-minute scan.
**Where this stack falls short:** [`LIMITATIONS.md`](LIMITATIONS.md).
**How to reproduce:** [`METHODOLOGY.md`](METHODOLOGY.md).

---

## The hardware envelope — three real paths

| Path | Hardware | Model | Quant | Context | Speed (FA on) |
|---|---|---|---|---|---|
| **What we built — accuracy** | Dual RTX 3090 (~$4,282 full rig) | Qwen3.5-27B | UD-Q5_K_XL | 96K | 28–36 t/s |
| **What we built — speed** | Same dual-3090 rig | GLM-4.7-Flash | UD-Q5_K_XL | 96K | 63–124 t/s |
| **Frontier-class local** | 2× RTX PRO 6000 Blackwell (192 GB) or 4× RTX 5090 (128 GB) — $15–25K rig | Kimi K2, GLM-5, large MoE | varies | varies | varies |

A note on the single-3090 path: a single-RTX-3090 setup is a documented community pattern (e.g., @sudoingX runs Qwen3.5 on single 3090s), but **we did not directly verify the single-3090 configuration in this experiment**. We focused on the dual-3090 setup because it gave us 96K context and the headroom to evaluate Qwen and GLM side-by-side. A dedicated single-3090 writeup is on the roadmap as a follow-up project.

The two-model split in the dual-3090 path isn't redundancy — it's a deliberate two-model setup. **GLM-4.7-Flash is dramatically faster** because it's MoE: only ~3B parameters are active per token (out of ~30B total) vs Qwen's full 27B-dense activation. That's a ~9× reduction in fixed per-token compute, which is why GLM hits ~124 t/s on fresh context and Qwen tops out around ~36 t/s. **GLM also degrades faster with context depth** — Amdahl's Law: once the per-token compute is cheap, the context-dependent cost (attention, KV cache reads, memory traffic) becomes a larger proportion of total work sooner. Net result across the full 96K window: GLM is 2.3–3.4× faster than Qwen at every context depth, but the gap narrows as context fills. **GLM-4.7-Flash is the speed driver in our setup** — its 2.3–3.4× advantage makes the agent feel responsive in extended sessions, and 93% ToolCall-15 accuracy handles most agentic work. **Qwen3.5-27B is the heavy hitter for accuracy-critical work** — slower, but its `<think>` traces, 5.9× flatter degradation curve, and 97% tool-call accuracy buy reliability when output quality matters more than turn-around time. Both models share the same 96K context budget, so picking between them per task — by restarting the inference server with the appropriate launch script — keeps the working-space math consistent across either choice.

---

## What's in this repo

```
.
├── DECISIONS.md              — 34 technical decisions + one-line reasoning each
├── LIMITATIONS.md            — where this stack falls short of frontier products
├── METHODOLOGY.md            — how to reproduce the benchmarks
├── benchmarks/
│   ├── flash_attention_canary.md       — 3.5× flatter degradation curve, full regression
│   ├── glm_vs_qwen.md                  — head-to-head: speed, reprocessing, tool calling
│   ├── flash_attention_comparison.png
│   └── glm_vs_qwen_comparison.png
├── configs/
│   ├── qwen3.5-27b_dual/launch.bat     — accuracy heavy hitter
│   ├── glm-4.7-flash/launch.bat        — speed daily driver
│   └── hermes/
│       ├── SOUL.md           — Karpathy-style agent identity (shape thinking, not tools)
│       ├── config.yaml       — Hermes config template
│       └── .env.example      — required env vars (Tavily, Telegram, etc.)
└── LICENSE                   — MIT
```

---

## Quick start

This setup uses **two machines**: an **inference server** (Windows or Linux with the GPUs) running llama.cpp, and a **client machine** (a Mac, in our setup) running the Hermes Agent — NousResearch's open-source agent application. The Hermes Agent is the *client app* that orchestrates your local LLM into a working agent; it is **not** itself a model. The two machines connect over Tailscale (encrypted mesh VPN, no port forwarding required).

1. **Build llama.cpp from source** (on the inference server) with CUDA support. You want build b8720 or newer for the Qwen3.5 thinking-mode tool-call fix (PR #20970) and the multi-GPU Flash Attention fix (PR #19866). Latest stable is fine.
2. **Download the GGUF models** from Unsloth's Hugging Face onto the inference server:
   - [`unsloth/Qwen3.5-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) — pull `UD-Q5_K_XL` for the dual-GPU primary
   - [`unsloth/GLM-4.7-Flash-GGUF`](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) — pull `UD-Q5_K_XL` for the speed alternative
3. **Adjust the `C:\` paths** in `configs/*/launch.bat` to match your install locations (the scripts assume `C:\llama-cpp\` and `C:\models\` — change to wherever you put llama.cpp and your GGUF files).
4. **Launch the inference server** by running the relevant `.bat` from `configs/`. It listens on port 8000.
5. **Install the Hermes Agent client app on the Mac** — this is the agent application that connects to your inference server. It is **not** a model; the models stay on the inference server you set up in steps 1-4:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   ```
6. **Copy the Hermes config files** to `~/.hermes/` on the Mac:
   - `configs/hermes/.env.example` → `~/.hermes/.env`, then fill in:
     - `OPENAI_BASE_URL` — your inference server's address (Tailscale IP, LAN IP, or `localhost`)
     - `TAVILY_API_KEY` — **enables web browsing for the agent** (lets it search the web during tool calls). Free tier at [tavily.com](https://tavily.com) covers 1,000 searches/month
     - `TELEGRAM_BOT_TOKEN` (optional) — enables mobile access via a Telegram bot
   - `configs/hermes/config.yaml` → `~/.hermes/config.yaml`
   - `configs/hermes/SOUL.md` → `~/.hermes/SOUL.md`
7. **Run `hermes`** from a terminal on the Mac — the agent boots, connects to your inference server, and you're ready.

For Hermes Agent setup details beyond what's here, see the [official Hermes Agent docs](https://hermes-agent.nousresearch.com/docs/).

---

## A note on the trajectory

The numbers in this repo are a snapshot from spring 2026. The local stack improved measurably between when this experiment ran and when this writeup published — Qwen 3.6 came out, Hermes shipped major versions, llama.cpp jumped 200+ builds. By the time you read this, the conservative version of every benchmark above is probably tighter than what's documented here.

That's the point. The trajectory matters more than the snapshot. If you replicate this on your own hardware at your own snapshot — write it up. The community needs more honest build logs and fewer demos.

---

## License

MIT. Use anything in this repo. If it helps your work, link back. If you build something interesting, let me know.

**Author:** Zachary Cangemi · [blog](https://blog.zacharycangemi.com) · [portfolio](https://zacharycangemi.com) · [GitHub](https://github.com/zacangemi)
