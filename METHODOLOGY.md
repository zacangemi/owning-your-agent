# Methodology — How to Reproduce This Benchmark

This document explains how the benchmarks in `benchmarks/` were run, so a reader can replicate them on their own hardware. The point isn't to copy our specific numbers — by the time you run this, the open-source stack will have shifted — it's to apply the same methodology to a current snapshot.

---

## Hardware (what we used)

- 2× NVIDIA RTX 3090 Founders Edition (48 GB combined VRAM)
- AMD Ryzen 9 7950X3D (16C / 32T)
- 96 GB DDR5-6000 (FCLK 2000 MHz, 1:1 ratio with memory controller)
- Samsung 990 PRO 2 TB NVMe (Gen 4)
- Windows 11 Pro (inference server)
- MacBook Pro M3 Pro (Hermes Agent client)
- Tailscale mesh VPN between Mac and Windows server

## Software stack (versions at time of testing)

- **llama.cpp:** b8720 (built from source, CUDA 12.6, sm_86)
- **NVIDIA driver:** 581.29
- **Hermes Agent:** v0.7.0 → v0.9.0 (multiple version bumps during the experiment)
- **vLLM:** 0.18.0 (in WSL2, for the Qwen FP8 reference run only)
- **ToolCall-15:** github.com/stevibe/ToolCall-15 (MIT, commit at time of testing)

## Models tested

- Qwen3.5-27B Unsloth UD-Q5_K_XL (production — quality)
- GLM-4.7-Flash Unsloth UD-Q5_K_XL (production — speed)
- Qwen3-Coder-Next 80B Q4_K_XL (eliminated after Phase 1)
- Qwen3.5-27B FP8 (official Qwen, vLLM reference only)

---

## Three-phase test plan

### Phase 1: Synthetic benchmark (ToolCall-15)

**What it measures:** Tool-call accuracy across 15 scenarios in 5 categories — tool selection, parameter precision, multi-step chains, restraint & refusal, error recovery. Each scenario scores 0-2.

**How to run:**

1. Clone ToolCall-15:
   ```bash
   git clone https://github.com/stevibe/ToolCall-15
   cd ToolCall-15
   npm install
   ```
2. Configure `.env` to point at your llama.cpp endpoint:
   ```
   LLAMACPP_HOST=http://<your-host>:8000
   LLM_MODELS=llamacpp:<your-model-file>.gguf
   MODEL_REQUEST_TIMEOUT_SECONDS=120
   ```
3. Launch your inference server using a launch script from `../configs/`
4. Start ToolCall-15: `npm run dev`
5. Open the dashboard, select your model, run all 15 scenarios

**What to record per model:**
- Category scores (A through E)
- Per-scenario notes (especially failures and partials)
- Generation speed (tok/s) during the run
- Total benchmark wall-clock time

**Important: temperature matters for thinking models.** ToolCall-15 defaults to temp=0 (greedy decoding) as standard benchmark practice. For thinking models like Qwen3.5, ALSO run at the model's recommended temperature (Qwen: 0.6) — greedy decoding makes thinking models over-cautious and can change the score. We documented a +3 point improvement on Qwen3.5-27B going from temp=0 to temp=0.6.

### Phase 2: Hermes functional tests

**What it measures:** Whether the agent system itself wires up correctly through Hermes — not just whether the model handles tools in isolation. ToolCall-15 uses mocked tools; this phase exercises the real Hermes tool pipeline.

**Five tests:**

1. `"Hello, what model are you?"` — connection sanity check.
2. `"Read the file [name]"` — `read_file` tool wired correctly, content returned.
3. `"Create hello.py that prints hello world"` — `write_file` tool, file actually appears on disk with correct content.
4. `"Run ls -la"` — `terminal` tool, command executed, output returned and parsed correctly.
5. `"Read hello.py, add error handling, run it"` — multi-step chain: read → edit → execute, with the agent tracking state across tool calls.

**What to record:**
- Pass/fail per test
- Any tool-call parsing errors (look for malformed JSON, unparsed XML tags)
- Multi-turn state handling quality (does the agent remember the file it just wrote?)
- Time to complete each test
- Subjective feel of the interaction (speed, responsiveness, clarity of reasoning)

### Phase 3: Realistic agentic coding

**What it measures:** Daily-use viability. Whether the agent produces useful work on real tasks, not just benchmark scenarios. This is the phase that earns the "daily-driver viable" claim.

**Three tasks:**

**Task A — Build a Python script.** Ask the agent to write a script that processes data in some way. We used: read .md result files from a directory, extract scores and speeds via regex, output a comparison table. Real edge cases (tilde paths, decimal parsing) emerge naturally.

What to record: number of tool calls, number of debug-fix cycles (the loop closing without your intervention is the point), time to working script, code quality assessment.

**Task B — Research pipeline.** Ask the agent to search a real API. arxiv works well — public, no key required, real-world messy. We asked: "search arxiv for papers about tool calling in LLMs published in 2026."

What to record: number of search iterations, whether the agent refines its queries when initial results are poor, whether it finds real verifiable data.

**Task C — Multi-file project.** Ask the agent to build a small project (Flask API works), seed it with intentional bugs, ask it to find and fix them, then write unit tests. This stresses cross-file navigation and debugging.

What to record: ability to navigate across files, debug runtime errors, write tests that actually run.

**General notes:**
- Tool call count
- Debug-fix cycles (closing the loop without your intervention is the daily-use signal)
- Total time per task
- Code quality (subjective but documented — junior/mid/senior level)
- Context consumed (token count after each task)
- Anything that surprised you

---

## How we measured generation speed at depth

Generation speed degrades linearly with context fill. To measure this on your hardware:

1. Launch your inference server with a large context window (we used 96K = 98,304 tokens).
2. Run a real agentic session — file ops, code edits, web searches, multi-step tasks — until the context naturally grows.
3. Record `eval time` and `tokens` from llama.cpp's per-request `slot print_timing` output.
4. Plot t/s vs context size in K tokens.
5. Fit a linear regression: `speed = a - b × K` (K = thousands of tokens).

For Qwen3.5-27B with Flash Attention on, our 50-data-point production session fits: `t/s = 35.12 - 0.076 × K` with R² = 0.995. The published chart in `benchmarks/flash_attention_comparison.png` was produced from this regression; the formula above is enough to plot your own data on your own hardware.

## How we measured the Flash Attention impact

Documented in detail in `benchmarks/flash_attention_canary.md`. Summary:

1. Run a canary session with `--flash-attn on` and record t/s vs context.
2. Compare against a baseline session with `--flash-attn off` (or use historical baseline data if you have it).
3. Fit a linear regression to both.
4. Compare the slopes. We measured FA on: -0.083 t/s per 1K tokens; FA off: -0.280 t/s per 1K tokens. Ratio: 3.37× flatter degradation curve with FA enabled.

**Safety layers required** for FA on Qwen3.5 + multi-GPU RTX 3090: `--cache-type-k f16 --cache-type-v f16`, environment variable `LLAMA_ATTN_ROT_DISABLE=1`, and layer split mode (`-sm layer`). Without these, you may hit the still-open crash path documented in llama.cpp issue #21383.

## How we compared GLM vs Qwen

Documented in detail in `benchmarks/glm_vs_qwen.md`. Summary:

- Same hardware, same software stack, same context window, same Flash Attention setting.
- Different model-specific settings (sampling, KV cache compatibility per architecture — MLA requires f16).
- Generation speed measured across the full context range on real agentic workloads.
- Tool-call accuracy from prior ToolCall-15 results.
- Reprocessing behavior tracked from llama.cpp server logs (look for "forcing full prompt re-processing" messages).

The comparison is honest about its limits: workloads were NOT controlled for identical content. GLM was tested on file ingestion + summarization; Qwen on mixed conversation + analysis. Direct token-efficiency comparison is not valid without controlled identical workloads — but the speed-vs-context curves are independent of workload content and ARE comparable.

---

## Bench-running discipline

A few practices that made these benchmarks usable in practice:

- **Run each test in a fresh Hermes session.** Memory and skills carry over but conversation context starts clean. This matters for speed-at-depth measurements — you want the context to grow naturally during the session, not start at 50K from carryover.
- **Use `tmux` for any long-running session.** SSH pipe break (laptop sleeping, network drop) can hang multi-GPU NCCL cleanup and freeze the inference server. `tmux` keeps the session alive.
- **Pin llama.cpp version during a benchmark run.** Builds ship 1-3× per week; mid-experiment updates can change the answers. Note the build number in your results.
- **Save the server log.** llama.cpp logs `slot print_timing` for every request. Don't trust eyeball measurements — extract from logs.
- **Verify GPU stability before measuring.** Disable any active background workloads that touch the GPU.

---

## Reproducibility limits

Some properties of our benchmarks are not perfectly reproducible:

- **Sampling temperature > 0 introduces non-determinism.** Run multiple seeds and report mean/variance for statistical significance.
- **llama.cpp version-to-version changes can shift results by 5-10%.** Pin the build for any benchmark run. Note version drift between runs.
- **Hermes Agent version changes can affect tool-call parsing.** Pin the version. Hermes is releasing weekly — by the time you run this, the version we used will be several releases old.
- **Hardware variations (RAM speed, motherboard PCIe topology, thermal limits) can shift speed measurements ~10-20%** even on "identical" GPUs in different rigs.

The right framing: these benchmarks document what's achievable on a specific hardware + software snapshot in spring 2026. Use the methodology to verify on your own setup at your own snapshot. The trajectory matters more than the specific numbers.
