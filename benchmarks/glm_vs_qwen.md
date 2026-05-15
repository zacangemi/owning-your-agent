# GLM-4.7-Flash vs Qwen3.5-27B — Model Comparison Results

*Flash Attention ON · 2× RTX 3090 · llama.cpp b8720 · Hermes Agent*
*Qwen validated 2026-04-13 · GLM tested 2026-04-17*

---

## Summary

Two models tested on identical hardware and software stack, both with flash attention
enabled, both on dual RTX 3090 with 96K context. GLM-4.7-Flash is 2.3-3.4x faster
than Qwen3.5-27B across the full context range, but degrades more steeply with context
depth. Qwen scores slightly higher on tool calling accuracy (97% vs 93%).

---

## Test Configuration

| Setting | Value (both models) |
|---------|-------------------|
| Hardware | 2x NVIDIA RTX 3090 (48 GB VRAM total) |
| Inference | llama.cpp b8720, native Windows, CUDA 12.6 |
| GPU split | Layer split (`-sm layer --tensor-split 1,1`) |
| Context | 98,304 tokens (96K) |
| Flash attention | ON |
| KV cache | f16 |
| Agent | Hermes Agent v0.9.0+ |
| Compaction | 85% threshold (83,558 tokens) |
| Streaming | Enabled |

### Model-Specific Settings

| Setting | Qwen3.5-27B | GLM-4.7-Flash |
|---------|------------|---------------|
| Quantization | UD-Q5_K_XL (18.78 GiB, 6.0 BPW) | UD-Q5_K_XL (20.20 GiB, 5.79 BPW) |
| Temp | 0.6 | 0.7 |
| Top-p | 0.95 | 1.0 |
| Min-p | 0 | 0.01 |
| Repeat penalty | 1.0 | 1.0 |
| Reasoning | `--reasoning-format deepseek` | `--reasoning-format deepseek` |
| ATTN_ROT_DISABLE | Set to 1 (Qwen crash workaround) | Cleared (irrelevant for MLA) |

---

## Architecture Comparison

| Property | Qwen3.5-27B | GLM-4.7-Flash |
|----------|------------|---------------|
| Type | **Dense** (all 27B params active per token) | **MoE** (30B total, ~3-3.6B active per token) |
| Attention | Hybrid: 16 full attention + 48 Gated DeltaNet | MLA (Multi-head Latent Attention), all 47 layers |
| KV cache | K + V stored (6,144 MiB at 96K) | K only, V-less (5,076 MiB, V = 0 MiB) |
| Experts | None | 64 routed, 4 active per token, 1 shared |
| Recurrent state | Yes (150 MiB, DeltaNet layers) | None |
| Native context | 262,144 | 202,752 |
| llama.cpp arch | `qwen35` | `deepseek2` |

### Why GLM Is Faster

GLM activates only ~3B parameters per token (4 of 64 experts + shared expert + attention).
Qwen activates all 27B parameters per token (dense architecture). This ~9x reduction in
active parameters is the primary reason for GLM's speed advantage.

### Why GLM Degrades More Steeply

Amdahl's Law: GLM made the fixed per-token compute cheap (3B vs 27B active params), so
the variable context-dependent cost (attention, KV cache reads, memory traffic) becomes
a larger *proportion* of total work sooner. The context cost is similar in absolute terms
between the models — it's just more visible when the baseline is fast.

---

## VRAM Footprint

| Resource | Qwen3.5-27B | GLM-4.7-Flash |
|----------|------------|---------------|
| GPU0 model buffer | 8,820 MiB | 10,267 MiB |
| GPU1 model buffer | 9,573 MiB | 10,205 MiB |
| CPU mapped | 834 MiB | 208 MiB |
| KV cache total | 6,144 MiB (K + V) | 5,076 MiB (K only, V-less) |
| Recurrent state | 150 MiB | None |
| Total projected | 26,693 MiB (56%) | 27,257 MiB (57%) |
| VRAM headroom | 21 GB free | 21 GB free |

Similar total VRAM despite fundamentally different architectures. GLM's model weights are
larger (more expert parameters) but KV cache is smaller (MLA V-less optimization).

---

## Generation Speed — Regression Analysis

### Fitted Curves

```
GLM-4.7-Flash:  t/s = 101.95 - 0.445 * K
Qwen3.5-27B:    t/s =  35.12 - 0.076 * K    (R² = 0.995)
```

### Speed Comparison Table

| Context | GLM t/s | Qwen t/s | GLM Advantage |
|--------:|--------:|---------:|--------------:|
| 0K (fresh) | 124.0 | 36.1 | 3.4x |
| 14K | 97.5 | 34.0 | 2.9x |
| 27K | 92.6 | 33.1 | 2.8x |
| 38K | 85.9 | 32.2 | 2.7x |
| 50K | 78.1 | 31.3 | 2.5x |
| 60K | 74.6 | 30.5 | 2.4x |
| 66K | 72.6 | 30.1 | 2.4x |
| 73K | 70.0 | 29.6 | 2.4x |
| 84K | 65.5 | 28.6 | 2.3x |
| 92K | 62.9 | — | — |

**GLM is 2.3-3.4x faster than Qwen across the entire context range.** Even at GLM's
worst (62.9 t/s at 92K), it is still 1.8x faster than Qwen's best (35.1 t/s fresh).

### Slope Comparison

```
GLM slope:  -0.445 t/s per 1K tokens (loses 49% over 92K)
Qwen slope: -0.076 t/s per 1K tokens (loses 21% over 87K)
Ratio: Qwen is 5.9x flatter
```

---

## Prompt Processing Speed

| Context Range | GLM prompt t/s | Qwen prompt t/s | GLM Advantage |
|--------------|---------------:|----------------:|--------------:|
| ~14K | 2,405 | 1,601 | 1.5x |
| ~50K | 1,089 | — | — |
| ~80K | ~480 | ~1,280 | Qwen faster |
| ~92K | 784 | — | — |

GLM prompt processing starts faster but degrades more at high context. At ~80K+, Qwen's
prompt eval is actually faster per token.

---

## Reprocessing and Checkpoint Behavior

| Metric | Qwen3.5-27B | GLM-4.7-Flash |
|--------|------------|---------------|
| Forced reprocessing events | **13** | **0** |
| Checkpoint restorations | 24 | 0 |
| Cache evictions | 11 | 11 |
| Canceled tasks | 4 | **12** |

### Qwen Reprocessing Problem

Qwen's hybrid architecture (DeltaNet + attention) causes checkpoint invalidation when
the conversation branch changes. The recurrent state cannot be partially restored,
forcing full re-evaluation from scratch. At 80K+ context, this costs 60-70 seconds of
dead air per event. Total reprocessing cost in the Qwen session: ~5.65 minutes.

### GLM: No Reprocessing Bug

GLM has pure MLA attention on all 47 layers — no recurrent state, no checkpoint
invalidation issue. Zero forced reprocessing events in the entire session.

**However**, GLM still has expensive cold prefills when Hermes changes prompt branches
(e.g., compaction). Task 4097 processed 92,257 tokens from scratch in 117.6 seconds.
The difference: Qwen's reprocessing is forced by architecture; GLM's is caused by
Hermes branch changes (operational, not architectural).

### GLM Compaction Noise

GLM had 4 compaction cycles, each burning through 3 canceled tasks before succeeding.
Hermes repeatedly launched large prompt work, canceled it mid-processing, then sent a
compacted prompt. 12 canceled tasks total vs Qwen's 4. This is Hermes orchestration
behavior, not a GLM inference issue.

After each compaction, GLM speed fully recovered to ~117 t/s (from ~65 t/s pre-compaction).

---

## Tool Calling Quality

### ToolCall-15 Benchmark (prior results)

| Category | Qwen3.5-27B | GLM-4.7-Flash |
|----------|------------|---------------|
| A: Tool Selection | 6/6 (100%) | 6/6 (100%) |
| B: Parameter Precision | 6/6 (100%) | 6/6 (100%) |
| C: Multi-Step Chains | 6/6 (100%) | 6/6 (100%) |
| D: Restraint & Refusal | 5/6 (83%) | 6/6 (100%) |
| E: Error Recovery | 6/6 (100%) | 4/6 (67%) |
| **Total** | **29/30 (97%)** | **28/30 (93%)** |

### Qwen Advantage: Thinking as Planning

Qwen's `<think>` traces act as a built-in planning layer — the model explicitly reasons
through multi-step tool chains before executing. This prevents errors before they happen
(e.g., looking up contacts before sending email, calculating dates step-by-step).

### GLM Advantage: Speed + No Reprocessing

GLM's 2.3-3.4x speed advantage means faster tool call round-trips. Combined with zero
reprocessing stalls, the overall agentic experience feels significantly more responsive.

### Real Session Tool Calling (GLM test)

- ~25 tool calls in the session, all parsed correctly
- Error recovery worked (wrong file path → search → retry)
- No broken `<think>` tags or malformed output
- Final summary was thorough despite 4 compaction events

---

## Time Savings Analysis

### Typical Agentic Response (500 tokens at 50K context)

```
Qwen:  500 / 31.3 = 16.0 seconds
GLM:   500 / 78.1 =  6.4 seconds
Savings: 9.6 seconds per response (2.5x faster)
```

### Large Generation (3000 tokens at 84K context)

```
Qwen:  3000 / 28.6 = 104.9 seconds (1.7 minutes)
GLM:   3000 / 65.5 =  45.8 seconds (0.8 minutes)
Savings: 59.1 seconds (2.3x faster)
```

### Reprocessing Dead Air (per event at 80K)

```
Qwen:  Full reprocess = 60-70 seconds (13 events in session)
GLM:   No reprocessing events (0 seconds)
Savings: 60-70 seconds per event, 5.65 minutes total over session
```

---

## Compaction Behavior

| Metric | Qwen3.5-27B | GLM-4.7-Flash |
|--------|------------|---------------|
| Compaction cycles | 1 (in validated run) | 4 |
| Pre-compaction peak | 87,388 tokens | 92,294 tokens |
| Post-compaction context | 12,710 tokens | ~3,361 tokens |
| Speed recovery | 33.97 t/s (immediate) | 116.87 t/s (immediate) |
| Canceled tasks per cycle | 0-2 | 3 per cycle |

GLM reached deeper context (92K vs 87K) before compaction fired. Post-compaction recovery
was immediate for both models. GLM's compaction cycles were noisier (more cancellations)
but this is a Hermes orchestration issue.

---

## Verdict

### GLM-4.7-Flash Wins On

- **Raw speed**: 2.3-3.4x faster across all context depths
- **No reprocessing stalls**: Zero dead air from checkpoint invalidation
- **Prompt processing**: 1.5x faster at low context
- **Compaction recovery**: Instant return to 117 t/s
- **Perceived responsiveness**: Dramatically better user experience

### Qwen3.5-27B Wins On

- **Speed stability**: 5.9x flatter degradation curve
- **Tool calling accuracy**: 97% vs 93% (ToolCall-15)
- **Thinking/planning**: Auditable reasoning traces prevent errors
- **Deep context performance**: Only 21% speed loss at 87K vs 49% for GLM at 92K

### Recommendation

**GLM-4.7-Flash for daily agentic work** — the speed advantage dominates the user
experience. For most tasks, 93% tool call accuracy is sufficient, and the 2.3-3.4x speed
improvement makes the agent feel alive rather than sluggish.

**Qwen3.5-27B for quality-critical tasks** — when tool call accuracy is paramount
(complex multi-step chains, data integrity across tool calls), Qwen's 97% score and
auditable thinking traces provide higher reliability.

---

## Files

- **Chart**: `glm_vs_qwen_comparison.png`
- **GLM server log**: `testing_output/glm_flash_llama_full.txt`
- **GLM Hermes output**: `testing_output/glm_flash_hermes_full.txt`
- **Qwen server log**: `output_test.txt`
- **Qwen FA results**: `flash_attention_canary_results_2026-04-11.md`
- **Chart source**: `glm_vs_qwen_chart.py`

---

## Experimental Notes

1. **Workloads were not identical.** GLM test was file ingestion + summarization. Qwen test
   was mixed conversation + analysis. Direct token efficiency comparison is not valid without
   controlled identical workloads.

2. **Hermes config showed "Qwen3.5-27B"** during the GLM test because the model name is
   hardcoded in config.yaml. The server was confirmed running GLM via `/v1/models` endpoint
   and server startup logs. Speed data (97-124 t/s vs 34 t/s) conclusively confirms GLM.

3. **Both models tested on llama.cpp b8720** for experimental consistency. Latest stable is
   b8831 (77 releases ahead) with 5-7% MoE speedup and VRAM leak fix. Post-experiment
   update is recommended.

4. **GLM's "Gated Delta Net" kernel activation** was observed in server logs despite GLM not
   being a DeltaNet model. This appears to be llama.cpp enabling a generic optimization path
   for the deepseek2 architecture, not an indication of recurrent state.
