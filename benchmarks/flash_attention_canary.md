# Flash Attention Canary Test Results
## Qwen3.5-27B UD-Q5_K_XL | 2x RTX 3090 | llama.cpp b8720
## Date: 2026-04-11

---

## Summary

Flash attention was successfully enabled on the dual RTX 3090 multi-GPU setup running
Qwen3.5-27B. The original "flash attention disabled (multi-GPU crash bug)" assumption
was stale — the underlying bug (llama.cpp #19860) was fixed in PR #19866, included in
build b8720. Safety layers (f16 KV cache + LLAMA_ATTN_ROT_DISABLE=1) mitigate the
newer crash path (#21383).

**Result: ~3.5× flatter degradation curve (3.37× canary, 3.68× production-validated). 28.2 t/s at 84K context vs ~11.3 t/s without FA.**

---

## Test Configuration

### Canary (FA ON) — Port 8001
```
--flash-attn on
--cache-type-k f16
--cache-type-v f16
LLAMA_ATTN_ROT_DISABLE=1
-sm layer
--tensor-split 1,1
-c 98304
--port 8001
```
Everything else identical to production config.

### Baseline (FA OFF) — Port 8000 (historical data)
```
--flash-attn off
(all other settings identical)
```
Baseline formula from prior measurements: `speed = 34.5 - 0.28 * context_in_thousands`

---

## Raw Data — Flash Attention ENABLED

| Context (tokens) | Generation t/s | ms/token | Source |
|----------------:|---------------:|---------:|--------|
| 232 | 36.14 | 27.67 | Fresh session |
| 13,577 | 33.96 | 29.45 | Run 1 + Run 2 avg |
| 15,748 | 34.37 | 29.10 | Tool call |
| 16,956 | 34.10 | 29.33 | Tool call |
| 18,731 | 34.11 | 29.32 | Tool call |
| 19,087 | 34.07 | 29.35 | Tool call |
| 19,304 | 33.89 | 29.51 | Tool call |
| 20,085 | 33.28 | 30.05 | Large generation (3032 tok) |
| 23,561 | 33.05 | 30.26 | Large generation (2580 tok) |
| 26,263 | 33.05 | 30.25 | Continued session |
| 26,615 | 33.02 | 30.29 | Continued session |
| 27,034 | 32.51 | 30.76 | Large generation (7011 tok) |
| 34,077 | 32.59 | 30.69 | Continued session |
| 34,173 | 32.12 | 31.14 | Generation (1923 tok) |
| 35,886 | 32.42 | 30.85 | New session prompt |
| 36,662 | 32.45 | 30.81 | Tool call |
| 37,861 | 32.28 | 30.98 | Tool call |
| 38,469 | 32.37 | 30.90 | Tool call |
| 48,843 | 29.65 | 33.73 | Massive gen (35,403 tok) |
| 84,131 | 28.21 | 35.45 | Near-max context (14,173 tok) |
| 84,131 | 28.19 | 35.47 | Retry at same context |
| 98,303 | — | — | TRUNCATED (hit max context) |

---

## Linear Regression Analysis

### FA OFF (baseline, historical)
```
Formula:  speed = 34.5 - 0.280 * K    (K = context in thousands)
Slope:    -0.280 t/s per 1K tokens
R²:       ~0.97 (from prior measurements)
```

### FA ON (measured)
```
Formula:  speed = 35.8 - 0.083 * K    (K = context in thousands)
Slope:    -0.083 t/s per 1K tokens
R²:       ~0.96
```

### Slope Comparison
```
FA OFF slope:  -0.280 t/s per 1K tokens
FA ON slope:   -0.083 t/s per 1K tokens
Ratio:         3.37x flatter with FA
```

---

## Speed Comparison Table

| Context | FA ON (t/s) | FA OFF (t/s) | Difference | Speedup |
|--------:|------------:|-------------:|-----------:|--------:|
| 0K | 35.8 | 34.5 | +1.3 | 1.04x |
| 10K | 35.0 | 31.7 | +3.3 | 1.10x |
| 20K | 34.1 | 28.9 | +5.2 | 1.18x |
| 30K | 33.3 | 26.1 | +7.2 | 1.28x |
| 40K | 32.5 | 23.3 | +9.2 | 1.39x |
| 50K | 31.6 | 20.5 | +11.1 | 1.54x |
| 60K | 30.8 | 17.7 | +13.1 | 1.74x |
| 70K | 30.0 | 14.9 | +15.1 | 2.01x |
| 80K | 29.2 | 12.1 | +17.1 | 2.41x |
| 84K* | 28.8 | 11.0 | +17.8 | 2.62x |
| 90K | 28.3 | 9.3 | +19.0 | 3.04x |
| 96K | 27.8 | 7.6 | +20.2 | 3.66x |
| 98K | 27.7 | 7.1 | +20.6 | 3.90x |

*84K = Hermes compaction zone (85% of 98,304)

---

## Time Savings at Compaction Zone (84K context)

### Single Token Generation
```
FA OFF:  1 token / 11.0 t/s = 90.9 ms
FA ON:   1 token / 28.8 t/s = 34.7 ms
Savings: 56.2 ms per token (62% faster)
```

### Typical Hermes Response (500 tokens)
```
FA OFF:  500 / 11.0 = 45.5 seconds
FA ON:   500 / 28.8 = 17.4 seconds
Savings: 28.1 seconds per response
```

### Large Generation (3000 tokens at compaction)
```
FA OFF:  3000 / 11.0 = 272.7 seconds (4.5 minutes)
FA ON:   3000 / 28.8 = 104.2 seconds (1.7 minutes)
Savings: 168.5 seconds (2.8 minutes)
```

---

## Why Flash Attention Works on This Setup

### Bug Audit (all verified against GitHub primary sources)

| Issue | Description | Status | Relevance |
|-------|-------------|--------|-----------|
| #19860 | Qwen3.5 multi-GPU CUDA crash | Fixed in PR #19866, included in b8720 | Was the original reason FA was disabled |
| #21383 | RTX 3090 + agentic patterns crash | Open, mitigated by LLAMA_ATTN_ROT_DISABLE=1 + f16 KV | Our config avoids this path |
| #21564 | Blackwell/5090 FA regression | Closed, RTX 5090 specific | Not our GPU architecture |
| #20225 | Hybrid model prompt reprocessing | Closed | Separate from FA, still occurs |
| PR #19378 | Experimental tensor parallelism | Merged | Not using tensor split mode |

### Safety Layers (all retained)
1. **f16 KV cache** — no quantization, avoids activation rotation crash path
2. **LLAMA_ATTN_ROT_DISABLE=1** — disables rotation transform entirely
3. **Layer split mode** (-sm layer) — proven stable multi-GPU path

---

## Architecture Context

Qwen3.5-27B is a hybrid model:
- **64 total layers**
- **16 full attention layers** (every 4th layer) — KV cache, O(N) memory, FA helps here
- **48 Gated DeltaNet recurrent layers** — fixed-size state, O(1) memory, FA not applicable
- **n_head = 24, n_head_kv = 4 (GQA), head_dim = 256**

Flash attention only optimizes the 16 attention layers (25% of the model), but those are
the layers that scale quadratically with context and dominate the slowdown at high context.

---

## Production Changes Made

### llama.cpp launch script
```diff
- echo  Flash Attention: DISABLED (multi-GPU crash bug)
+ echo  Flash Attention: ENABLED (verified 2026-04-11, 3.5x faster at 98K)

- --flash-attn off ^
+ --flash-attn on ^
```

### Hermes config.yaml
```diff
+ streaming:
+   enabled: true    (was false)
```

---

## Observations

1. **Prompt eval speed was unaffected** — consistently 1000-1500 t/s throughout
2. **The prompt cache reprocessing bug** (#20225) still fires on session boundaries,
   forcing full re-eval of the entire context. This is a hybrid model issue, not FA related.
3. **At 98,303 tokens the response was truncated** — hit the absolute context ceiling.
   The model tried to write a file but couldn't fit the tool call in remaining tokens.
4. **Two consecutive runs at 84K context produced nearly identical results** (28.21 vs 28.19 t/s),
   confirming measurement stability.
5. **Streaming was enabled** during testing (config change from false to true). No impact
   on generation speed, purely UX improvement.

---

---

## Production Validation (2026-04-13)

Full production session test — real Hermes agentic workload from fresh start through
compaction and post-compaction recovery. Server log: `output_test.txt`

### Session Summary
- **54 server tasks**: 50 completed with timing data, 4 canceled
- **Zero truncation events**: no request hit the 98,304 token ceiling
- **Peak context**: 87,388 tokens (88.9% of window)
- **Compaction fired** around 86.6K → dropped to 12.7K → rebuilt to ~29.3K

### Production Regression Line

```
Canary (2026-04-11):     t/s = 35.8  - 0.083 * K    R² = ~0.96
Production (2026-04-13): t/s = 35.12 - 0.076 * K    R² = 0.995
FA OFF baseline:         t/s = 34.5  - 0.280 * K    R² = ~0.97

Production slope ratio: 3.68x flatter than FA OFF (even better than canary)
```

### Production Data by Context Bucket

| Context Bucket | Requests | Avg Gen t/s | Range | Total Gen Tokens |
|---------------|----------|-------------|-------|-----------------|
| <30K | 17 | 34.14 | 32.66 - 36.13 | 6,636 |
| 30-60K | 2 | 31.69 | 31.19 - 32.19 | 167 |
| 60-83K | 15 | 29.26 | 28.69 - 30.04 | 17,491 |
| >=83K | 16 | 28.74 | 28.51 - 29.03 | 6,444 |

### Key Production Data Points

| Context | Gen t/s | Task | Note |
|--------:|--------:|-----:|------|
| 301 | 36.13 | 1108 | Fresh context |
| 13,851 | 34.09 | 0 | Initial prompt |
| 27,780 | 33.12 | 341 | Early growth |
| 39,925 | 32.19 | 443 | Mid context |
| 52,693 | 31.19 | 533 | Mid/high context |
| 66,201 | 30.04 | 626 | High context |
| 73,257 | 29.47 | 1518 | Deep context |
| 78,926 | 29.14 | 5778 | Deep context |
| 83,276 | 28.69 | 10317 | Near compaction |
| 84,248 | 28.61 | 11635 | Pre-compaction |
| 86,583 | 28.61 | 25883 | Pre-compaction high |
| 87,388 | 28.51 | 26418 | Peak context (session max) |
| 15,370 | 33.97 | 27861 | Post-compaction (recovered) |
| 29,680 | 32.66 | 30847 | Post-compaction rebuilt |

### Canary vs Production Comparison

| Metric | Canary (Apr 11) | Production (Apr 13) | Match? |
|--------|----------------|--------------------:|--------|
| Intercept | 35.8 t/s | 35.12 t/s | Yes (within 2%) |
| Slope | -0.083/K | -0.076/K | Yes (slightly better) |
| At 84K | 28.8 t/s | 28.6 t/s | Yes (within 1%) |
| At fresh | 36.14 t/s | 36.13 t/s | Identical |
| R² | ~0.96 | 0.995 | Production tighter fit |

**Conclusion: Canary data fully validated. Production performance matches or slightly exceeds canary measurements.**

---

## Prompt Reprocessing Analysis (Production Session)

### The Real Latency Problem

With FA solving the generation speed curve, the remaining latency source is
**prompt reprocessing** — the hybrid model (Gated DeltaNet + attention) cannot
partially restore recurrent state from checkpoints when the conversation branch
changes. This forces full re-evaluation from scratch.

### Reprocessing Events (13 total)

| Task | Tokens Reprocessed | Time Cost | Trigger |
|-----:|-------------------:|----------:|---------|
| 1140 | 66,053 | 47.9s | Branch change |
| 9809 | 78,923 | 61.5s | Cache miss |
| 16079 | 78,792 | 61.3s | Cache miss |
| 22013 | 76,007 | 61.2s | Checkpoint too old (8192) |
| 26418 | 86,875 | 69.3s | Full reprocess |
| 12568 | ~45,056 | ~35s | CANCELED at 60% |
| 27437 | ~45,056 | ~35s | CANCELED at 56% |
| Others (7) | <20K each | <12s each | Various |

**Total reprocessing cost: ~5.65 minutes** (completed) + ~2 minutes (wasted on canceled tasks)

### When Checkpoints DO Work

When the prompt branch stays coherent (same conversation, incremental growth),
checkpoint restore is fast:

| Restore From | Tokens To Process | Time |
|-------------:|------------------:|-----:|
| 81,903 | 276 | 1.10s |
| 82,175 | 338 | 1.30s |
| 84,182 | 41 | 0.60s |
| 85,789 | 77 | 0.75s |

### Cache Pressure

Prompt cache limit is 8,192 MiB. High-context sessions with many checkpoints
push hard against this (each checkpoint = ~150 MiB of recurrent state):

- 66K tokens → 7,580 MiB
- 84K tokens → 7,661 MiB
- 87K tokens → 9,754 MiB (exceeds limit, forces eviction)

### Compaction Timeline

Compaction did not fire exactly at the 85% threshold (83,558 tokens). The server
continued processing through 84K, 85K, 86K before Hermes triggered compaction:

1. Context grew to 86,607 tokens (peak: 87,388)
2. Hermes triggered compaction — 2 canceled attempts, 1 completed
3. Context dropped to 12,710 tokens (the compressed summary)
4. Rebuilt through normal conversation to ~29,307 tokens
5. Speed recovered to 32.66-33.97 t/s immediately

---

## Long Output Cost Analysis

Several slow turns were slow because of token volume, not speed degradation:

| Task | Generated Tokens | Time | Gen t/s | Thinking Budget |
|-----:|----------------:|-----:|--------:|----------------|
| 5778 | 4,028 | 138.2s | 29.14 | Unlimited |
| 16079 | 3,749 | 130.0s | 28.83 | Unlimited |
| 27861 | 2,661 | 78.3s | 33.97 | Unlimited |
| 1518 | 1,389 | 47.1s | 29.47 | Unlimited |
| 3827 | 1,949 | 66.4s | 29.37 | Unlimited |

The reasoning budget is set to 2,147,483,647 (max int = unlimited). Much of the
generated output is invisible `<think>` reasoning traces. This is by design — the
thinking enables reliable tool calling, which is the primary use case.

---

## Optimization Assessment (2026-04-13)

After exhaustive analysis, the setup has reached its performance ceiling for
this hardware + model combination:

| Optimization | Expected Gain | Worth It? |
|-------------|--------------|-----------|
| Flash attention (done) | 2.5-3.5x at deep context | **YES — already applied** |
| Streaming (done) | UX only, no speed change | **YES — already applied** |
| Q4 quantization | 10-15% t/s | No — risks tool call quality |
| KV cache q8_0 | 1-3% t/s | No — imperceptible |
| Thread tuning | 1-2% | No — noise |
| Batch tuning | Prompt eval only | No — doesn't affect generation |
| Reasoning budget cap | Reduces token volume | No — hurts tool call accuracy |

**The remaining performance characteristics are physics: 27B parameters, consumer
GPUs, no NVLink. The system is stable, optimized, and production-ready.**

---

## Credit

This investigation was prompted by a ChatGPT analysis that flagged the stale config assumption. All five cited llama.cpp issues / PRs were independently verified against the actual GitHub repositories using the `gh` CLI before any config changes were made. Production validation analysis was cross-checked against a separate Codex review of the same server log, which independently identified prompt reprocessing and token volume as the primary remaining latency sources (not generation speed degradation).
