# Limitations — What This Stack Can't Do

Read this before you commit hardware to a path you might regret. This document is the honest assessment of where the local stack falls short of frontier products today.

If you're skimming this for an answer to *"should I use this instead of Claude Code?"* — the answer is *for most everyday agentic coding work, yes; for frontier-class long-horizon reasoning, no.* The rest of this document explains where the line is.

---

## Where Claude Code wins

### Long-horizon coherence

Sonnet 4.6 and Opus 4.7 outclass Qwen3.5-27B and GLM-4.7-Flash on tasks past 50K context. The local models start losing the thread on complex multi-step plans — they'll forget constraints from earlier in the conversation, repeat work they've already done, or drift from the original goal. Hermes's compression helps (it summarizes middle turns when context fills), but the underlying model capacity gap is real and measurable.

If you need Opus-class reasoning on a 6-hour autonomous task across 200K context, run Claude Code. The local stack is not a substitute.

### Tool-call reliability ceiling

97% (Qwen3.5-27B) and 93% (GLM-4.7-Flash) on the ToolCall-15 benchmark are excellent for local models — both score within striking distance of frontier. They are not 99.9%. Anthropic doesn't publish their numbers, but the gap exists. For production systems where one failed tool call cascades into multiple downstream errors, that 3-7% gap can matter.

### Polish layer

Claude Code's UX is tuned. Error messages help you understand what went wrong. Slash commands are discoverable. The agentic loop feels smooth. Hermes is functional but rough in places. Hermes is also shipping features weekly — by the time you read this, both products will be different than they were when this was written.

### Plan mode

Claude Code has a dedicated plan mode where the agent drafts a plan before acting and lets you approve or modify it. Hermes has the `plan` skill — equivalent capability, less polished UX. If you do a lot of long-horizon planning work, Claude Code's plan mode is genuinely better today.

### Worktree-isolated subagent execution

Claude Code spawns subagents inside git worktrees, so parallel agent work doesn't conflict on shared files. Hermes has `delegate_task` for subagent delegation but doesn't natively isolate to worktrees. You can achieve the same with manual setup, but it's not zero-config.

### Context compaction quality

Claude Code's context compaction is tuned by Anthropic on their own models. Local compaction uses your local model to summarize itself — it works, but the summaries are lower quality than what Claude Code produces. For sessions that hit compaction repeatedly, the quality difference compounds.

---

## Hardware-specific limits

### Context window ceiling

96K tokens is the practical ceiling on dual 3090s with f16 KV cache and the UD-Q5_K_XL Qwen model. Larger windows are technically possible with KV quantization or smaller models, but every option degrades some property — speed, quality, or the headroom for inference overhead.

### Speed degradation at depth

On Qwen3.5-27B, generation speed degrades linearly with context: `t/s = 35.12 - 0.076 × K` (K = thousands of tokens, R² = 0.995). At fresh context, ~36 t/s. At 84K context, ~28.6 t/s. The degradation is mild because flash attention flattens the curve, but it's real.

On GLM-4.7-Flash, the degradation is steeper (Amdahl's Law — see `benchmarks/glm_vs_qwen.md`). GLM at fresh context is ~124 t/s; at 92K context, ~63 t/s. Still faster than Qwen at every context depth in our measurements, but the gap narrows.

### Single-GPU constraints — read this before committing to the accessible path

On a single RTX 3090, you're limited to Qwen3.5-27B at Q4_K_XL with q8_0 KV cache and ~32K context. The full agentic loop works — but the 32K window is meaningfully tighter than it sounds, and that affects what kinds of work this path is actually good for.

**The math.** Hermes loads about **12K tokens of system prompt at session start**: SOUL.md, tool definitions for all 30 built-in tools, descriptions for all 97 skills, persistent memory contents, and Hermes's own behavioral instructions. That overhead is constant — every conversation starts at ~37% of a 32K window already filled. Compaction fires at 85% of context (~27K on a 32K window), leaving roughly **15K of effective working space** for the actual conversation, tool calls, and file contents.

**What 15K of effective space buys you, in practice:**

- ~5-10 rounds of simple tool calls (file read, terminal command, single web search — ~1-2K each)
- ~3-5 rounds of coding work (read + edit + test + observe + iterate — ~3-5K each)
- ~1-3 rounds of research with multiple searches (~5-10K each)
- One large file read (1500+ line log) eats 30-40% of your budget alone

After compaction fires, Hermes summarizes middle turns and drops back to a smaller context — work continues, but conversational fidelity degrades and the model loses some of the earlier reasoning chain.

**Use the single-3090 path for:**

- Single-file edits and focused refactors
- Q&A and code review on small targets
- Short tool-call chains where you know what you want and the path is direct
- Narrow targeted automation (cron jobs, scheduled tasks, one-shot transformations)
- Mobile access via Telegram for quick agent queries from your phone

**Don't use it for:**

- Multi-file refactors that need the agent to hold many files in working memory
- Extended debugging sessions where context grows turn after turn
- Open-ended research that involves many searches with large result sets
- Sustained pair-programming sessions running hours at a stretch

**Quality tradeoff vs the dual-3090 path:**

- ~95%+ tool-calling at the recommended temperature (vs 97% with UD-Q5_K_XL on dual GPU)
- Q4_K_XL quantization is more aggressive than UD-Q5_K_XL — measurable quality gap exists in edge cases (long-form generation, complex multi-step reasoning)
- For ~90% of one-shot agentic tasks, you won't notice the difference

**Bottom line:** the single-3090 path is genuinely useful but it's a different *category* of agent work than the dual-3090 path enables. If your day-to-day involves the "don't use it for" list above, save up for the dual-3090 path or skip straight to frontier-class hardware. If your day-to-day is the "use it for" list, the accessible path will serve you well at a fraction of the cost.

See `configs/qwen3.5-27b_single/` for the launch script.

### Reprocessing penalty (Qwen-specific)

Qwen3.5-27B uses a hybrid DeltaNet + attention architecture. The recurrent state cannot be partially restored from checkpoints when the conversation branch changes — this forces full re-evaluation from scratch. In a 50-data-point production session, this triggered 13 full reprocessing events at 60-70 seconds each. Total dead air: ~5.65 minutes over a multi-hour session. GLM doesn't have this problem (pure MLA attention).

---

## What this experiment did NOT cover

To be explicit about scope:

- **No side-by-side timed comparison vs Claude Code on identical tasks.** Tool-call accuracy numbers are independent benchmarks; the user-experience comparison is qualitative.
- **No fine-tuning runs.** The hardware supports fine-tuning (LoRA, QLoRA, full-parameter with FSDP) but this experiment was about inference + agent setup only.
- **No SWE-Bench eval on the agentic stack.** ToolCall-15 measures tool calling; SWE-Bench measures coding correctness. They overlap but aren't the same. We did not run SWE-Bench through Hermes on our models.
- **No multi-day uptime measurement.** Sessions ran for hours, not days. Long-running stability beyond ~6 hours isn't documented.
- **No comparison against Cursor, Aider, OpenCode, or other AI coding tools.** The reference point is Claude Code (the closest functional equivalent to Hermes).
- **No evaluation of larger open-source models** (Kimi K2, GLM-5 full, DeepSeek V3.5+ at full precision). Those require bigger hardware than we have.

---

## When you should NOT use this stack

- You need frontier reasoning on tasks longer than 50K context → Claude Code or another frontier model.
- You need 99.9% tool-call reliability for production systems → frontier API.
- You're not comfortable debugging open-source software → start with a hosted product first; come back when you've used it enough to know what you want.
- Your work is bound by compliance requirements that prohibit running models on un-audited hardware → don't run locally without legal sign-off.
- You don't have the patience for the Hermes setup process — it's improving, but it's still rough. The "Hermes Agent is new" framing in the blog post is real.

---

## A note on the trajectory

This document is a snapshot. Every limitation listed above is being actively worked on by someone in the open-source ecosystem. Some of them will close in the next quarter — Hermes shipped two major version bumps between the experiment running and this writeup. Some will persist for longer.

The honest framing: **right now**, the local stack handles ~95% of everyday agentic coding work and falls short on the long-horizon frontier. **A year from now**, that ratio shifts further toward "local handles it." We don't know exactly when each specific limitation falls, but the trajectory is monotonic.

If a limitation in this list matters to your work, check whether it still applies before you act on it — it might already be obsolete.
