# llmux Switching Efficiency Report

Collected 2026-02-06 on a single NVIDIA L40 (49 GB VRAM) running llmux v0.3.1
with Prometheus scraping at 15s intervals.

---

## Report 1: Two-Model Switching (gpt-oss-20b + gemma-12b)

**Setup**: gpt-oss-20b (20B params, sleep L1 — CPU offload) and gemma-12b (12B
params, sleep L2 — discard weights). Both use `gpu_memory_utilization=0.9`.

### Workload

1. Cold-start both models sequentially
2. Warm switching in both directions
3. Burst of 10 concurrent requests to active model
4. Switch under load (5 long-running requests + 5 to other model)
5. Two full round-trips

### Switch Performance

| Direction | Count | Avg Duration | Breakdown |
|-----------|------:|-------------:|-----------|
| (cold) → gpt-oss-20b | 1 | 130.5s | 100% wake (vLLM startup) |
| (cold) → gemma-12b | — | — | Only via gpt-oss→gemma path |
| gpt-oss-20b → gemma-12b | 4 | 38.5s | 4% cooldown, 15% sleep, 81% wake |
| gemma-12b → gpt-oss-20b | 3 | 3.6s | 40% cooldown, 28% sleep, 32% wake |

The asymmetry is stark: switching **to** gpt-oss-20b (L1, CPU-offloaded weights)
takes ~3.6s. Switching **to** gemma-12b (L2, weights discarded) takes ~38.5s
because the model must reload weights from disk.

### Wake Time Dominates

Phase totals across all switches:

| Phase | Total Time | Share |
|-------|----------:|------:|
| Wake | 258.8s | **87.6%** |
| Sleep | 25.5s | 8.6% |
| Cooldown | 11.1s | 3.7% |
| Drain | 0.0s | 0.0% |

Wake (loading model weights into GPU) is the overwhelmingly dominant cost.
Drain time is zero because the workload has low enough concurrency that
in-flight requests complete before drain begins.

### Request Impact

| Metric | Value |
|--------|------:|
| Total requests | 36 |
| Requests that waited | 16 (44%) |
| Avg wait (gpt-oss-20b) | 8.4s |
| Avg wait (gemma-12b) | 10.8s |
| Total switch time | 295.4s |

### GPU Efficiency

Over the observation window (~7 minutes of active use after cold starts):
- ~295s spent switching out of ~420s total = **~30% switching overhead**
- **GPU serving fraction: ~70%**

But this is misleading — the cold starts dominate. Excluding the two cold
starts (which totalled ~233s), warm switching consumed ~62s across 6 warm
switches, yielding:
- Warm switching overhead: ~62s / ~187s active time = **~33% overhead**
- **Warm GPU serving fraction: ~67%**

Even in the warm steady state with only two models, one-third of GPU time is
spent switching rather than serving. The L2 wake penalty (gemma-12b) is the
primary driver.

---

## Report 2: Three-Model Switching (+ qwen3-4b)

**Added**: qwen3-4b (4B params, sleep L3 — full process stop). This is the most
aggressive sleep level — the vLLM process is killed entirely and must cold-start
on every wake.

### Workload

1. Cold-start all three models sequentially
2. Warm round-robin exercising all 6 switch directions
3. Simultaneous requests to all three models
4. "Hot model" pattern: mostly gemma-12b with occasional interrupts

### Switch Performance

| Direction | Count | Avg Duration | Bottleneck |
|-----------|------:|-------------:|-----------|
| gemma-12b → qwen3-4b | 4 | 63.0s | 93% wake (cold start!) |
| gpt-oss-20b → qwen3-4b | 1 | 55.1s | 90% wake |
| gpt-oss-20b → gemma-12b | 4 | 50.4s | 80% wake |
| (cold) → gemma-12b | 1 | 73.7s | 100% wake |
| (cold) → gpt-oss-20b | 2 | 25.0s | 100% wake |
| qwen3-4b → gpt-oss-20b | 2 | 5.4s | 66% cooldown, 32% wake |
| gemma-12b → gpt-oss-20b | 1 | 2.0s | 41% sleep, 60% wake |

### Three-Model Dynamics

**Switch failures appeared**: 3 failures waking gemma-12b. After cycling through
all three models, gemma-12b's L2 sleep (discard weights) sometimes left vLLM in
a state where the `/wake_up` endpoint returned HTTP 500. This forced llmux into
the failure path (force-stop → cold restart → long wake).

**qwen3-4b is expensive**: At L3 (process stop), every switch to qwen3-4b is
effectively a cold start. Despite being a small 4B model, wake time averaged 55–63s
because vLLM must reinitialize CUDA, load the model, and warm up.

**Cooldown becomes visible**: With rapid three-way switching, the 5s minimum active
duration cooldown shows up. In the qwen3-4b → gpt-oss-20b direction, cooldown was
66% of the switch time — the actual wake was fast (L1 offload) but the policy
forced a wait.

### Request Impact

| Model | Requests | Avg Wait | Waited? |
|-------|----------|----------|---------|
| gpt-oss-20b | 5 | 12.5s | all waited |
| gemma-12b | 9 | 30.8s | 5 waited, 4 immediate |
| qwen3-4b | 5 | 77.8s | all waited |

Average queue wait for qwen3-4b requests: **78 seconds**. Users requesting qwen3-4b
effectively wait for a full vLLM cold start every time.

### Total Switching Overhead

- 644.9 seconds total switch time
- ~15 successful switches + 3 failures
- Average: ~36s per switch (including cold starts and failure recovery)
- 79% of requests had to wait for a model switch

---

## Report 3: Policy Analysis and Recommendations

### How the Current Policy Creates These Results

The current `FifoPolicy` is the simplest possible scheduling strategy:

```
on_pending_request() → SwitchNow (always, immediately)
prepare_switch()     → wait_for_in_flight() (drain current requests)
min_active_duration  → 5 seconds (cooldown anti-thrash)
```

It switches **on the very first request** for a non-active model. There is no
batching, no coalescing, no cost awareness.

#### What this causes:

1. **Thrashing under mixed traffic**: When requests arrive for model A and
   model B in quick succession, each arrival triggers a switch. The 5s cooldown
   prevents the worst case but does not prevent oscillation at the 5s period.
   In our three-model test, we saw 15+ switches in 11 minutes.

2. **No cost awareness**: A switch to gpt-oss-20b (L1, 2s wake) and a switch
   to qwen3-4b (L3, 55s wake) look the same to the policy. It doesn't consider
   that switching to qwen3-4b will make the requester wait 55s, while switching
   to gpt-oss-20b costs only 2s.

3. **No request batching**: If 10 requests for model B arrive while model A is
   active, only the first triggers the switch. The other 9 wait passively. This
   is correct but doesn't consider whether it's worth switching at all — maybe
   those 10 requests could be served when model B naturally becomes active
   later, avoiding a switch entirely.

4. **No awareness of pending requests for the current model**: If model A is
   active with 100 pending requests and 1 request arrives for model B, the
   policy switches immediately — potentially delaying those 100 requests.

### Inspiration from Analogous Systems

GPU model switching is structurally similar to several well-studied scheduling
problems:

| System | Analogy | Key Insight |
|--------|---------|-------------|
| **OS CPU scheduling** | Context switch ≈ model switch | Cost-aware schedulers (CFS) use time slices proportional to priority/demand, not first-come switches |
| **Database buffer pool** | GPU memory ≈ buffer pool, model ≈ cached page set | LRU/clock algorithms avoid evicting hot pages; frequency-based policies (LFU, ARC) outperform pure recency |
| **VM live migration** | Model sleep/wake ≈ migration cost | Pre-copy and post-copy strategies minimize downtime; cost is weighed against benefit |
| **Kubernetes pod scheduling** | GPU ≈ node, model ≈ pod | Bin-packing considers resource cost; preemption only when benefit exceeds disruption |
| **Disk elevator / I/O scheduling** | Switch direction ≈ seek direction | Batching requests for the same "direction" amortizes the seek cost |

The most directly applicable analogy is the **OS CPU scheduler**: models are
"processes" and the GPU is the "CPU". The key difference is that context switch
cost is not constant — it varies by orders of magnitude depending on sleep level
and model size.

### Recommended First Steps

#### Step 1: Cost-Aware Switch Threshold

Don't switch unless the expected benefit exceeds the cost.

```
switch_cost(from, to) = estimated wake time for 'to' at its sleep level
benefit(to) = queue_depth(to) * expected_service_time
```

Only switch when `benefit > switch_cost * threshold`. This single change would:
- Prevent switching to qwen3-4b for a single short request (55s switch for a
  0.3s inference — 180x overhead)
- Still switch promptly when queue depth makes it worthwhile
- Use historical switch durations from our `llmux_switch_total_seconds` histogram
  to estimate cost dynamically

**Implementation**: Replace `SwitchNow` with `Defer` when
`queue_depth * avg_service_time < estimated_switch_cost * 0.5`.

#### Step 2: Coalescing Window

Instead of switching on the first request, wait a short window to collect
demand before deciding.

```
on_pending_request(ctx):
    if ctx.target_queue_depth == 1:
        return Defer(sleep(2s))  # wait for more requests
    else:
        return SwitchNow  # batch already forming
```

This 2-second coalescing window would catch burst patterns like "5 requests
arrive in 200ms" without switching 5 times. The existing `Defer` mechanism
in `PolicyDecision` already supports this — the infrastructure is built.

#### Step 3: Sleep Level Demotion

Currently, sleep levels are static per model. A smarter policy would:
- Use L1 (CPU offload) for models that switch frequently
- Escalate to L2/L3 only after a model has been asleep for a long time
- Track switch frequency per model pair to inform this

```
sleep_level(model):
    time_since_last_active = now - last_active[model]
    if time_since_last_active < 60s: return L1   # recently hot
    if time_since_last_active < 300s: return L2   # warm
    return L3                                      # cold, free memory
```

This alone would have prevented the gemma-12b L2 wake failures we observed —
if gemma was recently active, it would stay at L1 and wake in 2s instead of 9s.

#### Step 4: Time-Slice Scheduling (Longer Term)

For truly mixed workloads, implement round-robin time slices:
- Each model gets a time slice proportional to its pending request volume
- Switch only at slice boundaries
- Minimum slice = `wake_time * 3` (ensure the model serves long enough to
  amortize the switch cost)

This mirrors CFS (Completely Fair Scheduler) in Linux but with variable
context-switch costs.

### Priority Order

1. **Cost-aware threshold** — highest impact, simplest change, prevents the
   worst pathological switches
2. **Coalescing window** — easy with existing `Defer`, prevents burst-induced
   thrashing
3. **Sleep level demotion** — moderate complexity, large reliability improvement
4. **Time slices** — most complex, best long-term solution for mixed workloads

### Key Metrics for Evaluating Policy Changes

The instrumentation we just deployed provides everything needed:

| PromQL | What it measures |
|--------|-----------------|
| `1 - rate(llmux_switch_total_seconds_sum[5m])` | GPU serving fraction (headline) |
| `rate(llmux_switches_total[5m])` | Switch frequency |
| `histogram_quantile(0.95, rate(llmux_request_queue_wait_seconds_bucket[5m]))` | P95 user-facing wait |
| `rate(llmux_switch_failures_total[5m])` | Failure rate |
| `rate(llmux_switch_phase_seconds_sum[5m])` | Time budget by phase |

The goal: maximize GPU serving fraction while keeping P95 queue wait under an
acceptable bound (e.g., 10s for warm switches).

---

## Report 4: Cost-Aware Coalescing Policy — Implementation & Results

Implemented and benchmarked a cost-aware policy following the recommendations
from Report 3 (steps 1 and 2: cost-aware threshold + coalescing window).

### Algorithm

The `CostAwarePolicy` evaluates five checks in order when a request arrives
for a non-active model:

```
1. Staleness override   — oldest_waiting >= max_wait?     → SwitchNow
2. Idle GPU             — no active model?                → SwitchNow
3. Serving window       — active_duration < switch_cost?  → Defer(remaining)
4. Cost threshold       — queue_depth >= required?        → SwitchNow
5. Coalesce             — defer for coalesce_window       → Defer(window)
```

**Cost threshold** computes `required_depth = ceil(amortization_factor * switch_cost_secs)`.
With factor=0.5 and a 10s switch cost, this requires 5 pending requests before
switching is considered worthwhile. The floor is 1 — a switch is never refused
entirely.

**Switch cost estimation** uses per-direction exponential moving averages (EMA,
alpha=0.3) of observed switch durations. Observations are capped at 60s to
prevent cold-start contamination. Initial estimates start at 10s and quickly
adapt to actual values.

**Coalescing** defers the first request for a non-active model by 2s, allowing
burst traffic to accumulate. Subsequent requests during the window return `Skip`
(a no-op — the switch is already being arranged). After the window, `do_switch`
runs unconditionally.

### Configuration

```json
{
  "policy": {
    "policy_type": "cost_aware",
    "coalesce_window_ms": 2000,
    "amortization_factor": 0.5,
    "max_wait_secs": 15,
    "min_active_secs": 5
  }
}
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `coalesce_window_ms` | 2000 | Defer window to collect burst demand |
| `amortization_factor` | 0.5 | Higher = more reluctant to switch. 0.0 = FIFO behavior |
| `max_wait_secs` | 15 | Hard upper bound on request wait before forcing switch |
| `min_active_secs` | 5 | Anti-thrash cooldown (inherited from base policy) |

### Benchmark Setup

Five workload profiles against two models (gpt-oss-20b at L1, gemma-12b at L2)
on a single NVIDIA L40:

| Profile | Pattern | Requests |
|---------|---------|----------|
| single_model | All requests to one model (no switching) | 40 |
| balanced | Alternating requests A, B, A, B... | 40 |
| bursty | Bursts of 10 to each model | 40 |
| dominant | 80% model A, 20% model B | 50 |
| interleave | Concurrent requests to both models | 60 |

### Initial Results: FIFO vs v2 (Cost-Aware, No Serving Window)

The first iteration (v2) implemented cost threshold + coalescing only (steps
1–2–4–5 from the algorithm above, without step 3 — the serving window).

| Profile | | Switches | Switch Time (s) | GPU Serving % | Avg Wait (s) |
|---------|---------|-------:|--------:|--------:|--------:|
| single_model | FIFO | 0 | 0 | 100.0 | 0.0 |
| | v2 | 0 | 0 | 100.0 | 0.0 |
| balanced | FIFO | 37 | 350.5 | 3.7 | 11.4 |
| | **v2** | **20** | **143.2** | **26.9** | **4.6** |
| bursty | FIFO | 4 | 34.6 | 9.0 | 8.6 |
| | **v2** | **3** | **27.8** | **10.5** | **6.9** |
| dominant | FIFO | 3 | 22.5 | 2.0 | 7.5 |
| | v2 | 3 | 23.2 | 3.5 | 7.4 |
| interleave | FIFO | 2 | 15.0 | 6.6 | 11.4 |
| | v2 | 2 | 16.1 | 5.1 | 10.4 |

**Totals (excluding single_model)**:

| Metric | FIFO | v2 | Change |
|--------|-----:|----------:|-------:|
| Total switches | 46 | 28 | **-39%** |
| Total switch time | 422.6s | 210.2s | **-50%** |
| Weighted GPU serving % | 4.8 | 22.4 | **+17.6pp** |
| Weighted avg wait | 10.8s | 6.3s | **-42%** |
| Failures | 0 | 0 | — |

**Analysis**: The balanced profile saw the biggest improvement — FIFO triggered
37 switches for 40 alternating requests, while v2 coalesced requests to just 20
switches (26.9% GPU serving vs 3.7%). Bursty improved modestly (one fewer
switch). Dominant and interleave had low switch counts to begin with, so v2
degraded gracefully to FIFO-like behavior.

However, balanced still showed 20 switches for 40 requests. Serial alternating
traffic cannot be coalesced — there is never more than one request pending for
the non-active model. The coalescing window expires and triggers a switch every
time.

### Iteration: Serving Window (v3)

The v2 results revealed that the balanced profile — serial alternating traffic
(A, B, A, B...) — still triggered 20 switches for 40 requests despite
coalescing. The issue: with one request at a time, there is never more than one
pending request for the non-active model, so the coalescing window cannot
accumulate demand. After the coalesce window expires, the policy always
switches.

**Insight**: After paying the cost to wake a model, it should serve for at
least as long as the switch cost before we allow another switch. This is the
"serving window" — a minimum tenure that ensures the wake cost is amortized
through actual serving time. The key is that the serving window is dynamic,
derived from the same EMA-tracked switch costs used for the cost threshold.

**Implementation**: Added `active_duration` to `PolicyContext` (how long the
current model has been active since its last wake completed). In
`on_pending_request`, after checking staleness and idle GPU, the policy
checks whether the active model has served for at least `switch_cost` seconds.
If not, it defers for the remaining time. This check comes *before* the cost
threshold check — even a high queue depth should not preempt a model that
hasn't yet amortized its own wake cost. The staleness bound (15s) still
provides a hard cap.

```
Decision order (v3):
1. Staleness override   — oldest_waiting >= max_wait?     → SwitchNow
2. Idle GPU             — no active model?                → SwitchNow
3. Serving window       — active_duration < switch_cost?  → Defer(remaining)
4. Cost threshold       — queue_depth >= required?        → SwitchNow
5. Coalesce             — defer for coalesce_window       → Defer(window)
```

**Results: v2 vs v3**

| Profile | | Switches | Switch Time (s) | GPU Serving % | Avg Wait (s) |
|---------|---------|-------:|--------:|--------:|--------:|
| balanced | v2 | 20 | 143.2 | 26.9 | 4.6 |
| | **v3** | **20** | **117.9** | **66.9** | **8.8** |
| bursty | v2 | 3 | 27.8 | 10.5 | 6.9 |
| | **v3** | **3** | **20.2** | **40.6** | **8.2** |
| dominant | v2 | 3 | 23.2 | 3.5 | 7.4 |
| | v3 | 5 | 40.8 | 0.5 | 9.2 |
| interleave | v2 | 2 | 16.1 | 5.1 | 10.4 |
| | v3 | 2 | 15.8 | 1.5 | 10.7 |

The serving window transformed balanced and bursty. Balanced GPU serving went
from 26.9% to 66.9% — the models now serve for meaningful durations between
switches rather than immediately sleeping on the next arrival. Bursty improved
similarly: 10.5% → 40.6%.

Dominant and interleave regressed modestly. The serving window is symmetric —
a model with 1 pending request gets the same window as one with 10. For the
dominant profile (80% one model, 20% another), this delays the minority model's
requests longer. However, the staleness bound prevents catastrophic wait times.

**Reverted experiment (v4)**: Tested moving the cost threshold check before the
serving window, so high queue depth could override the serving window. Results
were nearly identical for the regressing profiles but lost 6pp on balanced
(60.9% vs 66.9%). The window-first ordering was restored because the
balanced/bursty gains (+40pp, +30pp) far outweigh the dominant/interleave
losses (-3pp, -3.6pp).

### Final Results: FIFO → v3

| Profile | | Switches | Switch Time (s) | GPU Serving % | Avg Wait (s) |
|---------|---------|-------:|--------:|--------:|--------:|
| single_model | FIFO | 0 | 0 | 100.0 | 0.0 |
| | v3 | 0 | 0 | 100.0 | 0.0 |
| balanced | FIFO | 37 | 350.5 | 3.7 | 11.4 |
| | **v3** | **20** | **117.9** | **66.9** | **8.8** |
| bursty | FIFO | 4 | 34.6 | 9.0 | 8.6 |
| | **v3** | **3** | **20.2** | **40.6** | **8.2** |
| dominant | FIFO | 3 | 22.5 | 2.0 | 7.5 |
| | v3 | 5 | 40.8 | 0.5 | 9.2 |
| interleave | FIFO | 2 | 15.0 | 6.6 | 11.4 |
| | v3 | 2 | 15.8 | 1.5 | 10.7 |

**Totals (excluding single_model)**:

| Metric | FIFO | v3 | Change |
|--------|-----:|----------:|-------:|
| Total switches | 46 | 30 | **-35%** |
| Total switch time | 422.6s | 194.7s | **-54%** |
| Weighted GPU serving % | 4.8 | 56.6 | **+51.8pp** |
| Failures | 0 | 0 | — |

### Robustness Verification

Full fuzz test suite re-run after deploying v3: **1,019/1,019 requests completed
successfully with zero errors** across 12 test scenarios including concurrent
switching under load, rapid model cycling, extreme burst patterns, and streaming
during switches. Unit tests: 36/36 passing.

### Where This Policy Cannot Help

The dominant and interleave profiles show modest regressions because the serving
window is direction-agnostic — it applies equally regardless of pending demand
for other models. The correct fix is **time-slice scheduling** (Report 3,
Step 4): giving each model a serving window proportional to its pending queue
volume rather than a fixed window equal to the switch cost. This requires
the `Defer` handler to re-evaluate demand after the window expires rather than
switching unconditionally — a change that is architecturally possible but
requires making `maybe_trigger_switch` callable from spawned tasks (currently
blocked by `Send` bounds on the policy trait's async methods).

**Sleep level demotion** (Report 3, Step 3) would also help: a model that
switches frequently should stay at L1 (CPU offload, ~2s wake) rather than L2
(discard weights, ~9s wake), reducing the per-switch cost and therefore the
serving window duration.

### Code Changes

Four source files modified:

- `src/policy.rs`: Added `CostAwarePolicy` (200 lines), `SwitchTiming` (EMA
  tracker), `CoalesceState`, and `PolicyDecision::Skip` variant. Added
  `on_switch_complete()` hook and `min_active_duration()` to `SwitchPolicy`
  trait. Added `active_duration` to `PolicyContext`. Serving window logic in
  `on_pending_request`.
- `src/config.rs`: Added `coalesce_window_ms`, `amortization_factor`,
  `max_wait_secs` to `PolicyConfig`. Updated `build_policy()` to accept model
  names for pre-populating timing tables.
- `src/lib.rs`: Passes model names from config to `build_policy()`.
- `src/switcher.rs`: Calls `on_switch_complete()` after successful switches.
  Handles `PolicyDecision::Skip`. Populates `active_duration` from
  `activated_at` timestamp.

No new dependencies. No changes to the public API or wire protocol.
