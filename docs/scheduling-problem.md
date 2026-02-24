# The llmux Scheduling Problem

## Problem statement

A single GPU serves N model classes. Each model class has its own request queue.
Serving requires that the model's weights are loaded on the GPU — only one model
can be active at a time. Switching from model i to model j costs C(i,j) seconds
during which no requests are served (the GPU is loading weights).

Requests arrive online. We don't know the future.

**Objectives** (in priority order):

1. **No starvation**: every request must be served within some bounded time
2. **High utilization**: maximize the fraction of GPU-seconds spent serving
   requests vs. switching models

These are in tension. Utilization wants long uninterrupted runs per model
(amortize switch cost over many served requests). Starvation protection wants
responsiveness to queued requests for non-active models.

## What this is (and isn't)

This is **single-processor scheduling with heterogeneous context switch costs**.

In queueing theory, the precise name is a **polling system**: a single server
cyclically visits multiple queues, with switchover times between them. The
literature studies these extensively — the key design axes are:

| Axis | Options | Analogy in llmux |
|------|---------|------------------|
| **Service discipline** | Exhaustive, gated, k-limited | How long to serve the active model before considering a switch |
| **Routing** | Cyclic, demand-based, priority | Which model to switch to next |
| **Preemption** | Preemptive, non-preemptive | Whether to interrupt the active model's queue mid-service |

The distinguishing feature of our system vs. classical polling:

- **Switch costs are large and heterogeneous.** Classical polling assumes
  constant or negligible switchover. Here, C(i,j) ranges from 2s (L1 warm
  offload) to 60s+ (L3 cold start), and varies by direction. The ratio of
  switch cost to service time can be 100:1 or worse.
- **Arrivals are bursty and unpredictable.** Traffic comes from humans using
  chat UIs or batch pipelines — arrival patterns shift abruptly.

## Current state

The only policy currently implemented is **FIFO**: switch immediately on the
first request for a non-active model. This is simple and predictable, but
causes excessive switching under mixed workloads — every interleaved arrival
triggers a full switch.

## The two real axes

If we strip away implementation details, the scheduling problem has two
independent knobs:

### Axis 1: When to leave the current model (utilization)

> "How many requests should the active model serve before we're willing to pay
> the switch cost?"

This is the **service discipline** and it directly controls utilization. The
spectrum:

```
Immediate                              Exhaustive
(switch on 1st arrival)                (drain queue completely)

FIFO ──────────────────────────────── ???
low utilization                        high utilization
low latency for                        high latency for
  non-active model                       non-active model
```

The optimal point on this spectrum depends on the switch cost relative to
service time. When C/s is small (say, 2s switch, 1s service = 2:1), switching
aggressively is fine. When C/s is large (60s switch, 0.5s service = 120:1),
you need to serve a lot of requests to amortize each switch.

### Axis 2: When to force a switch (starvation protection)

> "How long is any request allowed to wait before we override the utilization
> objective and switch anyway?"

This is the **staleness bound** and it directly controls worst-case latency.
FIFO doesn't need one (it always switches immediately), but any smarter policy
will need a hard override to prevent starvation.

**The relationship between the axes**: the staleness bound only fires when the
service discipline fails to switch soon enough on its own. For exhaustive
service, staleness is the *only* mechanism that triggers a switch to a
non-active model. For amortized service, staleness is a safety net that rarely
fires because other mechanisms usually trigger first.

## Design considerations

### Proportional service quanta

The serving quantum (minimum time to serve before switching) should be
proportional to the switch cost to the *next* model, not a fixed parameter.

### Directional cost awareness in routing

When choosing which model to switch to, the cost C(from, to) matters. If model
B has 5 requests and model C has 4, but switching to B costs 60s and switching
to C costs 2s, a cost-aware policy should prefer C.

### Preemption during switch

Currently, a switch is atomic: drain, sleep, wake. During the wake phase
(which dominates — 87.6% of switch time), no requests are served. If requests
arrive for the *old* model during the wake, they queue and wait.

An alternative: allow the old model to keep serving during wake if the old
model's process is still alive (which it is for `keep_running` eviction). This
is like "pre-copy migration" in VM live migration.

### Adaptive staleness bounds

A static staleness bound (max_wait) could adapt to:
- Current switch cost estimate (higher cost → higher max_wait acceptable)
- Current queue depth imbalance (large imbalance → lower max_wait)
- Historical arrival patterns (bursty traffic → longer coalescing makes sense)

## Open questions

1. For the 2-model case (the most common deployment), routing is trivial and
   the only question is *when* to switch. Does a simple amortized approach
   (serve for at least C seconds, then switch if demand justifies it) suffice?

2. Should routing consider expected cost (C * demand_ratio) rather than just
   demand?

3. What's the right way to evaluate policies? The discrete-event simulator
   (`tests/policy_sim.rs`) tests synthetic workloads — are there representative
   real-world arrival traces we should be testing against?

4. Should there be a time-level scheduler that runs independently of request
   arrivals, or is a request-triggered policy sufficient?
