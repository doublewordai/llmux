//! Discrete-event policy simulator for comparing scheduling algorithms.
//!
//! Design principles (fixing issues in the previous simulator):
//!
//! 1. **The engine has no routing logic.** All switch decisions go through the
//!    policy. The engine only: enqueues arrivals, serves requests, fires deferred
//!    timers, and dispatches scheduler ticks. If a policy never switches to a
//!    queue, those requests rot — and the results show it.
//!
//! 2. **Deferred decisions fire at their scheduled time.** `DeferFor(delay)`
//!    pushes a real `DeferFired` event into the heap. No opportunistic checking.
//!
//! 3. **Per-direction switch costs.** `SwitchCosts` maps (from, to) → cost.
//!    Asymmetric costs (L1→L2 = 3.6s, L2→L1 = 38.5s) are first-class.
//!
//! 4. **Wall time is real elapsed time**, not a synthetic calculation. GPU
//!    utilization = serve_time / wall_time, which correctly accounts for idle
//!    periods.
//!
//! 5. **No end-of-simulation drain hack.** If requests aren't served, the
//!    results show it (served < total). Policies must handle all routing.

#![allow(dead_code, clippy::too_many_arguments)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};

// ── Switch costs ────────────────────────────────────────────────────────────

/// Per-direction switch costs. C(from, to) can differ from C(to, from).
#[derive(Clone)]
struct SwitchCosts {
    /// Map from "from→to" to cost in seconds. Missing entries use `default`.
    costs: HashMap<(String, String), f64>,
    default: f64,
}

impl SwitchCosts {
    /// Uniform cost: C(i,j) = cost for all i != j.
    fn uniform(cost: f64) -> Self {
        Self {
            costs: HashMap::new(),
            default: cost,
        }
    }

    /// Asymmetric costs from a list of (from, to, cost) triples.
    fn asymmetric(pairs: &[(&str, &str, f64)], default: f64) -> Self {
        let mut costs = HashMap::new();
        for &(from, to, cost) in pairs {
            costs.insert((from.to_string(), to.to_string()), cost);
        }
        Self { costs, default }
    }

    fn cost(&self, from: &str, to: &str) -> f64 {
        self.costs
            .get(&(from.to_string(), to.to_string()))
            .copied()
            .unwrap_or(self.default)
    }
}

// ── Events ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Event {
    /// A request arrives and joins a model's queue.
    Arrive { model: String, time: f64 },
    /// The active model finishes serving one request.
    Served { time: f64 },
    /// A switch completes — the target model is now active.
    SwitchComplete { model: String, time: f64 },
    /// A deferred policy decision fires.
    DeferFired { model: String, time: f64 },
    /// Background scheduler tick.
    Tick { time: f64 },
}

impl Event {
    fn time(&self) -> f64 {
        match self {
            Event::Arrive { time, .. }
            | Event::Served { time }
            | Event::SwitchComplete { time, .. }
            | Event::DeferFired { time, .. }
            | Event::Tick { time } => *time,
        }
    }

    /// Tie-breaking priority: SwitchComplete > Served > DeferFired > Tick > Arrive.
    /// This ensures switches complete before we try to serve, and serves complete
    /// before we evaluate new arrivals at the same timestamp.
    fn priority(&self) -> u8 {
        match self {
            Event::SwitchComplete { .. } => 0,
            Event::Served { .. } => 1,
            Event::DeferFired { .. } => 2,
            Event::Tick { .. } => 3,
            Event::Arrive { .. } => 4,
        }
    }
}

// Min-heap by (time, priority)
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time() == other.time() && self.priority() == other.priority()
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap: smallest time first, then smallest priority first
        other
            .time()
            .partial_cmp(&self.time())
            .unwrap_or(Ordering::Equal)
            .then(other.priority().cmp(&self.priority()))
    }
}

// ── Policy interface ────────────────────────────────────────────────────────

/// What the policy sees when a request arrives for a non-active model.
struct PolicyCtx<'a> {
    /// Which model the request is for.
    target_model: &'a str,
    /// Currently active model (always Some — cold start handled by engine).
    active_model: &'a str,
    /// Number of requests queued for the target model (including this one).
    target_queue_depth: usize,
    /// Number of requests queued for the active model.
    active_queue_depth: usize,
    /// How long the oldest request for the target has been waiting (seconds).
    oldest_waiting: f64,
    /// Whether a request is currently being served (1 or 0).
    active_in_flight: usize,
    /// How long the active model has been active since last switch (seconds).
    active_duration: f64,
    /// Switch cost from active to target.
    switch_cost: f64,
}

/// What the background scheduler sees on each tick.
struct ScheduleCtx<'a> {
    /// Currently active model (None only if idle).
    active_model: Option<&'a str>,
    /// How long the active model has been active.
    active_duration: f64,
    /// Queue depth per model.
    queue_depths: &'a HashMap<String, usize>,
    /// Whether a request is currently being served.
    active_in_flight: usize,
    /// Switch costs (so the scheduler can consider cost).
    switch_costs: &'a SwitchCosts,
}

enum Decision {
    /// Switch to the target model now (after draining in-flight).
    SwitchNow,
    /// Wait this many seconds, then switch unconditionally.
    DeferFor(f64),
    /// Do nothing — another mechanism will handle it (e.g., scheduler).
    Skip,
}

trait SimPolicy {
    /// Called when a request arrives for a non-active model.
    fn on_request(&self, ctx: &PolicyCtx) -> Decision;

    /// Called on each scheduler tick. Return Some(model) to switch.
    /// Default: no scheduler.
    fn on_tick(&self, _ctx: &ScheduleCtx) -> Option<String> {
        None
    }

    /// Whether this policy uses a background scheduler.
    fn has_scheduler(&self) -> bool {
        false
    }

    /// Policy name for output.
    fn name(&self) -> &str;
}

// ── Simulation state ────────────────────────────────────────────────────────

struct SimState {
    events: BinaryHeap<Event>,
    queues: HashMap<String, VecDeque<f64>>,
    active: Option<String>,
    active_since: f64,
    serving: bool,       // true if a request is currently being served
    switching: bool,     // true during a switch (no serving)
    waits: Vec<f64>,     // per-request wait time (arrival → start serving)
    served: usize,
    switches: usize,
    total_switch_time: f64,
    total_serve_time: f64,
    /// Time of the first arrival (start of the workload window).
    first_arrival: f64,
    /// Time when the last request finished being served.
    last_serve_completion: f64,
    /// Tracks which models have a pending DeferFired event, to avoid duplicates.
    defer_pending: HashMap<String, bool>,
}

impl SimState {
    fn new() -> Self {
        Self {
            events: BinaryHeap::new(),
            queues: HashMap::new(),
            active: None,
            active_since: 0.0,
            serving: false,
            switching: false,
            waits: Vec::new(),
            served: 0,
            switches: 0,
            total_switch_time: 0.0,
            total_serve_time: 0.0,
            first_arrival: f64::INFINITY,
            last_serve_completion: 0.0,
            defer_pending: HashMap::new(),
        }
    }

    fn queue_depths(&self) -> HashMap<String, usize> {
        self.queues.iter().map(|(m, q)| (m.clone(), q.len())).collect()
    }

    /// Try to start serving the next request from the active model's queue.
    fn try_serve(&mut self, now: f64, service_time: f64) {
        if self.serving || self.switching {
            return;
        }
        if let Some(ref active) = self.active {
            if let Some(arrived) = self.queues.get_mut(active).and_then(|q| q.pop_front()) {
                self.waits.push(now - arrived);
                self.served += 1;
                self.serving = true;
                self.total_serve_time += service_time;
                let completion = now + service_time;
                self.last_serve_completion = self.last_serve_completion.max(completion);
                self.events.push(Event::Served { time: completion });
            }
        }
    }

    /// Initiate a switch. Requires not currently switching.
    fn start_switch(&mut self, target: &str, now: f64, cost: f64) {
        assert!(!self.switching, "start_switch called while already switching");
        self.switching = true;
        self.serving = false;
        self.switches += 1;
        self.total_switch_time += cost;
        self.events.push(Event::SwitchComplete {
            model: target.to_string(),
            time: now + cost,
        });
    }
}

// ── Results ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SimResult {
    policy: String,
    profile: String,
    total_requests: usize,
    served: usize,
    switches: usize,
    switch_time: f64,
    serve_time: f64,
    wall_time: f64,
    /// serve_time / wall_time — fraction of time GPU was doing useful work.
    gpu_utilization: f64,
    max_wait: f64,
    avg_wait: f64,
    p99_wait: f64,
}

impl std::fmt::Display for SimResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<14} served={}/{:<4} sw={:<3} gpu={:>5.1}% max_w={:>6.1}s p99_w={:>6.1}s avg_w={:>5.1}s",
            self.policy,
            self.served,
            self.total_requests,
            self.switches,
            self.gpu_utilization * 100.0,
            self.max_wait,
            self.p99_wait,
            self.avg_wait,
        )
    }
}

// ── Engine ───────────────────────────────────────────────────────────────────

fn simulate(
    policy: &dyn SimPolicy,
    requests: &[Request],
    switch_costs: &SwitchCosts,
    service_time: f64,
    profile: &str,
) -> SimResult {
    let mut s = SimState::new();

    // Seed arrival events and initialize queues.
    for r in requests {
        s.queues.entry(r.model.clone()).or_default();
        s.events.push(Event::Arrive {
            model: r.model.clone(),
            time: r.arrive_at,
        });
    }

    // Seed scheduler ticks if the policy uses one.
    if policy.has_scheduler() {
        let max_time = requests
            .iter()
            .map(|r| r.arrive_at)
            .fold(0.0_f64, f64::max)
            + 600.0; // generous runway
        let mut t = 0.5;
        while t < max_time {
            s.events.push(Event::Tick { time: t });
            t += 0.5;
        }
    }

    while let Some(event) = s.events.pop() {
        let _now = event.time();


        match event {
            Event::SwitchComplete { model, time } => {
                s.active = Some(model);
                s.active_since = time;
                s.switching = false;
                s.serving = false;
                s.try_serve(time, service_time);
            }

            Event::Served { time } => {
                if s.switching {
                    // A switch started while this serve was in-flight.
                    // The serve completes but we can't start a new one.
                    s.serving = false;
                    continue;
                }
                s.serving = false;
                s.try_serve(time, service_time);
            }

            Event::Arrive { model, time } => {
                s.queues.get_mut(&model).unwrap().push_back(time);
                s.first_arrival = s.first_arrival.min(time);

                if s.switching {
                    continue;
                }

                // If this is for the active model and we're idle, start serving.
                if s.active.as_deref() == Some(&model) {
                    s.try_serve(time, service_time);
                    continue;
                }

                // Cold start: no active model yet — switch immediately, no cost.
                if s.active.is_none() {
                    s.active = Some(model.clone());
                    s.active_since = time;
                    s.try_serve(time, service_time);
                    continue;
                }

                // Non-active model — ask the policy.
                let active = s.active.as_deref().unwrap();
                let cost = switch_costs.cost(active, &model);
                let ctx = PolicyCtx {
                    target_model: &model,
                    active_model: active,
                    target_queue_depth: s.queues.get(&model).map(|q| q.len()).unwrap_or(0),
                    active_queue_depth: s.queues.get(active).map(|q| q.len()).unwrap_or(0),
                    oldest_waiting: s.queues
                        .get(&model)
                        .and_then(|q| q.front())
                        .map(|t| time - t)
                        .unwrap_or(0.0),
                    active_in_flight: usize::from(s.serving),
                    active_duration: time - s.active_since,
                    switch_cost: cost,
                };

                match policy.on_request(&ctx) {
                    Decision::SwitchNow => {
                        // Drain: if serving, wait for it to finish first.
                        // In a discrete sim with single-serve, we can just check.
                        if s.serving {
                            // The Served event will fire. We record a deferred
                            // switch at the current time (meaning: switch ASAP
                            // after drain). The DeferFired will pick it up.
                            if !s.defer_pending.get(&model).copied().unwrap_or(false) {
                                s.defer_pending.insert(model.clone(), true);
                                s.events.push(Event::DeferFired {
                                    model: model.clone(),
                                    time,  // fire immediately — will recheck after Served
                                });
                            }
                        } else {
                            s.start_switch(&model, time, cost);
                        }
                    }
                    Decision::DeferFor(delay) => {
                        if !s.defer_pending.get(&model).copied().unwrap_or(false) {
                            s.defer_pending.insert(model.clone(), true);
                            s.events.push(Event::DeferFired {
                                model: model.clone(),
                                time: time + delay,
                            });
                        }
                    }
                    Decision::Skip => {}
                }
            }

            Event::DeferFired { model, time } => {
                s.defer_pending.insert(model.clone(), false);

                if s.switching {
                    continue;
                }

                // If the target queue is empty, the request was already served
                // (model became active by other means). Drop the defer.
                if s.queues.get(&model).map(|q| q.is_empty()).unwrap_or(true) {
                    continue;
                }

                // If this model is now active, just serve.
                if s.active.as_deref() == Some(&model) {
                    s.try_serve(time, service_time);
                    continue;
                }

                // If currently serving, we need to wait for drain.
                // Re-defer with a tiny delay to check again after the serve completes.
                if s.serving {
                    if !s.defer_pending.get(&model).copied().unwrap_or(false) {
                        s.defer_pending.insert(model.clone(), true);
                        // Schedule just after the next Served event (~service_time from now worst case).
                        // Using a small epsilon so the BinaryHeap fires it after the Served.
                        s.events.push(Event::DeferFired {
                            model: model.clone(),
                            time: time + 0.001,
                        });
                    }
                    continue;
                }

                // Switch now.
                let active = s.active.as_deref().unwrap();
                let cost = switch_costs.cost(active, &model);
                s.start_switch(&model, time, cost);
            }

            Event::Tick { time } => {
                if s.switching {
                    continue;
                }

                let depths = s.queue_depths();
                let ctx = ScheduleCtx {
                    active_model: s.active.as_deref(),
                    active_duration: time - s.active_since,
                    queue_depths: &depths,
                    active_in_flight: usize::from(s.serving),
                    switch_costs,
                };

                if let Some(target) = policy.on_tick(&ctx) {
                    if s.active.as_deref() != Some(target.as_str()) && !s.serving {
                        if let Some(active) = s.active.as_deref() {
                            let cost = switch_costs.cost(active, &target);
                            s.start_switch(&target, time, cost);
                        }
                    }
                }
            }
        }
    }

    // Compute results.
    // Wall time = first arrival to last serve completion. This is the window
    // during which the workload was active. Scheduler ticks after all work is
    // done don't inflate the denominator.
    let wall_time = if s.served > 0 {
        s.last_serve_completion - s.first_arrival
    } else {
        0.0
    };
    let gpu_utilization = if wall_time > 0.0 {
        s.total_serve_time / wall_time
    } else {
        0.0
    };

    let max_wait = s.waits.iter().cloned().fold(0.0_f64, f64::max);
    let avg_wait = if s.waits.is_empty() {
        0.0
    } else {
        s.waits.iter().sum::<f64>() / s.waits.len() as f64
    };
    let p99_wait = percentile(&s.waits, 0.99);

    SimResult {
        policy: policy.name().to_string(),
        profile: profile.to_string(),
        total_requests: requests.len(),
        served: s.served,
        switches: s.switches,
        switch_time: s.total_switch_time,
        serve_time: s.total_serve_time,
        wall_time,
        gpu_utilization,
        max_wait,
        avg_wait,
        p99_wait,
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64 - 1.0) * p).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Policies ────────────────────────────────────────────────────────────────

/// FIFO: switch immediately on first request for a non-active model.
struct Fifo;

impl SimPolicy for Fifo {
    fn on_request(&self, _ctx: &PolicyCtx) -> Decision {
        Decision::SwitchNow
    }
    fn name(&self) -> &str {
        "fifo"
    }
}

// ── Workload ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Request {
    model: String,
    arrive_at: f64,
}

/// LCG-based pseudo-random workload generator.
///
/// Generates `n` requests across `num_models` models with random inter-arrival
/// times. The `bias` parameter controls skew: 0.5 = uniform across models,
/// 0.9 = 90% of requests go to model A.
fn random_workload(n: usize, num_models: usize, bias: f64, rate: f64, seed: u64) -> Vec<Request> {
    let mut state = seed;
    let mut r = Vec::new();
    let mut t = 0.0;
    let models: Vec<String> = (0..num_models).map(|i| ((b'A' + i as u8) as char).to_string()).collect();
    let mean_gap = 1.0 / rate;

    for _ in 0..n {
        // Advance LCG
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        // Pick model: use bias for 2-model case, uniform for N>2
        let model = if num_models == 2 {
            if (state >> 33) as f64 / ((1u64 << 31) as f64) < bias {
                &models[0]
            } else {
                &models[1]
            }
        } else {
            &models[(state >> 33) as usize % num_models]
        };

        // Exponential inter-arrival (approximated via uniform [0, 2*mean])
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let gap = ((state >> 40) as f64 / (1u64 << 24) as f64) * 2.0 * mean_gap;
        t += gap;

        r.push(Request { model: model.clone(), arrive_at: t });
    }
    r
}

// ── Test harness ────────────────────────────────────────────────────────────

fn make_policies() -> Vec<Box<dyn SimPolicy>> {
    vec![
        Box::new(Fifo),
    ]
}

fn run_suite(
    label: &str,
    switch_costs: &SwitchCosts,
    service_time: f64,
    workloads: &[(&str, Vec<Request>)],
) {
    let policies = make_policies();

    println!("\n{}", "=".repeat(110));
    println!("  {}", label);
    println!("{}", "=".repeat(110));

    for (name, reqs) in workloads {
        println!("\n  {} ({} requests):", name, reqs.len());
        for policy in &policies {
            let r = simulate(policy.as_ref(), reqs, switch_costs, service_time, name);
            println!("    {}", r);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn policy_simulation() {
    let workloads: Vec<(&str, Vec<Request>)> = vec![
        // 2 models, balanced, moderate arrival rate
        ("2m_balanced_500",  random_workload(500, 2, 0.5, 5.0, 42)),
        // 2 models, skewed 80/20
        ("2m_skewed_500",    random_workload(500, 2, 0.8, 5.0, 123)),
        // 2 models, heavily skewed 95/5
        ("2m_heavy_500",     random_workload(500, 2, 0.95, 5.0, 456)),
        // 2 models, balanced, high arrival rate (saturated)
        ("2m_saturated_500", random_workload(500, 2, 0.5, 20.0, 789)),
        // 3 models, uniform
        ("3m_uniform_500",   random_workload(500, 3, 0.5, 5.0, 1000)),
    ];

    // Vary switch cost
    for &sc in &[2.0, 10.0, 30.0] {
        run_suite(
            &format!("symmetric sc={}s svc=0.5s", sc),
            &SwitchCosts::uniform(sc),
            0.5,
            &workloads,
        );
    }

    // Asymmetric: A fast to wake (3.6s), B slow (38.5s)
    let costs = SwitchCosts::asymmetric(
        &[("A", "B", 38.5), ("B", "A", 3.6)],
        10.0,
    );
    run_suite(
        "asymmetric A→B=38.5s B→A=3.6s svc=0.5s",
        &costs,
        0.5,
        &workloads,
    );
}
