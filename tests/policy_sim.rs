//! Discrete-event policy simulator for comparing scheduling algorithms.
//!
//! Models the real system: requests arrive, queue, and trigger policy evaluation.
//! The active model serves requests one at a time. A switch incurs a fixed cost
//! during which no serving happens (GPU is wasted on model loading).
//!
//! Three policies are compared:
//! - **FIFO**: Always switches on request arrival for non-active model.
//! - **CostAware**: Serving window + cost threshold + coalescing. Reactive only.
//! - **TimeSlice**: Drain-first with proactive scheduler. Never preempts; only
//!   switches when active model is idle. Matches production `TimeSlicePolicy`.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Request {
    model: String,
    arrive_at: f64,
}

#[derive(Debug, Clone)]
struct Result {
    policy: String,
    profile: String,
    requests: usize,
    switches: usize,
    switch_time: f64,
    wall_time: f64,
    gpu_pct: f64,
    max_wait: f64,
    avg_wait: f64,
}

impl std::fmt::Display for Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<14} reqs={:<4} sw={:<3} sw_t={:>6.1}s gpu={:>5.1}% max_w={:>6.1}s avg_w={:>5.1}s",
            self.policy, self.requests, self.switches,
            self.switch_time, self.gpu_pct * 100.0,
            self.max_wait, self.avg_wait,
        )
    }
}

#[derive(Debug, Clone)]
enum Event {
    Arrive { model: String, time: f64 },
    Served { time: f64 },
    Tick { time: f64 },
}

impl Event {
    fn time(&self) -> f64 {
        match self {
            Event::Arrive { time, .. } | Event::Served { time } | Event::Tick { time } => *time,
        }
    }
}

// Min-heap by time
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time() == other.time()
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
        other
            .time()
            .partial_cmp(&self.time())
            .unwrap_or(Ordering::Equal)
    }
}

// ── Policy decisions ─────────────────────────────────────────────────────────

enum Decision {
    SwitchNow,
    DeferFor(f64),
    Skip,
}

struct PolicyCtx {
    active_model: Option<String>,
    target_queue_depth: usize,
    oldest_waiting: f64,
    active_in_flight: usize,
    active_duration: f64,
}

struct ScheduleCtx {
    active_model: Option<String>,
    queue_depths: HashMap<String, usize>,
    active_in_flight: usize,
}

trait SimPolicy {
    fn on_request(&self, ctx: &PolicyCtx) -> Decision;
    fn name(&self) -> &str;
    fn has_scheduler(&self) -> bool {
        false
    }
    fn on_tick(&self, _ctx: &ScheduleCtx) -> Option<String> {
        None
    }
}

// ── FIFO ─────────────────────────────────────────────────────────────────────

struct FifoSim;

impl SimPolicy for FifoSim {
    fn on_request(&self, _ctx: &PolicyCtx) -> Decision {
        Decision::SwitchNow
    }
    fn name(&self) -> &str {
        "fifo"
    }
}

// ── CostAware ────────────────────────────────────────────────────────────────

struct CostAwareSim {
    switch_cost: f64,
    max_wait: f64,
    amortization_factor: f64,
    coalesce_window: f64,
}

impl SimPolicy for CostAwareSim {
    fn on_request(&self, ctx: &PolicyCtx) -> Decision {
        if ctx.oldest_waiting >= self.max_wait {
            return Decision::SwitchNow;
        }
        if ctx.active_model.is_none() {
            return Decision::SwitchNow;
        }
        // Serving window: active model must serve at least switch_cost seconds
        if ctx.active_duration < self.switch_cost {
            return Decision::DeferFor(self.switch_cost - ctx.active_duration);
        }
        // Cost threshold
        let required = ((self.amortization_factor * self.switch_cost).ceil() as usize).max(1);
        if ctx.target_queue_depth >= required {
            return Decision::SwitchNow;
        }
        Decision::DeferFor(self.coalesce_window)
    }
    fn name(&self) -> &str {
        "cost_aware"
    }
}

// ── TimeSlice (drain-first + scheduler) ──────────────────────────────────────
//
// The winning algorithm from iterative simulation. Two rules:
// 1. Never switch reactively (except staleness/cold start) → Skip
// 2. Scheduler switches only when active model is completely idle

struct TimeSliceSim {
    max_wait: f64,
}

impl SimPolicy for TimeSliceSim {
    fn on_request(&self, ctx: &PolicyCtx) -> Decision {
        if ctx.oldest_waiting >= self.max_wait {
            return Decision::SwitchNow;
        }
        if ctx.active_model.is_none() {
            return Decision::SwitchNow;
        }
        Decision::Skip
    }
    fn name(&self) -> &str {
        "time_slice"
    }
    fn has_scheduler(&self) -> bool {
        true
    }

    fn on_tick(&self, ctx: &ScheduleCtx) -> Option<String> {
        let active = ctx.active_model.as_deref()?;

        // Only switch when active model has completely drained
        let active_depth = ctx.queue_depths.get(active).copied().unwrap_or(0);
        if active_depth > 0 || ctx.active_in_flight > 0 {
            return None;
        }

        // Pick model with most waiting requests
        ctx.queue_depths
            .iter()
            .filter(|(m, d)| m.as_str() != active && **d > 0)
            .max_by_key(|(_, d)| **d)
            .map(|(m, _)| m.clone())
    }
}

// ── Simulation engine ────────────────────────────────────────────────────────

fn simulate(
    policy: &dyn SimPolicy,
    requests: &[Request],
    switch_cost: f64,
    service_time: f64,
    profile: &str,
) -> Result {
    let mut events = BinaryHeap::new();
    let mut queues: HashMap<String, VecDeque<f64>> = HashMap::new();
    let mut active: Option<String> = None;
    let mut active_since = 0.0_f64;
    let mut serving = false;
    let mut switches = 0_usize;
    let mut total_switch_time = 0.0;
    let mut waits = Vec::new();
    let mut served = 0_usize;
    let mut switching_until: Option<f64> = None;
    let mut switch_target: Option<String> = None;
    let mut deferred: HashMap<String, f64> = HashMap::new();

    for r in requests {
        queues.entry(r.model.clone()).or_default();
        events.push(Event::Arrive {
            model: r.model.clone(),
            time: r.arrive_at,
        });
    }

    if policy.has_scheduler() {
        let max_time = requests.iter().map(|r| r.arrive_at).fold(0.0_f64, f64::max) + 300.0;
        let mut t = 0.5;
        while t < max_time {
            events.push(Event::Tick { time: t });
            t += 0.5;
        }
    }

    while let Some(event) = events.pop() {
        let now = event.time();

        // Complete any pending switch
        if let Some(done) = switching_until {
            if now >= done {
                active = switch_target.take();
                active_since = done;
                switching_until = None;
                serving = false;
                try_serve(
                    &active,
                    &mut queues,
                    &mut events,
                    &mut waits,
                    &mut served,
                    done,
                    service_time,
                    &mut serving,
                );
            }
        }

        match event {
            Event::Arrive { model, time } => {
                queues.get_mut(&model).unwrap().push_back(time);

                if switching_until.is_some() {
                    continue;
                }

                // Active model got a request — start serving if idle
                if active.as_ref() == Some(&model) && !serving {
                    try_serve(
                        &active,
                        &mut queues,
                        &mut events,
                        &mut waits,
                        &mut served,
                        now,
                        service_time,
                        &mut serving,
                    );
                    continue;
                }

                // Non-active model — evaluate policy
                if active.as_ref() != Some(&model) {
                    let ctx = PolicyCtx {
                        active_model: active.clone(),
                        target_queue_depth: queues.get(&model).map(|q| q.len()).unwrap_or(0),
                        oldest_waiting: queues
                            .get(&model)
                            .and_then(|q| q.front())
                            .map(|t| now - t)
                            .unwrap_or(0.0),
                        active_in_flight: usize::from(serving),
                        active_duration: now - active_since,
                    };

                    match policy.on_request(&ctx) {
                        Decision::SwitchNow => {
                            if serving {
                                deferred.insert(model.clone(), now);
                                continue;
                            }
                            do_switch(
                                &model,
                                now,
                                switch_cost,
                                &mut active,
                                &mut switching_until,
                                &mut switch_target,
                                &mut switches,
                                &mut total_switch_time,
                                &mut serving,
                            );
                        }
                        Decision::DeferFor(delay) => {
                            let at = now + delay;
                            if !deferred.contains_key(&model) || deferred[&model] > at {
                                deferred.insert(model.clone(), at);
                            }
                        }
                        Decision::Skip => {}
                    }
                }
            }

            Event::Served { time } => {
                if switching_until.is_some() {
                    continue;
                }
                serving = false;

                // Check deferred switches that are due
                let best = deferred
                    .iter()
                    .filter(|(_, at)| **at <= time)
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(m, _)| m.clone());

                if let Some(target) = best {
                    deferred.remove(&target);
                    if queues.get(&target).map(|q| !q.is_empty()).unwrap_or(false) {
                        do_switch(
                            &target,
                            time,
                            switch_cost,
                            &mut active,
                            &mut switching_until,
                            &mut switch_target,
                            &mut switches,
                            &mut total_switch_time,
                            &mut serving,
                        );
                        continue;
                    }
                }

                // Serve next from active model
                try_serve(
                    &active,
                    &mut queues,
                    &mut events,
                    &mut waits,
                    &mut served,
                    time,
                    service_time,
                    &mut serving,
                );

                // If idle, check for waiting requests on other models
                if !serving {
                    let stalest = queues
                        .iter()
                        .filter(|(m, q)| active.as_ref() != Some(*m) && !q.is_empty())
                        .max_by(|(_, a), (_, b)| {
                            let wa = time - a.front().unwrap();
                            let wb = time - b.front().unwrap();
                            wa.partial_cmp(&wb).unwrap()
                        })
                        .map(|(m, _)| m.clone());

                    if let Some(target) = stalest {
                        deferred.remove(&target);
                        do_switch(
                            &target,
                            time,
                            switch_cost,
                            &mut active,
                            &mut switching_until,
                            &mut switch_target,
                            &mut switches,
                            &mut total_switch_time,
                            &mut serving,
                        );
                    }
                }
            }

            Event::Tick { time } => {
                if switching_until.is_some() {
                    continue;
                }

                let depths: HashMap<String, usize> =
                    queues.iter().map(|(m, q)| (m.clone(), q.len())).collect();

                let ctx = ScheduleCtx {
                    active_model: active.clone(),
                    queue_depths: depths,
                    active_in_flight: usize::from(serving),
                };

                if let Some(target) = policy.on_tick(&ctx) {
                    if active.as_deref() != Some(&target) {
                        if serving {
                            deferred.insert(target, time);
                        } else {
                            deferred.clear();
                            do_switch(
                                &target,
                                time,
                                switch_cost,
                                &mut active,
                                &mut switching_until,
                                &mut switch_target,
                                &mut switches,
                                &mut total_switch_time,
                                &mut serving,
                            );
                        }
                    }
                }
            }
        }
    }

    // Drain remaining after final switch completes
    if let Some(done) = switching_until {
        active = switch_target.take();
        serving = false;
        let mut t = done;
        loop {
            let mut dummy = BinaryHeap::new();
            try_serve(
                &active, &mut queues, &mut dummy, &mut waits, &mut served, t, service_time,
                &mut serving,
            );
            if serving {
                t += service_time;
                serving = false;
            } else {
                break;
            }
        }
    }

    // Drain any stragglers (switching between remaining queues)
    {
        let mut t = requests
            .iter()
            .map(|r| r.arrive_at)
            .fold(0.0_f64, f64::max)
            + 300.0;
        for _ in 0..1000 {
            if served >= requests.len() {
                break;
            }
            let mut dummy = BinaryHeap::new();
            try_serve(
                &active, &mut queues, &mut dummy, &mut waits, &mut served, t, service_time,
                &mut serving,
            );
            if serving {
                t += service_time;
                serving = false;
            }
            if !queues.values().any(|q| !q.is_empty()) {
                break;
            }
            for (m, q) in &queues {
                if !q.is_empty() && active.as_ref() != Some(m) {
                    switches += 1;
                    total_switch_time += switch_cost;
                    t += switch_cost;
                    active = Some(m.clone());
                    break;
                }
            }
        }
    }

    let wall_time = served as f64 * service_time + total_switch_time;

    assert_eq!(
        served,
        requests.len(),
        "{}/{}: served={}, total={}",
        policy.name(),
        profile,
        served,
        requests.len()
    );

    let gpu = if wall_time > 0.0 {
        1.0 - (total_switch_time / wall_time)
    } else {
        1.0
    };
    let max_w = waits.iter().cloned().fold(0.0_f64, f64::max);
    let avg_w = if waits.is_empty() {
        0.0
    } else {
        waits.iter().sum::<f64>() / waits.len() as f64
    };

    Result {
        policy: policy.name().to_string(),
        profile: profile.to_string(),
        requests: requests.len(),
        switches,
        switch_time: total_switch_time,
        wall_time,
        gpu_pct: gpu,
        max_wait: max_w,
        avg_wait: avg_w,
    }
}

fn try_serve(
    active: &Option<String>,
    queues: &mut HashMap<String, VecDeque<f64>>,
    events: &mut BinaryHeap<Event>,
    waits: &mut Vec<f64>,
    served: &mut usize,
    now: f64,
    service_time: f64,
    serving: &mut bool,
) {
    if let Some(a) = active.as_ref() {
        if let Some(arrived) = queues.get_mut(a).and_then(|q| q.pop_front()) {
            waits.push(now - arrived);
            *served += 1;
            *serving = true;
            events.push(Event::Served {
                time: now + service_time,
            });
        }
    }
}

fn do_switch(
    target: &str,
    now: f64,
    switch_cost: f64,
    active: &mut Option<String>,
    switching_until: &mut Option<f64>,
    switch_target: &mut Option<String>,
    switches: &mut usize,
    total_switch_time: &mut f64,
    serving: &mut bool,
) {
    if active.is_some() {
        *switching_until = Some(now + switch_cost);
        *switch_target = Some(target.to_string());
        *switches += 1;
        *total_switch_time += switch_cost;
        *serving = false;
    } else {
        // Cold start — no switch cost
        *active = Some(target.to_string());
        *serving = false;
    }
}

// ── Workload profiles ────────────────────────────────────────────────────────

fn single_model(n: usize) -> Vec<Request> {
    (0..n)
        .map(|i| Request {
            model: "A".into(),
            arrive_at: i as f64 * 0.1,
        })
        .collect()
}

fn balanced(n_pairs: usize) -> Vec<Request> {
    let mut r = Vec::new();
    for i in 0..n_pairs {
        r.push(Request {
            model: "A".into(),
            arrive_at: (i * 2) as f64 * 0.1,
        });
        r.push(Request {
            model: "B".into(),
            arrive_at: (i * 2 + 1) as f64 * 0.1,
        });
    }
    r
}

fn bursty(burst_size: usize, bursts: usize) -> Vec<Request> {
    let mut r = Vec::new();
    for b in 0..bursts {
        let m = if b % 2 == 0 { "A" } else { "B" };
        let base = b as f64 * 5.0;
        for i in 0..burst_size {
            r.push(Request {
                model: m.into(),
                arrive_at: base + i as f64 * 0.01,
            });
        }
    }
    r
}

fn dominant(total: usize, pct: f64) -> Vec<Request> {
    let mut r = Vec::new();
    let b_total = (total as f64 * (1.0 - pct)) as usize;
    let mut bc = 0;
    for i in 0..total {
        let t = i as f64 * 0.05;
        let bt = ((i + 1) as f64 / total as f64 * b_total as f64) as usize;
        if bc < bt && bc < b_total {
            r.push(Request {
                model: "B".into(),
                arrive_at: t,
            });
            bc += 1;
        } else {
            r.push(Request {
                model: "A".into(),
                arrive_at: t,
            });
        }
    }
    r
}

fn interleave(per_model: usize) -> Vec<Request> {
    let mut r = Vec::new();
    for i in 0..per_model {
        r.push(Request {
            model: "A".into(),
            arrive_at: i as f64 * 0.01,
        });
        r.push(Request {
            model: "B".into(),
            arrive_at: i as f64 * 0.01 + 0.001,
        });
    }
    r
}

fn surge(base_rate: usize, surge_size: usize) -> Vec<Request> {
    let mut r = Vec::new();
    let interval = 1.0 / base_rate as f64;
    for i in 0..(base_rate * 10) {
        r.push(Request {
            model: "A".into(),
            arrive_at: i as f64 * interval,
        });
    }
    for i in 0..surge_size {
        r.push(Request {
            model: "B".into(),
            arrive_at: 5.0 + i as f64 * 0.01,
        });
    }
    r
}

fn three_model(per_model: usize) -> Vec<Request> {
    let mut r = Vec::new();
    let models = ["A", "B", "C"];
    for round in 0..2 {
        for (mi, m) in models.iter().enumerate() {
            let base = (round * 3 + mi) as f64 * 3.0;
            for i in 0..per_model {
                r.push(Request {
                    model: m.to_string(),
                    arrive_at: base + i as f64 * 0.1,
                });
            }
        }
    }
    r
}

fn random_mixed(n: usize, seed: u64) -> Vec<Request> {
    let mut state = seed;
    let mut r = Vec::new();
    let mut t = 0.0;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let m = if (state >> 33) % 2 == 0 { "A" } else { "B" };
        t += ((state >> 40) % 200) as f64 / 1000.0;
        r.push(Request {
            model: m.into(),
            arrive_at: t,
        });
    }
    r
}

// ── Test runner ──────────────────────────────────────────────────────────────

fn run_benchmark(switch_cost: f64, service_time: f64, max_wait: f64) {
    let policies: Vec<Box<dyn SimPolicy>> = vec![
        Box::new(FifoSim),
        Box::new(CostAwareSim {
            switch_cost,
            max_wait,
            amortization_factor: 0.5,
            coalesce_window: 2.0,
        }),
        Box::new(TimeSliceSim { max_wait }),
    ];

    let profiles: Vec<(&str, Vec<Request>)> = vec![
        ("single_model", single_model(40)),
        ("balanced", balanced(20)),
        ("bursty", bursty(10, 4)),
        ("dominant", dominant(50, 0.8)),
        ("interleave", interleave(30)),
        ("surge", surge(2, 20)),
        ("three_model", three_model(10)),
        ("random_50", random_mixed(50, 42)),
        ("random_100", random_mixed(100, 123)),
        ("random_200", random_mixed(200, 456)),
        ("dominant_95", dominant(80, 0.95)),
        ("heavy_burst", bursty(50, 6)),
    ];

    println!("\n{}", "=".repeat(110));
    println!(
        "  sc={}s  svc={}s  max_wait={}s",
        switch_cost, service_time, max_wait
    );
    println!("{}", "=".repeat(110));

    let mut all: Vec<Result> = Vec::new();

    for (name, reqs) in &profiles {
        println!("\n  {} ({} requests):", name, reqs.len());
        for policy in &policies {
            let r = simulate(policy.as_ref(), reqs, switch_cost, service_time, name);
            println!("    {}", r);
            all.push(r);
        }
    }

    println!("\n  AGGREGATE (excl single_model):");
    for pname in &["fifo", "cost_aware", "time_slice"] {
        let rs: Vec<&Result> = all
            .iter()
            .filter(|r| r.policy == *pname && r.profile != "single_model")
            .collect();
        let st: f64 = rs.iter().map(|r| r.switch_time).sum();
        let wt: f64 = rs.iter().map(|r| r.wall_time).sum();
        let sw: usize = rs.iter().map(|r| r.switches).sum();
        let gpu = if wt > 0.0 { 1.0 - (st / wt) } else { 1.0 };
        let mw = rs.iter().map(|r| r.max_wait).fold(0.0_f64, f64::max);
        let total_reqs: f64 = rs.iter().map(|r| r.requests as f64).sum();
        let aw: f64 = rs.iter().map(|r| r.avg_wait * r.requests as f64).sum::<f64>() / total_reqs;
        println!(
            "    {:<14} switches={:<4} gpu={:>5.1}% max_wait={:>6.1}s avg_wait={:>5.1}s",
            pname,
            sw,
            gpu * 100.0,
            mw,
            aw
        );
    }
}

#[test]
fn policy_simulation_benchmark() {
    println!("\n{}", "#".repeat(110));
    println!("# POLICY COMPARISON: FIFO vs CostAware vs TimeSlice (drain-first + scheduler)");
    println!("{}", "#".repeat(110));

    // Vary switch cost (2s=L1 warm, 5s=L1 cold, 10s=L2, 20s=L3)
    for &sc in &[2.0, 5.0, 10.0, 20.0] {
        run_benchmark(sc, 0.5, 15.0);
    }

    // Vary service time at sc=10s
    println!("\n{}", "#".repeat(110));
    println!("# SENSITIVITY: vary service_time at sc=10s");
    println!("{}", "#".repeat(110));
    for &svc in &[0.1, 1.0, 2.0] {
        run_benchmark(10.0, svc, 15.0);
    }
}
