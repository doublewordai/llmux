//! Empirical switch cost tracking using exponential moving average.
//!
//! Records the wall-clock duration of each model switch and maintains a
//! smoothed estimate per (from, to) pair. Cold starts (switching from Idle)
//! are tracked separately from model-to-model switches.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;

/// Key for the cost matrix: (from_model, to_model).
/// `None` in the first position means a cold start from Idle.
type CostKey = (Option<String>, String);

/// Tracks empirical switch costs using exponential moving average.
///
/// Thread-safe via `std::sync::RwLock` — writes are cheap (no await points)
/// and already serialized by the switcher's `switch_lock`.
pub struct SwitchCostTracker {
    /// EMA smoothing factor (0..1). Higher = more weight on recent samples.
    alpha: f64,
    costs: RwLock<HashMap<CostKey, Duration>>,
}

impl SwitchCostTracker {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            costs: RwLock::new(HashMap::new()),
        }
    }

    /// Record an observed switch duration.
    pub fn record(&self, from: Option<&str>, to: &str, duration: Duration) {
        let key = (from.map(str::to_string), to.to_string());
        let mut costs = self.costs.write().unwrap();
        match costs.get_mut(&key) {
            Some(existing) => {
                let old = existing.as_secs_f64();
                let new = duration.as_secs_f64();
                *existing = Duration::from_secs_f64(self.alpha * new + (1.0 - self.alpha) * old);
            }
            None => {
                costs.insert(key, duration);
            }
        }
    }

    /// Estimated cost of switching from one model to another.
    /// Returns `None` if no switch between this pair has been observed.
    pub fn estimate(&self, from: Option<&str>, to: &str) -> Option<Duration> {
        let key = (from.map(str::to_string), to.to_string());
        self.costs.read().unwrap().get(&key).copied()
    }

    /// Estimated costs from the given model to all observed targets.
    pub fn estimates_from(&self, from: Option<&str>) -> HashMap<String, Duration> {
        let costs = self.costs.read().unwrap();
        let from_owned = from.map(str::to_string);
        costs
            .iter()
            .filter(|((f, _), _)| *f == from_owned)
            .map(|((_, to), &cost)| (to.clone(), cost))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_observation_is_exact() {
        let tracker = SwitchCostTracker::new(0.3);
        tracker.record(Some("a"), "b", Duration::from_secs(10));
        assert_eq!(tracker.estimate(Some("a"), "b"), Some(Duration::from_secs(10)));
    }

    #[test]
    fn ema_smooths_subsequent_observations() {
        let tracker = SwitchCostTracker::new(0.5);
        tracker.record(Some("a"), "b", Duration::from_secs(10));
        tracker.record(Some("a"), "b", Duration::from_secs(20));
        // EMA: 0.5 * 20 + 0.5 * 10 = 15
        let est = tracker.estimate(Some("a"), "b").unwrap();
        assert!((est.as_secs_f64() - 15.0).abs() < 0.001);
    }

    #[test]
    fn cold_start_tracked_separately() {
        let tracker = SwitchCostTracker::new(0.3);
        tracker.record(None, "a", Duration::from_secs(5));
        tracker.record(Some("b"), "a", Duration::from_secs(15));

        assert_eq!(tracker.estimate(None, "a"), Some(Duration::from_secs(5)));
        assert_eq!(tracker.estimate(Some("b"), "a"), Some(Duration::from_secs(15)));
    }

    #[test]
    fn unknown_pair_returns_none() {
        let tracker = SwitchCostTracker::new(0.3);
        assert_eq!(tracker.estimate(Some("a"), "b"), None);
    }

    #[test]
    fn estimates_from_filters_correctly() {
        let tracker = SwitchCostTracker::new(0.3);
        tracker.record(Some("a"), "b", Duration::from_secs(10));
        tracker.record(Some("a"), "c", Duration::from_secs(20));
        tracker.record(Some("b"), "a", Duration::from_secs(5));

        let from_a = tracker.estimates_from(Some("a"));
        assert_eq!(from_a.len(), 2);
        assert_eq!(from_a["b"], Duration::from_secs(10));
        assert_eq!(from_a["c"], Duration::from_secs(20));

        let from_b = tracker.estimates_from(Some("b"));
        assert_eq!(from_b.len(), 1);
    }
}
