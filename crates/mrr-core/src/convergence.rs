//! Generic convergence criteria for power iteration

/// Cycle information used to determine stopping
pub struct CycleInfo {
    pub cycle: usize,
    pub k: f64,
    pub k_prev: f64,
}

/// Any type that can decide to stop or keep going
pub trait StoppingRule {
    fn should_stop(&self, info: &CycleInfo) -> bool;
}

/// Stop when k stops changing
pub struct KChange {
    tol: f64,
}

impl KChange {
    pub fn new(tol: f64) -> Self {
        Self { tol }
    }
}

impl StoppingRule for KChange {
    fn should_stop(&self, info: &CycleInfo) -> bool {
        (info.k - info.k_prev).abs() < self.tol
    }
}

/// Stop when a max number of cycles is reached
pub struct MaxCycles {
    max: usize,
}

impl MaxCycles {
    pub fn new(max: usize) -> Self {
        Self { max }
    }
}

impl StoppingRule for MaxCycles {
    fn should_stop(&self, info: &CycleInfo) -> bool {
        info.cycle >= self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(cycle: usize, k: f64, k_prev: f64) -> CycleInfo {
        CycleInfo { cycle, k, k_prev }
    }

    #[test]
    fn max_cycles_stops_exactly_at_max() {
        let rule = MaxCycles::new(5);
        assert!(!rule.should_stop(&info(4, 1.0, 1.0)));
        assert!(rule.should_stop(&info(5, 1.0, 1.0)));
    }

    #[test]
    fn k_change_stops_only_below_tolerance() {
        let rule = KChange::new(1e-3);
        assert!(!rule.should_stop(&info(3, 1.0, 1.01)));
        assert!(rule.should_stop(&info(3, 1.0, 1.0005)));
    }

    #[test]
    fn k_change_does_not_stop_on_first_cycle() {
        // The driver starts with k = 0, k_prev = -1 so no tolerance can trigger
        assert!(!KChange::new(1e-3).should_stop(&info(0, 0.0, -1.0)));
    }
}
