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

impl StoppingRule for KChange {
    fn should_stop(&self, info: &CycleInfo) -> bool {
        (info.k - info.k_prev).abs() < self.tol
    }
}

/// Stop when a max number of cycles is reached
pub struct MaxCycles {
    max: usize,
}

impl StoppingRule for MaxCycles {
    fn should_stop(&self, info: &CycleInfo) -> bool {
        info.cycle >= self.max
    }
}
