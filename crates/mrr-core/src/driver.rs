//! Logic for overall power iteration and simulation

use crate::traits::{Emission, Gather, Medium, Proposal};

struct RunResult {
    k: f64,
}

/// Parameters influencing simulation
struct RunConfig {
    n_inactive_cycles: usize,
    n_active_cycles: usize,
}

/// Power iteration
fn power_iterate<M: Medium, G: Gather, E: Emission, P: Proposal>(cfg: &RunConfig) -> RunResult {
    RunResult { k: 1.0 }
}
