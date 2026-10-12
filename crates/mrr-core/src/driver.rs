//! Logic for overall power iteration and simulation

use crate::{
    convergence::{CycleInfo, StoppingRule},
    traits::{Emission, Gather, Medium, Proposal},
};

struct RunResult {
    k: f64,
}

/// Parameters influencing simulation
struct RunConfig {
    rules: Vec<Box<dyn StoppingRule>>,
}

/// Power iteration
fn power_iterate<M: Medium, G: Gather, E: Emission, P: Proposal>(cfg: &RunConfig) -> RunResult {
    let cycle: usize = 0;
    let k: f64 = 0.0;
    let k_prev: f64 = 0.0;
    loop {
        let info: CycleInfo = CycleInfo { cycle, k, k_prev };
        if cfg.rules.iter().any(|r| r.should_stop(&info)) {
            break;
        }

        // Perform iteration here
    }
    RunResult { k: 1.0 }
}
