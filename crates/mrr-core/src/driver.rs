//! Logic for overall power iteration and simulation

use crate::{
    atoms::Atoms,
    convergence::{CycleInfo, StoppingRule},
    init::isotropic_white_uniform,
    moments::MomentLayout,
    sampler::{RandomStreams, purpose},
    traits::{Emission, Gather, Medium, Proposal},
};

// Output of simulation
struct RunResult {
    k: f64,
}

/// Parameters influencing simulation
struct RunConfig {
    rules: Vec<Box<dyn StoppingRule>>,
    n_source_atoms: usize, // Number of atoms in empirical source
    max_legendre_order: usize,
    n_groups: usize,
    seed: u64, // Base seed: every random stream in the run is derived from this
}

/// Power iteration
fn power_iterate<M: Medium, G: Gather, E: Emission, P: Proposal<M>>(
    medium: &M,
    cfg: &RunConfig,
) -> RunResult {
    let cycle: usize = 0;
    let k: f64 = 0.0;
    let k_prev: f64 = 0.0;

    // Initialize empirical source (once, before iterating)
    let layout = MomentLayout::from_legendre_order(cfg.max_legendre_order, cfg.n_groups);
    let init_streams = RandomStreams {
        base_seed: cfg.seed,
        purpose: purpose::INIT,
        cycle: 0,
    };
    let atoms = isotropic_white_uniform(medium, &layout, cfg.n_source_atoms, &init_streams);

    loop {
        let info: CycleInfo = CycleInfo { cycle, k, k_prev };
        if cfg.rules.iter().any(|r| r.should_stop(&info)) {
            break;
        }

        // Sample new points according to proposal
        // (use RandomStreams { base_seed: cfg.seed, purpose: purpose::PROPOSAL, cycle: cycle as u64 })

        // Perform gather (fully abstracted?)

        // Compute new emission source

        // Normalize and compute k
    }
    RunResult { k: 1.0 }
}
