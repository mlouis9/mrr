//! Logic for overall power iteration and simulation

use crate::{
    atoms::Atoms,
    convergence::{CycleInfo, StoppingRule},
    init::isotropic_white_uniform,
    moments::MomentLayout,
    sampler::RandomSampler,
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
}

/// Power iteration
fn power_iterate<M: Medium, G: Gather, E: Emission, P: Proposal>(
    medium: &M,
    cfg: &RunConfig,
) -> RunResult {
    let cycle: usize = 0;
    let k: f64 = 0.0;
    let k_prev: f64 = 0.0;
    loop {
        let info: CycleInfo = CycleInfo { cycle, k, k_prev };
        if cfg.rules.iter().any(|r| r.should_stop(&info)) {
            break;
        }

        // Initialize empirical source
        let layout = MomentLayout::from_legendre_order(cfg.max_legendre_order, cfg.n_groups);
        let mut sampler = RandomSampler::new(0);
        let atoms = isotropic_white_uniform(medium, &layout, cfg.n_source_atoms, &mut sampler);

        // Sample new points according to proposal

        // Perform gather (fully abstracted?)

        // Compute new emission source

        // Normalize and compute k
    }
    RunResult { k: 1.0 }
}
