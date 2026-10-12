//! Logic for overall power iteration and simulation

use crate::{
    atoms::Atoms,
    convergence::{CycleInfo, StoppingRule},
    init::isotropic_white_uniform,
    moments::MomentLayout,
    sampler::{RandomStreams, purpose},
    traits::{Emission, Gather, Medium, Proposal, Source},
};

// Output of simulation
struct RunResult {
    k: f64,
}

/// Parameters influencing simulation
struct RunConfig<P> {
    rules: Vec<Box<dyn StoppingRule>>,
    n_source_atoms: usize, // Number of atoms in empirical source
    max_legendre_order: usize,
    n_groups: usize,
    seed: u64, // Base seed: every random stream in the run is derived from this
    proposal: P,
}

/// Power iteration
fn power_iterate<M: Medium, G: Gather<M>, E: Emission, P: Proposal<M>>(
    medium: &M,
    gather: &G,
    cfg: &RunConfig<P>,
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
    let proposal = &cfg.proposal;

    loop {
        let info: CycleInfo = CycleInfo { cycle, k, k_prev };
        if cfg.rules.iter().any(|r| r.should_stop(&info)) {
            break;
        }

        // Sample new points and weights according to proposal
        let proposal_streams = RandomStreams {
            base_seed: cfg.seed,
            purpose: purpose::PROPOSAL,
            cycle: cycle,
        };
        let (points, weights) =
            proposal.sample_target_atoms(medium, &Source::Existing(&atoms), &proposal_streams);
        let mut target_atoms = Atoms::new(cfg.n_source_atoms, layout.clone());

        // Create empty atom array from sampled points and weights
        for atom_idx in 0..cfg.n_source_atoms {
            target_atoms.push(points[atom_idx], weights[atom_idx], &layout.zero_block());
        }

        // Perform gather
        for target_idx in 0..cfg.n_source_atoms {
            let moments = gather.gather(&atoms, &target_atoms, target_idx);
            target_atoms.update_atom_moments(target_idx, &moments);
        }

        // Compute new emission source

        // Normalize and compute k
    }
    RunResult { k: 1.0 }
}
