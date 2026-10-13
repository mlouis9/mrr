//! Logic for overall power iteration and simulation

use crate::{
    atoms::Atoms,
    convergence::{CycleInfo, StoppingRule},
    init::isotropic_white_uniform,
    moments::MomentLayout,
    sampler::{RandomStreams, purpose},
    traits::{Emission, Gather, Medium, Proposal, Source, Targets},
};

// Output of simulation
pub struct RunResult {
    k: f64,
}

/// Parameters influencing simulation
pub struct RunConfig<P> {
    rules: Vec<Box<dyn StoppingRule>>,
    /// Number of atoms in empirical source
    n_source_atoms: usize,
    max_legendre_order: usize,
    n_groups: usize,
    /// Base seed: every random stream in the run is derived from this
    seed: u64,
    proposal: P,
}

/// Power iteration
pub fn power_iterate<M: Medium, G: Gather<M>, E: Emission<M>, P: Proposal<M>>(
    medium: &M,
    gather: &G,
    emission: &E,
    cfg: &RunConfig<P>,
) -> RunResult {
    let mut cycle: usize = 0;
    let mut k: f64 = 0.0;
    let mut k_prev: f64 = -1.0; // Ensure loop runs for at least one iter

    // Initialize empirical source (once, before iterating)
    let layout = MomentLayout::from_legendre_order(cfg.max_legendre_order, cfg.n_groups);
    let zero_block = layout.zero_block();
    let init_streams = RandomStreams {
        base_seed: cfg.seed,
        purpose: purpose::INIT,
        cycle: 0,
    };
    let mut atoms = isotropic_white_uniform(medium, &layout, cfg.n_source_atoms, &init_streams);
    let proposal = &cfg.proposal;

    // Power iteration
    loop {
        let info = CycleInfo { cycle, k, k_prev };
        if cfg.rules.iter().any(|r| r.should_stop(&info)) {
            break;
        }

        // Sample new points and weights according to proposal
        let proposal_streams = RandomStreams {
            base_seed: cfg.seed,
            purpose: purpose::PROPOSAL,
            cycle,
        };
        let Targets { points, weights } =
            proposal.sample_target_atoms(medium, &Source::Existing(&atoms), &proposal_streams);
        let mut target_atoms = Atoms::new(cfg.n_source_atoms, layout);

        // Create empty atom array from sampled points and weights
        for atom_idx in 0..points.len() {
            target_atoms.push(points[atom_idx], weights[atom_idx], &zero_block);
        }

        // Perform gather
        for target_idx in 0..points.len() {
            let moments = gather.gather(medium, &atoms, points[target_idx]);
            target_atoms.update_atom_moments(target_idx, &moments);
        }

        // Compute new emission source
        for target_idx in 0..points.len() {
            let moments = emission.compute_emission(medium, target_atoms.atom_moments(target_idx));
            target_atoms.update_atom_moments(target_idx, &moments);
        }

        // Normalize and compute k
        k_prev = k;
        k = target_atoms.compute_integral() / medium.volume();
        target_atoms.scale_moments(1.0 / k);
        atoms = target_atoms;
        cycle += 1;
    }
    RunResult { k }
}

#[cfg(test)]
mod tests {
    use super::*;
}
