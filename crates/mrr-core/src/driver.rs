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
pub struct RunConfig {
    pub rules: Vec<Box<dyn StoppingRule>>,
    /// Number of atoms in empirical source
    pub n_source_atoms: usize,
    pub max_legendre_order: usize,
    /// Base seed: every random stream in the run is derived from this
    pub seed: u64,
}

/// Contains the specification of the "problem" distinct from the parameters (often compute-dependent
/// numerical parameters). Note for some fields this distinction is somewhat contrived, but it is
/// useful for managing complexity regardless
pub struct Problem<M: Medium, G: Gather<M>, E: Emission<M>, P: Proposal<M>> {
    medium: M,
    gather: G,
    emission: E,
    proposal: P,
}

// Stopping condition for power iteration that stops if any of the rules are true
fn should_stop(cfg: &RunConfig, info: &CycleInfo) -> bool {
    cfg.rules.iter().any(|r| r.should_stop(info))
}

// Generate proposal atoms
fn propose<M: Medium, P: Proposal<M>>(
    seed: u64,
    cycle: usize,
    medium: &M,
    proposal: &P,
    atoms: &Atoms<M::Point>,
    zero_block: &[f64],
) -> Atoms<M::Point> {
    let proposal_streams = RandomStreams {
        base_seed: seed,
        purpose: purpose::PROPOSAL,
        cycle,
    };
    let layout = atoms.layout();
    let Targets { points, weights } =
        proposal.sample_target_atoms(medium, &Source::Existing(atoms), &proposal_streams);
    Atoms::from_targets(points.clone(), weights, *layout, zero_block)
}

fn gather_flux<M: Medium, G: Gather<M>>(
    medium: &M,
    gather: &G,
    points: &Vec<M::Point>,
    atoms: &Atoms<M::Point>,
    targets: &mut Atoms<M::Point>,
) {
    for (target_idx, &point) in points.iter().enumerate() {
        let moments = gather.gather(medium, atoms, point);
        targets.update_atom_moments(target_idx, &moments);
    }
}

fn apply_emission<M: Medium, E: Emission<M>>(
    medium: &M,
    emission: &E,
    targets: &mut Atoms<M::Point>,
) {
    for target_idx in 0..targets.len() {
        let moments = emission.compute_emission(medium, targets.atom_moments(target_idx));
        targets.update_atom_moments(target_idx, &moments);
    }
}

fn normalize<M: Medium>(medium: &M, targets: &mut Atoms<M::Point>) -> f64 {
    let k = targets.compute_integral() / medium.volume();
    targets.scale_moments(1.0 / k);
    k
}

/// Power iteration
pub fn power_iterate<M: Medium, G: Gather<M>, E: Emission<M>, P: Proposal<M>>(
    problem: &Problem<M, G, E, P>,
    cfg: &RunConfig,
) -> RunResult {
    // Unpack problem
    let Problem {
        medium,
        gather,
        emission,
        proposal,
    } = problem;

    // Initialize loop parameters
    let mut cycle: usize = 0;
    let mut k: f64 = 0.0;
    let mut k_prev: f64 = -1.0; // Ensure loop runs for at least one iter

    // Initialize empirical source (once, before iterating)
    let layout = MomentLayout::from_legendre_order(cfg.max_legendre_order, medium.n_groups());
    let zero_block = layout.zero_block();
    let init_streams = RandomStreams {
        base_seed: cfg.seed,
        purpose: purpose::INIT,
        cycle: 0,
    };
    let mut atoms = isotropic_white_uniform(medium, &layout, cfg.n_source_atoms, &init_streams);

    // Power iteration
    loop {
        let info = CycleInfo { cycle, k, k_prev };
        if should_stop(cfg, &info) {
            break;
        }

        // Sample new points and weights according to proposal
        let mut targets = propose(cfg.seed, cycle, medium, proposal, &atoms, &zero_block);
        let points = targets.points().to_vec();

        // Perform gather
        gather_flux(medium, gather, &points, &atoms, &mut targets);

        // Compute new emission source
        apply_emission(medium, emission, &mut targets);

        // Normalize and compute k
        (k_prev, k) = (k, normalize(medium, &mut targets));
        atoms = targets;
        cycle += 1;
    }
    RunResult { k }
}

#[cfg(test)]
mod tests {
    use super::*;
}
