//! Logic for overall power iteration and simulation

use crate::{
    atoms::Atoms,
    convergence::{CycleInfo, StoppingRule},
    init::isotropic_white_uniform,
    moments::MomentLayout,
    sampler::{RandomStreams, purpose},
    traits::{Emission, Gather, Medium, Proposal, Source, Targets},
};

/// Output of simulation
pub struct RunResult {
    /// Final eigenvalue estimate
    pub k: f64,
    /// Number of power-iteration cycles performed
    pub cycles: usize,
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
pub struct Problem<M, G, E, P> {
    medium: M,
    gather: G,
    emission: E,
    proposal: P,
}

impl<M, G, E, P> Problem<M, G, E, P> {
    pub fn new(medium: M, gather: G, emission: E, proposal: P) -> Self {
        Self {
            medium,
            gather,
            emission,
            proposal,
        }
    }
}

impl<M: Medium, G: Gather<M>, E: Emission<M>, P: Proposal<M>> Problem<M, G, E, P> {
    /// Initialize source
    fn init_src(&self, cfg: &RunConfig) -> Atoms<M::Point> {
        let layout =
            MomentLayout::from_legendre_order(cfg.max_legendre_order, self.medium.n_groups());
        let init_streams = RandomStreams {
            base_seed: cfg.seed,
            purpose: purpose::INIT,
            cycle: 0,
        };
        isotropic_white_uniform(&self.medium, &layout, cfg.n_source_atoms, &init_streams)
    }

    /// Generate proposal atoms
    fn propose(&self, seed: u64, cycle: usize, atoms: &Atoms<M::Point>) -> Atoms<M::Point> {
        let proposal_streams = RandomStreams {
            base_seed: seed,
            purpose: purpose::PROPOSAL,
            cycle,
        };
        let layout = atoms.layout();
        let Targets { points, weights } = self.proposal.sample_target_atoms(
            &self.medium,
            &Source::Existing(atoms),
            &proposal_streams,
        );
        Atoms::from_targets(points, weights, *layout, &layout.zero_block())
    }

    fn gather_flux(&self, atoms: &Atoms<M::Point>, targets: &mut Atoms<M::Point>) {
        for target_idx in 0..targets.len() {
            let point = targets.points()[target_idx];
            let moments = self.gather.gather(&self.medium, atoms, point);
            targets.update_atom_moments(target_idx, &moments);
        }
    }

    fn apply_emission(&self, targets: &mut Atoms<M::Point>) {
        for target_idx in 0..targets.len() {
            let moments = self
                .emission
                .compute_emission(&self.medium, targets.atom_moments(target_idx));
            targets.update_atom_moments(target_idx, &moments);
        }
    }

    fn normalize(&self, targets: &mut Atoms<M::Point>) -> f64 {
        let k = targets.compute_integral() / self.medium.volume();
        targets.scale_moments(1.0 / k);
        k
    }
}

/// Stopping condition for power iteration that stops if any of the rules are true
fn should_stop(cfg: &RunConfig, info: &CycleInfo) -> bool {
    cfg.rules.iter().any(|r| r.should_stop(info))
}

/// Power iteration
pub fn power_iterate<M: Medium, G: Gather<M>, E: Emission<M>, P: Proposal<M>>(
    problem: &Problem<M, G, E, P>,
    cfg: &RunConfig,
) -> RunResult {
    // Initialize loop parameters
    let mut cycle: usize = 0;
    let mut k: f64 = 0.0;
    let mut k_prev: f64 = -1.0; // Ensure loop runs for at least one iter

    // Initialize empirical source (once, before iterating)
    let mut atoms = problem.init_src(cfg);

    // Power iteration
    loop {
        let info = CycleInfo { cycle, k, k_prev };
        if should_stop(cfg, &info) {
            break;
        }

        // Sample new points and weights according to proposal
        let mut targets = problem.propose(cfg.seed, cycle, &atoms);

        // Perform gather
        problem.gather_flux(&atoms, &mut targets);

        // Compute new emission source
        problem.apply_emission(&mut targets);

        // Normalize and compute k
        (k_prev, k) = (k, problem.normalize(&mut targets));
        atoms = targets;
        cycle += 1;
    }
    RunResult { k, cycles: cycle }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convergence::MaxCycles;
    use crate::proposal::UniformProposal;
    use crate::test_utils::*;

    /// With a gather that always returns the isotropic white block and an
    /// identity emission, every cycle reproduces a source of integral V, so
    /// k must be exactly 1 and the run must stop after MaxCycles cycles.
    /// This exercises init, proposal, gather, emission, normalization and
    /// stopping together.
    #[test]
    fn steady_source_gives_k_one_and_stops_at_max_cycles() {
        let n_groups = 3;
        let max_order = 2;
        let medium = TestSlab::new(1.5, n_groups, 0.0..1.0);
        let block = MomentLayout::from_legendre_order(max_order, n_groups).isotropic_white_block();
        let problem = Problem::new(
            medium,
            ConstGather::new(block),
            IdentityEmission::new(),
            UniformProposal,
        );
        let cfg = RunConfig {
            rules: vec![Box::new(MaxCycles::new(5))],
            n_source_atoms: 500,
            max_legendre_order: max_order,
            seed: 11,
        };

        let result = power_iterate(&problem, &cfg);

        assert_eq!(result.cycles, 5);
        assert_close(result.k, 1.0);
    }
}
