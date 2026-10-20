//! Logic for overall power iteration and simulation

use crate::{
    atoms::Atoms,
    convergence::{CycleInfo, StoppingRule},
    init::isotropic_white_uniform,
    moments::MomentLayout,
    sampler::{AnyStreams, SamplerKind, purpose},
    traits::{Emission, Emitted, Gather, Geometry, Proposal, Source, Targets},
};

/// Output of simulation
pub struct RunResult {
    /// Final eigenvalue estimate
    pub k: f64,
    /// Number of power-iteration cycles performed
    pub cycles: usize,
}

/// The state of the empirical source after a cycle, handed to the
/// `on_cycle` callback of [`power_iterate_with`].
///
/// `cycle` counts completed cycles: 0 is the initial source (where `k` is
/// `None`), 1 is the source after the first gather/emission, and so on.
/// The atoms are borrowed, so a callback that wants to keep them must copy
/// what it needs (e.g. write them to a file).
pub struct CycleSnapshot<'a, P> {
    pub cycle: usize,
    pub k: Option<f64>,
    pub atoms: &'a Atoms<P>,
}

/// How the eigenvalue is updated each cycle.
///
/// Both use the emission parts at the new nodes, `Φ(Q_s)` and `Φ(Q_f)` (fission
/// without `1/k`), with the source normalized so that `Φ(Q) = V`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KUpdate {
    /// `k = Φ(Q_f) / (V − Φ(Q_s))`: the k that makes the new source integrate
    /// to `V` exactly. Lowest noise when every node value is moderate, but the
    /// denominator can turn negative when one node scores very high (a node
    /// next to an atom under a non-pole-matched proposal in 3D), which aborts
    /// the run.
    Split,
    /// `λ = (Φ(Q_s) + Φ(Q_f)/k_n) / V`, `k_{n+1} = k_n λ`, then the source is
    /// divided by `λ`. The standard power-iteration update: always positive, so
    /// robust to heavy-tailed node values. `initial` is `k_0`.
    Multiplicative { initial: f64 },
}

/// Parameters influencing simulation
pub struct RunConfig {
    pub rules: Vec<Box<dyn StoppingRule>>,
    /// Number of atoms in empirical source
    pub n_source_atoms: usize,
    /// Base seed: every random stream in the run is derived from this
    pub seed: u64,
    /// Eigenvalue update rule
    pub k_update: KUpdate,
    /// Pseudo-random (MC) or scrambled Sobol (RQMC) point sets for every
    /// draw: initialization and every cycle's proposal
    pub sampler: SamplerKind,
}

/// Contains the specification of the "problem" distinct from the parameters (often compute-dependent
/// numerical parameters). Note for some fields this distinction is somewhat contrived, but it is
/// useful for managing complexity regardless
pub struct Problem<G, K, E, P> {
    geometry: G,
    /// Moment block carried by each atom (angular order x energy bins). Set by
    /// whoever builds the physics, since it knows the energy representation.
    layout: MomentLayout,
    gather: K,
    emission: E,
    proposal: P,
}

impl<G, K, E, P> Problem<G, K, E, P> {
    pub fn new(geometry: G, layout: MomentLayout, gather: K, emission: E, proposal: P) -> Self {
        Self {
            geometry,
            layout,
            gather,
            emission,
            proposal,
        }
    }
}

impl<G: Geometry, K: Gather<G>, E: Emission<G>, P: Proposal<G>> Problem<G, K, E, P> {
    /// Initialize source
    fn init_src(&self, cfg: &RunConfig) -> Atoms<G::Point> {
        let init_streams = AnyStreams::new(cfg.sampler, cfg.seed, purpose::INIT, 0);
        isotropic_white_uniform(
            &self.geometry,
            &self.layout,
            cfg.n_source_atoms,
            &init_streams,
        )
    }

    /// Generate proposal atoms
    fn propose(&self, cfg: &RunConfig, cycle: usize, atoms: &Atoms<G::Point>) -> Atoms<G::Point> {
        let n_targets = cfg.n_source_atoms;
        let proposal_streams = AnyStreams::new(cfg.sampler, cfg.seed, purpose::PROPOSAL, cycle);
        let layout = atoms.layout();
        let Targets { points, weights } = self.proposal.sample_target_atoms(
            &self.geometry,
            &Source::Existing(atoms, n_targets),
            &proposal_streams,
        );
        Atoms::from_targets(points, weights, *layout, &layout.zero_block())
    }

    fn gather_flux(&self, atoms: &Atoms<G::Point>, targets: &mut Atoms<G::Point>) {
        for target_idx in 0..targets.len() {
            let point = targets.points()[target_idx];
            let moments = self.gather.gather(&self.geometry, atoms, point);
            targets.update_atom_moments(target_idx, &moments);
        }
    }

    /// Replace each target's flux moments by its emission moments and return k.
    ///
    /// Emission gives scattering (Q_s) and fission (Q_f, without 1/k) parts;
    /// see [`KUpdate`] for how they are combined. Either way the new source
    /// integrates to V. `k_current` is the estimate before this cycle (used
    /// by the multiplicative update).
    fn emit_and_normalize(
        &self,
        targets: &mut Atoms<G::Point>,
        update: KUpdate,
        k_current: f64,
    ) -> f64 {
        let layout = self.layout;
        let mut emitted: Vec<Emitted> = Vec::with_capacity(targets.len());
        let (mut phi_s, mut phi_f) = (0.0, 0.0);
        for target_idx in 0..targets.len() {
            let point = targets.points()[target_idx];
            let weight = targets.weights()[target_idx];
            let e = self.emission.compute_emission(
                &self.geometry,
                point,
                targets.atom_moments(target_idx),
            );
            phi_s += weight * layout.block_integral(&e.scattering);
            phi_f += weight * layout.block_integral(&e.fission);
            emitted.push(e);
        }

        let volume = self.geometry.volume();
        // (k used to combine the parts, scale applied to the result, new k)
        let (k_mix, scale, k_new) = match update {
            KUpdate::Split => {
                let denom = volume - phi_s;
                assert!(
                    phi_f > 0.0 && denom > 0.0,
                    "cannot determine k: Phi(Q_f) = {phi_f:e}, V - Phi(Q_s) = {denom:e} \
                     (heavy-tailed node values? consider KUpdate::Multiplicative)"
                );
                let k = phi_f / denom;
                (k, 1.0, k)
            }
            KUpdate::Multiplicative { .. } => {
                let lambda = (phi_s + phi_f / k_current) / volume;
                assert!(
                    lambda > 0.0 && lambda.is_finite(),
                    "cannot update k: emission integral ratio {lambda:e}"
                );
                (k_current, 1.0 / lambda, k_current * lambda)
            }
        };

        for (target_idx, e) in emitted.iter().enumerate() {
            let moments: Vec<f64> = e
                .scattering
                .iter()
                .zip(&e.fission)
                .map(|(s, f)| scale * (s + f / k_mix))
                .collect();
            targets.update_atom_moments(target_idx, &moments);
        }
        k_new
    }
}

/// Stopping condition for power iteration that stops if any of the rules are true
fn should_stop(cfg: &RunConfig, info: &CycleInfo) -> bool {
    cfg.rules.iter().any(|r| r.should_stop(info))
}

/// Power iteration
pub fn power_iterate<G: Geometry, K: Gather<G>, E: Emission<G>, P: Proposal<G>>(
    problem: &Problem<G, K, E, P>,
    cfg: &RunConfig,
) -> RunResult {
    power_iterate_with(problem, cfg, |_| {})
}

/// Power iteration that calls `on_cycle` with the initial source and with
/// the source after every cycle. Core does no I/O itself: tallies, logging
/// and snapshot files all live in the callback.
pub fn power_iterate_with<G: Geometry, K: Gather<G>, E: Emission<G>, P: Proposal<G>>(
    problem: &Problem<G, K, E, P>,
    cfg: &RunConfig,
    mut on_cycle: impl FnMut(&CycleSnapshot<G::Point>),
) -> RunResult {
    // Initialize loop parameters
    let mut cycle: usize = 0;
    let mut k: f64 = 0.0;
    let mut k_prev: f64 = -1.0; // Ensure loop runs for at least one iter

    // Initialize empirical source (once, before iterating)
    let mut atoms = problem.init_src(cfg);
    on_cycle(&CycleSnapshot {
        cycle,
        k: None,
        atoms: &atoms,
    });

    // Power iteration
    loop {
        let info = CycleInfo { cycle, k, k_prev };
        if should_stop(cfg, &info) {
            break;
        }

        // Sample new points and weights according to proposal
        let mut targets = problem.propose(cfg, cycle, &atoms);

        // Perform gather
        problem.gather_flux(&atoms, &mut targets);

        // Compute new emission source, k, and normalize
        let k_current = match (cycle, cfg.k_update) {
            (0, KUpdate::Multiplicative { initial }) => initial,
            _ => k,
        };
        (k_prev, k) = (
            k,
            problem.emit_and_normalize(&mut targets, cfg.k_update, k_current),
        );
        atoms = targets;
        cycle += 1;
        on_cycle(&CycleSnapshot {
            cycle,
            k: Some(k),
            atoms: &atoms,
        });
    }
    RunResult { k, cycles: cycle }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convergence::MaxCycles;
    use crate::proposal::UniformProposal;
    use crate::test_utils::*;

    /// Run with a gather that always returns the isotropic white block (whose
    /// integral over the source is V) and the given emission split.
    fn run(scatter: f64, fission: f64, n_groups: usize) -> (RunResult, f64) {
        let max_order = 2;
        let geometry = TestSlab::new(1.5, 0.0..1.0);
        let volume = geometry.volume();
        let layout = MomentLayout::from_legendre_order(max_order, n_groups);
        let problem = Problem::new(
            geometry,
            layout,
            ConstGather::new(layout.isotropic_white_block()),
            ScaledEmission::new(scatter, fission),
            UniformProposal,
        );
        let cfg = RunConfig {
            rules: vec![Box::new(MaxCycles::new(5))],
            n_source_atoms: 500,
            seed: 11,
            k_update: KUpdate::Split,
            sampler: Default::default(),
        };
        (power_iterate(&problem, &cfg), volume)
    }

    /// Fission-only emission of the gathered flux reproduces a source of
    /// integral V, so k must be exactly 1. This exercises init, proposal,
    /// gather, emission, normalization and stopping together.
    #[test]
    fn steady_source_gives_k_one_and_stops_at_max_cycles() {
        let (result, _) = run(0.0, 1.0, 3);
        assert_eq!(result.cycles, 5);
        assert_close(result.k, 1.0);
    }

    /// With Phi(flux) = V, Q_s = c * flux and Q_f = f * flux give
    /// k = f V / (V - c V) = f / (1 - c).
    #[test]
    fn k_follows_scatter_fission_split() {
        let (result, _) = run(0.5, 0.6, 2);
        assert_close(result.k, 0.6 / (1.0 - 0.5));
    }

    #[test]
    fn callback_sees_initial_source_and_every_cycle() {
        let geometry = TestSlab::new(1.0, 0.0..1.0);
        let layout = MomentLayout::from_legendre_order(0, 1);
        let problem = Problem::new(
            geometry,
            layout,
            ConstGather::new(layout.isotropic_white_block()),
            ScaledEmission::new(0.5, 0.6),
            UniformProposal,
        );
        let cfg = RunConfig {
            rules: vec![Box::new(MaxCycles::new(3))],
            n_source_atoms: 10,
            seed: 1,
            k_update: KUpdate::Split,
            sampler: Default::default(),
        };
        let mut seen = Vec::new();
        let result = power_iterate_with(&problem, &cfg, |snap| {
            seen.push((snap.cycle, snap.k, snap.atoms.len()));
        });
        assert_eq!(seen.len(), 4);
        assert_eq!(seen[0], (0, None, 10));
        for (i, &(cycle, k, n)) in seen.iter().enumerate().skip(1) {
            assert_eq!(cycle, i);
            assert_close(k.unwrap(), 1.2);
            assert_eq!(n, 10);
        }
        assert_eq!(seen.last().unwrap().1, Some(result.k));
    }

    /// With Φ(flux) = V, Q_s = c·flux, Q_f = f·flux: λ = c + f/k_n, so the
    /// multiplicative update k_{n+1} = c k_n + f converges geometrically to
    /// f/(1 − c), the same fixed point as the split update.
    #[test]
    fn multiplicative_update_converges_to_the_same_k() {
        let geometry = TestSlab::new(1.5, 0.0..1.0);
        let layout = MomentLayout::from_legendre_order(1, 2);
        let problem = Problem::new(
            geometry,
            layout,
            ConstGather::new(layout.isotropic_white_block()),
            ScaledEmission::new(0.5, 0.6),
            UniformProposal,
        );
        let cfg = RunConfig {
            rules: vec![Box::new(MaxCycles::new(60))],
            n_source_atoms: 50,
            seed: 3,
            k_update: KUpdate::Multiplicative { initial: 1.0 },
            sampler: Default::default(),
        };
        let mut ks = Vec::new();
        let result = power_iterate_with(&problem, &cfg, |s| ks.extend(s.k));
        // First step from k_0 = 1: k_1 = 0.5·1 + 0.6
        assert_close(ks[0], 1.1);
        assert_close(result.k, 0.6 / (1.0 - 0.5));
    }

    #[test]
    #[should_panic(expected = "cannot determine k")]
    fn pure_scattering_has_no_k() {
        run(0.9, 0.0, 1);
    }
}
