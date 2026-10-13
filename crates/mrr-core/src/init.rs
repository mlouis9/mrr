//! Code for initializing empirical source

use crate::atoms::Atoms;
use crate::moments::MomentLayout;
use crate::proposal::UniformProposal;
use crate::sampler::SamplerStreams;
use crate::traits::{Medium, Proposal, Source, Targets};

/// Sample atoms isotropic in angle, white in energy, and uniform in space.
///
/// Atoms are produced in fixed-size chunks, each from its own sampler stream
/// (see `sampler.rs`), and the chunks are then joined in chunk order. Because
/// no chunk depends on another, the `map` below can later become a rayon
/// `into_par_iter()` without changing the result.
pub fn isotropic_white_uniform<M: Medium>(
    medium: &M,
    layout: &MomentLayout,
    n_atoms: usize,
    streams: &impl SamplerStreams,
) -> Atoms<M::Point> {
    let block = layout.isotropic_white_block();
    let Targets { points, weights } =
        UniformProposal.sample_target_atoms(medium, &Source::Initialize(n_atoms), streams);
    Atoms::from_targets(points, weights, *layout, &block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{CHUNK_SIZE, GridStreams1D, RandomStreams, purpose};
    use crate::test_utils::*;
    use approx;

    #[test]
    fn grid_init_is_exact_across_chunks() {
        // More than one chunk, last one partial
        let n = CHUNK_SIZE + 3;
        let medium = TestSlab::new(2.0, 1, 0.0..1.0);
        let layout = MomentLayout::from_legendre_order(2, 1);
        let atoms = isotropic_white_uniform(&medium, &layout, n, &GridStreams1D { n });

        assert_eq!(atoms.len(), n);
        for (i, &x) in atoms.points().iter().enumerate() {
            assert_eq!(x, (i as f64 + 0.5) / n as f64 * 2.0);
        }
        assert!(
            atoms
                .weights()
                .iter()
                .all(|&w| w == medium.volume() / n as f64)
        );
        let sum = atoms.weights().iter().sum::<f64>();
        approx::assert_relative_eq!(sum, medium.volume(), max_relative = 1e-12,);
    }

    #[test]
    fn isotropic_white_source_integrates_to_volume() {
        #[track_caller]
        fn check_integral(n_groups: usize) {
            let max_order = 3;
            let n_atoms = CHUNK_SIZE + 3;
            let medium = TestSlab::new(1.5, n_groups, 0.0..1.0);
            let layout = MomentLayout::from_legendre_order(max_order, n_groups);
            let streams = RandomStreams {
                base_seed: 1,
                purpose: purpose::INIT,
                cycle: 0,
            };
            let atoms = isotropic_white_uniform(&medium, &layout, n_atoms, &streams);
            assert_close(atoms.compute_integral(), medium.volume());
        }

        check_integral(1);
        check_integral(3);
    }
}
