//! Code for initializing empirical source

use crate::atoms::Atoms;
use crate::moments::MomentLayout;
use crate::proposal::UniformProposal;
use crate::sampler::SamplerStreams;
use crate::traits::{Geometry, Proposal, Source, Targets};

/// Sample atoms isotropic in angle, white in energy, and uniform in space.
///
/// Atoms are produced in fixed-size chunks, each from its own sampler stream
/// (see `sampler.rs`), and the chunks are then joined in chunk order. Because
/// no chunk depends on another, the `map` below can later become a rayon
/// `into_par_iter()` without changing the result.
pub fn isotropic_white_uniform<G: Geometry>(
    geometry: &G,
    layout: &MomentLayout,
    n_atoms: usize,
    streams: &impl SamplerStreams,
) -> Atoms<G::Point> {
    let block = layout.isotropic_white_block();
    let Targets { points, weights } =
        UniformProposal.sample_target_atoms(geometry, &Source::Initialize(n_atoms), streams);
    Atoms::from_targets(points, weights, *layout, &block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{CHUNK_SIZE, GridStreams1D, RandomStreams, purpose};
    use crate::test_utils::*;

    #[test]
    fn grid_init_is_exact_across_chunks() {
        // More than one chunk, last one partial
        let n = CHUNK_SIZE + 3;
        let geometry = TestSlab::new(2.0, 0.0..1.0);
        let layout = MomentLayout::from_legendre_order(2, 1);
        let atoms = isotropic_white_uniform(&geometry, &layout, n, &GridStreams1D::new(n));

        assert_eq!(atoms.len(), n);
        for (i, &x) in atoms.points().iter().enumerate() {
            assert_eq!(x, (i as f64 + 0.5) / n as f64 * 2.0);
        }
        assert!(
            atoms
                .weights()
                .iter()
                .all(|&w| w == geometry.volume() / n as f64)
        );
        let sum = atoms.weights().iter().sum::<f64>();
        assert_close(sum, geometry.volume());
    }

    /// The isotropic white source integrates to V for any number of groups,
    /// and every atom carries q0 = 1/G in each group (higher moments zero).
    #[track_caller]
    fn check_isotropic_white_source(n_groups: usize) {
        let max_order = 3;
        let n_atoms = CHUNK_SIZE + 3;
        let geometry = TestSlab::new(1.5, 0.0..1.0);
        let layout = MomentLayout::from_legendre_order(max_order, n_groups);
        let streams = RandomStreams {
            base_seed: 1,
            purpose: purpose::INIT,
            cycle: 0,
        };
        let atoms = isotropic_white_uniform(&geometry, &layout, n_atoms, &streams);

        assert_close(atoms.compute_integral(), geometry.volume());
        assert_all_close(atoms.atom_moments(0), &layout.isotropic_white_block());
        assert_close(
            atoms.atom_moments(0)[layout.index(0, 0)],
            1.0 / n_groups as f64,
        );
    }

    #[test]
    fn isotropic_white_source_integrates_to_volume_one_group() {
        check_isotropic_white_source(1);
    }

    #[test]
    fn isotropic_white_source_integrates_to_volume_three_groups() {
        check_isotropic_white_source(3);
    }
}
