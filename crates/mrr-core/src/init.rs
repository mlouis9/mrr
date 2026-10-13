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
    use crate::sampler::{CHUNK_SIZE, GridStreams1D};
    use approx;

    /// Minimal 1D medium on [0, width) that accepts every sample
    struct TestSlab {
        width: f64,
    }

    impl Medium for TestSlab {
        type Point = f64;
        const SAMPLE_DIM: usize = 1;
        fn n_groups(&self) -> usize {
            1
        }
        fn optical_depth(&self, _: &f64, _: &f64) -> f64 {
            0.0
        }
        fn volume(&self) -> f64 {
            self.width
        }
        fn map_uniform(&self, u: &[f64]) -> Option<f64> {
            Some(u[0] * self.width)
        }
    }

    #[test]
    fn grid_init_is_exact_across_chunks() {
        // More than one chunk, last one partial
        let n = CHUNK_SIZE + 3;
        let medium = TestSlab { width: 2.0 };
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
}
