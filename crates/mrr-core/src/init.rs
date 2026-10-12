//! Code for initializing empirical source

use crate::atoms::Atoms;
use crate::moments::MomentLayout;
use crate::sampler::{CHUNK_SIZE, SamplerStreams, UnitSampler};
use crate::traits::Medium;

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
    let weight = 1.0 / n_atoms as f64;
    let n_chunks = n_atoms.div_ceil(CHUNK_SIZE);

    let chunks: Vec<Atoms<M::Point>> = (0..n_chunks)
        .map(|chunk| {
            // The last chunk may be smaller than CHUNK_SIZE
            let n_here = CHUNK_SIZE.min(n_atoms - chunk * CHUNK_SIZE);
            let mut sampler = streams.stream(chunk);
            let mut part = Atoms::new(n_here, *layout);
            let mut u = vec![0.0; M::SAMPLE_DIM];
            while part.len() < n_here {
                sampler.next_point(&mut u);
                if let Some(point) = medium.map_uniform(&u) {
                    part.push(point, weight, &block);
                }
            }
            part
        })
        .collect();

    // Join in chunk order (rayon's ordered `collect` would preserve this too)
    let mut atoms = Atoms::new(n_atoms, *layout);
    for part in chunks {
        atoms.append(part);
    }
    atoms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::GridStreams1D;

    /// Minimal 1D medium on [0, width) that accepts every sample
    struct TestSlab {
        width: f64,
    }

    impl Medium for TestSlab {
        type Point = f64;
        const SAMPLE_DIM: usize = 1;
        fn n_groups(&self) -> usize { 1 }
        fn optical_depth(&self, _: f64, _: f64) -> f64 { 0.0 }
        fn volume(&self) -> f64 { self.width }
        fn map_uniform(&self, u: &[f64]) -> Option<f64> { Some(u[0] * self.width) }
    }

    #[test]
    fn grid_init_is_exact_across_chunks() {
        // More than one chunk, last one partial
        let n = CHUNK_SIZE + 3;
        let medium = TestSlab { width: 2.0 };
        let layout = MomentLayout::from_legendre_order(2, 1);
        let atoms = isotropic_white_uniform(&medium, &layout, n, &GridStreams1D { n });

        assert_eq!(atoms.len(), n);
        for (i, &x) in atoms.positions().iter().enumerate() {
            assert_eq!(x, (i as f64 + 0.5) / n as f64 * 2.0);
        }
        assert!(atoms.weights().iter().all(|&w| w == 1.0 / n as f64));
    }
}
