//! Code for initializing empirical source

use crate::atoms::Atoms;
use crate::moments::MomentLayout;
use crate::sampler::UnitSampler;
use crate::traits::Medium;

/// Sample atoms isotropic in angle, white in energy, and uniform in space
pub fn isotropic_white_uniform<M: Medium>(
    medium: &M,
    layout: &MomentLayout,
    n_atoms: usize,
    sampler: &mut impl UnitSampler,
) -> Atoms<M::Point> {
    let mut atoms = <Atoms<M::Point>>::new(n_atoms, *layout);
    let block = layout.isotropic_white_block();
    while atoms.len() < n_atoms {
        let mut u = vec![0.0; M::SAMPLE_DIM];
        sampler.next_point(&mut u);
        if let Some(point) = medium.map_uniform(&u) {
            atoms.push(point, 1.0 / n_atoms as f64, &block);
        }
    }
    atoms
}
