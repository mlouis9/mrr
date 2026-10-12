//! Defines empirical emission source measure

use crate::moments::MomentLayout;

// Struct of arrays that contain properties of atoms templated by point type
pub struct Atoms<P> {
    positions: Vec<P>,
    weights: Vec<f64>,
    moments: Vec<f64>,
    layout: MomentLayout,
}

impl<P> Atoms<P> {
    pub fn new(n_atoms: usize, layout: MomentLayout) -> Self {
        Self {
            positions: Vec::with_capacity(n_atoms),
            weights: Vec::with_capacity(n_atoms),
            moments: layout.new_moments_array(n_atoms),
            layout,
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn push(&mut self, position: P, weight: f64, moment_block: &[f64]) {
        assert_eq!(moment_block.len(), self.layout.block_len());
        self.positions.push(position);
        self.weights.push(weight);
        self.moments.extend_from_slice(moment_block);
    }
}
