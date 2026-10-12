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

    /// Move all atoms of `other` onto the end of `self` (used to join chunks)
    pub fn append(&mut self, mut other: Atoms<P>) {
        assert_eq!(other.layout.block_len(), self.layout.block_len());
        self.positions.append(&mut other.positions);
        self.weights.append(&mut other.weights);
        self.moments.append(&mut other.moments);
    }

    pub fn positions(&self) -> &[P] {
        &self.positions
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn layout(&self) -> &MomentLayout {
        &self.layout
    }

    /// Return the moment block belonging to atom `idx`.
    pub fn atom_moments(&self, idx: usize) -> &[f64] {
        assert!(idx < self.len());
        &self.moments[self.layout.atom_range(idx)]
    }

    /// Return a mutable reference to the moment block belonging to atom `idx`.
    fn atom_moments_mut(&mut self, idx: usize) -> &mut [f64] {
        assert!(idx < self.len());
        let range = self.layout.atom_range(idx);
        &mut self.moments[range]
    }

    pub fn update_atom_moments(&mut self, idx: usize, moments: &[f64]) {
        assert!(idx < self.len());
        assert_eq!(moments.len(), self.layout.block_len());

        self.atom_moments_mut(idx).copy_from_slice(moments);
    }
}
