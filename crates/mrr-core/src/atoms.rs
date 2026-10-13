//! Defines empirical emission source measure

use crate::moments::MomentLayout;

// Struct of arrays that contain properties of atoms templated by point type
pub struct Atoms<P> {
    points: Vec<P>,
    weights: Vec<f64>,
    moments: Vec<f64>,
    layout: MomentLayout,
}

impl<P> Atoms<P> {
    pub fn new(n_atoms: usize, layout: MomentLayout) -> Self {
        Self {
            points: Vec::with_capacity(n_atoms),
            weights: Vec::with_capacity(n_atoms),
            moments: layout.new_moments_array(n_atoms),
            layout,
        }
    }

    /// The number of atoms
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn push(&mut self, point: P, weight: f64, moment_block: &[f64]) {
        assert_eq!(moment_block.len(), self.layout.block_len());
        self.points.push(point);
        self.weights.push(weight);
        self.moments.extend_from_slice(moment_block);
    }

    /// Move all atoms of `other` onto the end of `self` (used to join chunks)
    pub fn append(&mut self, mut other: Atoms<P>) {
        assert_eq!(other.layout.block_len(), self.layout.block_len());
        self.points.append(&mut other.points);
        self.weights.append(&mut other.weights);
        self.moments.append(&mut other.moments);
    }

    pub fn points(&self) -> &[P] {
        &self.points
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

    /// Compute integral of source, used for source normalization and k computation
    pub fn compute_integral(&self) -> f64 {
        self.moments
            .chunks_exact(self.layout.block_len()) // Create an iterator of atoms by chunking by block_len
            .zip(self.weights()) // Zip moments and weights iterator of (block, w)
            .map(|(block, &w)| {
                w * self
                    .layout
                    .moment_l_indices(0)
                    .map(|i| block[i])
                    .sum::<f64>() // Sum of moment 0 over energies using iterator mapping
            })
            .sum() // Sum over atoms
    }

    /// When the integral is computed, this method is used to normalize to unity
    pub fn scale_moments(&mut self, factor: f64) {
        for m in self.moments.iter_mut() {
            *m *= factor;
        }
    }
}
