//! Defines empirical emission source measure

use crate::moments::MomentLayout;

// Struct of arrays that contain properties of atoms templated by point type
pub struct Atoms<P> {
    points: Vec<P>,
    weights: Vec<f64>,
    moments: Vec<f64>,
    layout: MomentLayout,
}

impl<P: Copy + Clone> Atoms<P> {
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
            .map(|(block, &w)| w * self.layout.block_integral(block))
            .sum() // Sum over atoms
    }

    /// When the integral is computed, this method is used to normalize to unity
    pub fn scale_moments(&mut self, factor: f64) {
        for m in self.moments.iter_mut() {
            *m *= factor;
        }
    }

    pub fn from_targets(
        points: Vec<P>,
        weights: Vec<f64>,
        layout: MomentLayout,
        block: &[f64],
    ) -> Atoms<P> {
        assert_eq!(points.len(), weights.len());
        assert_eq!(layout.block_len(), block.len());
        let mut atoms = Atoms::new(points.len(), layout);
        for atom_idx in 0..points.len() {
            atoms.push(points[atom_idx], weights[atom_idx], block);
        }
        atoms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    // Order 1 (2 moments) with 2 groups: block = [g0: l0 l1 | g1: l0 l1]
    fn layout() -> MomentLayout {
        MomentLayout::from_legendre_order(1, 2)
    }

    fn two_atoms() -> Atoms<f64> {
        let mut atoms = Atoms::new(2, layout());
        atoms.push(0.25, 0.5, &[1.0, 2.0, 3.0, 4.0]);
        atoms.push(0.75, 1.5, &[5.0, 6.0, 7.0, 8.0]);
        atoms
    }

    #[test]
    fn push_then_read_back() {
        let atoms = two_atoms();
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms.points(), &[0.25, 0.75]);
        assert_eq!(atoms.weights(), &[0.5, 1.5]);
        assert_eq!(atoms.atom_moments(1), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn update_changes_only_that_atom() {
        let mut atoms = two_atoms();
        atoms.update_atom_moments(0, &[0.0; 4]);
        assert_eq!(atoms.atom_moments(0), &[0.0; 4]);
        assert_eq!(atoms.atom_moments(1), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn append_keeps_order() {
        let mut atoms = two_atoms();
        atoms.append(two_atoms());
        assert_eq!(atoms.points(), &[0.25, 0.75, 0.25, 0.75]);
        assert_eq!(atoms.atom_moments(2), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn integral_uses_only_l0_moments() {
        // Atom 0: w = 0.5, q0 = 1 + 3; atom 1: w = 1.5, q0 = 5 + 7.
        // The l = 1 entries (2, 4, 6, 8) must not contribute.
        assert_close(two_atoms().compute_integral(), 0.5 * 4.0 + 1.5 * 12.0);
    }

    #[test]
    fn scaling_moments_scales_the_integral() {
        let mut atoms = two_atoms();
        let before = atoms.compute_integral();
        atoms.scale_moments(0.5);
        assert_close(atoms.compute_integral(), 0.5 * before);
    }
}
