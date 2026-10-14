//! Describes how Legendre angular moments are implemented
use std::ops::Range;

/// Shape of one point's moment block: `n_moments` angular moments for each
/// of `n_groups` energy bins, stored group-major (all moments of bin 0,
/// then all of bin 1, ...).
///
/// `n_groups` is the number of energy bins carried *per atom*: G for
/// multigroup (energy stored as a block), 1 for continuous energy (energy
/// stored in the atom's point instead). Nothing here assumes which.
///
/// Moments are indexed by a flat index `k`. In 1D, k = ℓ. In 3D the mapping
/// from (ℓ, m) to k is defined in `mrr_math`, not here
#[derive(Clone, Copy)]
pub struct MomentLayout {
    n_moments: usize,
    n_groups: usize,
}

impl MomentLayout {
    pub fn from_legendre_order(max_order: usize, n_groups: usize) -> Self {
        Self {
            n_moments: max_order + 1,
            n_groups,
        }
    }

    pub fn new_moments_array(&self, n_atoms: usize) -> Vec<f64> {
        Vec::with_capacity(n_atoms * self.n_moments * self.n_groups)
    }

    pub fn block_len(&self) -> usize {
        self.n_moments * self.n_groups
    }

    /// Position of moment `l` of group `g` inside one atom's block, group major
    pub fn index(&self, l: usize, g: usize) -> usize {
        l + self.n_moments * g
    }

    /// Positions of moment `l` of all groups inside one atom's block
    pub fn moment_l_indices(&self, l: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.n_groups).map(move |g| self.index(l, g))
    }

    fn isotropic_angular_block(&self) -> Vec<f64> {
        let mut moments = vec![0.0; self.n_moments];
        moments[0] = 1.0;
        moments
    }

    /// Create an isotropic angular block in every energy group for a single atom
    pub fn isotropic_white_block(&self) -> Vec<f64> {
        self.isotropic_angular_block()
            .repeat(self.n_groups)
            .into_iter()
            .map(|m| m / self.n_groups as f64) // Divide 1.0 by N_groups so moment block sums to 1
            .collect()
    }

    /// Integral of one atom's block over angle and energy: the sum of the
    /// l = 0 moments over all energy bins
    pub fn block_integral(&self, block: &[f64]) -> f64 {
        self.moment_l_indices(0).map(|i| block[i]).sum()
    }

    /// Create a zero block for a single atom
    pub fn zero_block(&self) -> Vec<f64> {
        vec![0.0; self.block_len()]
    }

    /// Return the range of the flat moments array corresponding to atom `idx`.
    pub fn atom_range(&self, idx: usize) -> Range<usize> {
        let start = idx * self.block_len();
        start..start + self.block_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    // Order 2 (3 moments) with 2 groups: block = [g0: l0 l1 l2 | g1: l0 l1 l2]
    fn layout() -> MomentLayout {
        MomentLayout::from_legendre_order(2, 2)
    }

    #[test]
    fn block_len_is_moments_times_groups() {
        assert_eq!(layout().block_len(), 6);
    }

    #[test]
    fn index_is_group_major() {
        let l = layout();
        assert_eq!(l.index(0, 0), 0);
        assert_eq!(l.index(2, 0), 2);
        assert_eq!(l.index(0, 1), 3);
        assert_eq!(l.index(1, 1), 4);
    }

    #[test]
    fn moment_l_indices_visit_every_group() {
        let l = layout();
        assert_eq!(l.moment_l_indices(0).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(l.moment_l_indices(1).collect::<Vec<_>>(), vec![1, 4]);
    }

    #[test]
    fn atom_range_selects_one_block() {
        let l = layout();
        assert_eq!(l.atom_range(0), 0..6);
        assert_eq!(l.atom_range(1), 6..12);
    }

    #[test]
    fn isotropic_white_block_has_q0_one_over_g() {
        let l = MomentLayout::from_legendre_order(2, 3);
        let third = 1.0 / 3.0;
        assert_all_close(
            &l.isotropic_white_block(),
            &[third, 0.0, 0.0, third, 0.0, 0.0, third, 0.0, 0.0],
        );
    }

    #[test]
    fn zero_block_is_all_zeros() {
        assert_eq!(layout().zero_block(), vec![0.0; 6]);
    }
}
