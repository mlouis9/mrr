//! Shape of the angular × energy moment block every atom carries
use std::ops::Range;

/// Shape of one point's moment block: `n_moments` angular moments for each
/// of `n_groups` energy bins, stored group-major (all moments of bin 0,
/// then all of bin 1, ...).
///
/// `n_groups` is the number of energy bins carried *per atom*: G for
/// multigroup (energy stored as a block), 1 for continuous energy (energy
/// stored in the atom's point instead). Nothing here assumes which.
///
/// Moments are indexed by a flat index `k`. In 1D, k = ℓ (Legendre moments);
/// in 3D, k = ℓ² + ℓ + m (real spherical harmonics, see
/// `mrr_math::harmonics`). Either way `k = 0` is the isotropic moment, equal
/// to the angular integral, which is all this type relies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MomentLayout {
    n_moments: usize,
    n_groups: usize,
}

impl MomentLayout {
    /// Any number of angular moments per energy bin
    pub fn new(n_moments: usize, n_groups: usize) -> Self {
        assert!(
            n_moments > 0 && n_groups > 0,
            "a moment block needs moments and groups"
        );
        Self {
            n_moments,
            n_groups,
        }
    }

    /// 1D slab: Legendre moments ℓ = 0..=L, so L + 1 per group
    pub fn from_legendre_order(max_order: usize, n_groups: usize) -> Self {
        Self::new(max_order + 1, n_groups)
    }

    /// 3D: real spherical harmonics (ℓ, m) with ℓ = 0..=L, so (L + 1)² per group
    pub fn from_harmonic_order(max_order: usize, n_groups: usize) -> Self {
        Self::new((max_order + 1) * (max_order + 1), n_groups)
    }

    pub fn n_moments(&self) -> usize {
        self.n_moments
    }

    pub fn n_groups(&self) -> usize {
        self.n_groups
    }

    /// The moments of group `g` inside one atom's block
    pub fn group_range(&self, g: usize) -> Range<usize> {
        let start = g * self.n_moments;
        start..start + self.n_moments
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
    fn harmonic_layout_has_square_moment_count() {
        let l = MomentLayout::from_harmonic_order(2, 3);
        assert_eq!((l.n_moments(), l.n_groups(), l.block_len()), (9, 3, 27));
        assert_eq!(l.index(0, 1), 9);
        assert_eq!(l.group_range(2), 18..27);
        // Moment 0 is still the total, so integrals and white blocks carry over
        let white = l.isotropic_white_block();
        assert_close(l.block_integral(&white), 1.0);
        assert_eq!(white[l.index(3, 0)], 0.0);
        assert_eq!(
            MomentLayout::from_legendre_order(2, 3),
            MomentLayout::new(3, 3)
        );
    }

    #[test]
    #[should_panic(expected = "needs moments and groups")]
    fn empty_layout_panics() {
        MomentLayout::new(0, 1);
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
