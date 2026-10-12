//! Describes how Legendre angular moments are implemented
use std::ops::Range;

/// Shape of one point's moment block: `n_moments` angular moments for each
/// of `n_groups` energy groups, stored group-major (all moments of group 0,
/// then all of group 1, ...).
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

    fn isotropic_angular_block(&self) -> Vec<f64> {
        let mut moments = vec![0.0; self.n_moments];
        moments[0] = 1.0;
        moments
    }

    /// Create an isotropic angular block in every energy group for a single atom
    pub fn isotropic_white_block(&self) -> Vec<f64> {
        self.isotropic_angular_block().repeat(self.n_groups)
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
