//! Describes how Legendre angular moments are implemented

/// Shape of one point's moment block: `n_moments` angular moments for each
/// of `n_groups` energy groups, stored group-major (all moments of group 0,
/// then all of group 1, ...).
///
/// Moments are indexed by a flat index `k`. In 1D, k = ℓ. In 3D the mapping
/// from (ℓ, m) to k is defined in `mrr_math`, not here
pub struct MomentLayout {
    n_moments: usize,
    n_groups: usize,
}
