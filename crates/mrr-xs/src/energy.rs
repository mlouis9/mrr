//! Energy structures

use crate::error::XsError;
use mrr_math::Bins;

/// Multigroup energy structure.
///
/// **Group convention:** group 0 is the *highest*-energy group, as in nuclear
/// data. Internally the edges are stored increasing (as `Bins` requires), so
/// group `g` corresponds to bin `n_groups - 1 - g`; that conversion happens
/// only in `bin_of_group` / `group_of_bin`.
///
/// **Edge convention:** each group is half-open below its upper edge,
/// `[lower, upper)`, except that the overall top edge belongs to group 0 (an
/// energy exactly at the top of the structure is inside the problem).
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyGroups {
    bins: Bins,
}

impl EnergyGroups {
    /// Build from group edges in eV listed in **descending** order (fastest
    /// group first), e.g. `[2e7, 1e5, 0.625, 1e-5]` for three groups.
    pub fn new(mut edges_desc_ev: Vec<f64>) -> Result<Self, XsError> {
        // Bins needs increasing edges; descending input reversed is increasing
        edges_desc_ev.reverse();
        let bins = Bins::new(edges_desc_ev)?;

        // Bins guarantees finite edges, so a plain comparison suffices
        if bins.lower() <= 0.0 {
            return Err(XsError::NonPositiveEnergy {
                value: bins.lower(),
            });
        }
        Ok(Self { bins })
    }

    pub fn n_groups(&self) -> usize {
        self.bins.n_bins()
    }

    /// Group edges in eV, in the input (descending) order
    pub fn edges_desc(&self) -> Vec<f64> {
        self.bins.edges().iter().rev().copied().collect()
    }

    /// (lower, upper) energy of group `g` in eV
    pub fn bounds(&self, g: usize) -> (f64, f64) {
        self.bins.bin(self.bin_of_group(g))
    }

    /// Width of group `g` in eV
    pub fn width(&self, g: usize) -> f64 {
        self.bins.bin_width(self.bin_of_group(g))
    }

    /// Group containing energy `e` (eV), or `None` outside the structure or NaN
    pub fn group_of(&self, e: f64) -> Option<usize> {
        if e == self.bins.upper() {
            return Some(0); // top edge belongs to the highest group
        }
        self.bins.locate(e).map(|i| self.group_of_bin(i))
    }

    // The only two places that know groups and bins run in opposite directions
    fn bin_of_group(&self, g: usize) -> usize {
        assert!(g < self.n_groups(), "group {g} out of range");
        self.n_groups() - 1 - g
    }

    fn group_of_bin(&self, i: usize) -> usize {
        self.n_groups() - 1 - i
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_math::bins::BinsError;

    /// Three uneven groups: fast [1e5, 2e7), epithermal [0.625, 1e5), thermal [1e-5, 0.625)
    fn three_group() -> EnergyGroups {
        EnergyGroups::new(vec![2e7, 1e5, 0.625, 1e-5]).unwrap()
    }

    #[test]
    fn accepts_descending_edges() {
        let groups = three_group();
        assert_eq!(groups.n_groups(), 3);
        assert_eq!(groups.edges_desc(), vec![2e7, 1e5, 0.625, 1e-5]);
    }

    #[test]
    fn single_group() {
        let groups = EnergyGroups::new(vec![2e7, 1e-5]).unwrap();
        assert_eq!(groups.n_groups(), 1);
        assert_eq!(groups.group_of(1.0), Some(0));
    }

    #[test]
    fn group_zero_is_the_fastest() {
        let groups = three_group();
        assert_eq!(groups.bounds(0), (1e5, 2e7));
        assert_eq!(groups.bounds(1), (0.625, 1e5));
        assert_eq!(groups.bounds(2), (1e-5, 0.625));
        assert_eq!(groups.width(1), 1e5 - 0.625);
    }

    #[test]
    fn group_of_inside_each_group() {
        let groups = three_group();
        assert_eq!(groups.group_of(1e6), Some(0));
        assert_eq!(groups.group_of(10.0), Some(1));
        assert_eq!(groups.group_of(0.025), Some(2));
    }

    #[test]
    fn group_of_at_edges() {
        let groups = three_group();
        assert_eq!(groups.group_of(2e7), Some(0)); // top edge is inside
        assert_eq!(groups.group_of(1e5), Some(0)); // interior edge belongs to the higher group
        assert_eq!(groups.group_of(0.625), Some(1));
        assert_eq!(groups.group_of(1e-5), Some(2)); // bottom edge is inside
    }

    #[test]
    fn group_of_outside_or_nan_is_none() {
        let groups = three_group();
        assert_eq!(groups.group_of(3e7), None);
        assert_eq!(groups.group_of(1e-6), None);
        assert_eq!(groups.group_of(f64::NAN), None);
    }

    #[test]
    fn group_of_inverts_bounds() {
        let groups = three_group();
        for g in 0..groups.n_groups() {
            let (lo, hi) = groups.bounds(g);
            assert_eq!(groups.group_of(0.5 * (lo + hi)), Some(g));
        }
    }

    #[test]
    fn rejects_non_positive_lowest_edge() {
        assert!(matches!(
            EnergyGroups::new(vec![2e7, 1.0, 0.0]),
            Err(XsError::NonPositiveEnergy { value }) if value == 0.0
        ));
        assert!(matches!(
            EnergyGroups::new(vec![1.0, -1.0]),
            Err(XsError::NonPositiveEnergy { .. })
        ));
    }

    #[test]
    fn rejects_increasing_input() {
        // Ascending input reversed is descending, so Bins rejects it
        assert!(matches!(
            EnergyGroups::new(vec![1e-5, 0.625, 2e7]),
            Err(XsError::Bins(BinsError::NotIncreasing { .. }))
        ));
    }

    #[test]
    fn passes_through_bins_errors() {
        assert!(matches!(
            EnergyGroups::new(vec![1.0]),
            Err(XsError::Bins(BinsError::TooFewEdges { found: 1 }))
        ));
        assert!(matches!(
            EnergyGroups::new(vec![f64::NAN, 1.0]),
            Err(XsError::Bins(BinsError::NonFinite { .. }))
        ));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn bounds_of_missing_group_panics() {
        three_group().bounds(3);
    }

    #[test]
    fn error_message_is_readable() {
        let err = EnergyGroups::new(vec![1.0, 0.0]).unwrap_err();
        assert_eq!(err.to_string(), "energy must be positive, got 0");
    }
}
