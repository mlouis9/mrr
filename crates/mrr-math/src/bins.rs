//! Validated data structures useful for mathematical operations

use std::fmt;

/// Index of the first element that is not strictly larger than its predecessor,
/// or `None` if `v` is strictly increasing. Comparisons involving NaN count as
/// "not increasing".
#[allow(clippy::neg_cmp_op_on_partial_ord)] // intentional: NaN must count as "not increasing"
pub fn first_non_increasing<T: PartialOrd>(v: &[T]) -> Option<usize> {
    v.windows(2).position(|w| !(w[0] < w[1])).map(|i| i + 1)
}

/// `true` if every element is strictly larger than the one before it
pub fn is_strictly_increasing<T: PartialOrd>(v: &[T]) -> bool {
    first_non_increasing(v).is_none()
}

/// Why a set of edges could not form `Bins`
#[derive(Debug, Clone, PartialEq)]
pub enum BinsError {
    /// At least two edges are needed to make one bin
    TooFewEdges { found: usize },
    /// Edge `index` is NaN or infinite
    NonFinite { index: usize },
    /// Edge `index` is not strictly larger than edge `index - 1`
    NotIncreasing { index: usize },
}

impl fmt::Display for BinsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinsError::TooFewEdges { found } => {
                write!(f, "need at least 2 bin edges, found {found}")
            }
            BinsError::NonFinite { index } => write!(f, "bin edge {index} is not finite"),
            BinsError::NotIncreasing { index } => write!(
                f,
                "bin edge {index} is not larger than bin edge {}",
                index - 1
            ),
        }
    }
}

impl std::error::Error for BinsError {}

/// Partition of [edges[0], edges[n]) into n half-open bins [e_i, e_{i+1}).
///
/// Guaranteed by construction: at least 2 edges, all finite, strictly
/// increasing. The field is private so the guarantee cannot be broken.
#[derive(Debug, Clone, PartialEq)]
pub struct Bins {
    edges: Vec<f64>,
}

impl Bins {
    pub fn new(edges: Vec<f64>) -> Result<Self, BinsError> {
        // Order matters: count first, then finiteness (so NaN/inf are
        // reported as such rather than as an ordering problem), then order.
        if edges.len() < 2 {
            return Err(BinsError::TooFewEdges { found: edges.len() });
        }
        if let Some(index) = edges.iter().position(|e| !e.is_finite()) {
            return Err(BinsError::NonFinite { index });
        }
        if let Some(index) = first_non_increasing(&edges) {
            return Err(BinsError::NotIncreasing { index });
        }
        Ok(Self { edges })
    }

    /// Number of bins (one fewer than the number of edges). Never zero.
    pub fn n_bins(&self) -> usize {
        self.edges.len() - 1
    }

    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    /// Lower end of the whole partition
    pub fn lower(&self) -> f64 {
        self.edges[0]
    }

    /// Upper end of the whole partition
    pub fn upper(&self) -> f64 {
        self.edges[self.edges.len() - 1]
    }

    /// Total width covered by all bins
    pub fn span(&self) -> f64 {
        self.upper() - self.lower()
    }

    /// Edges (lower, upper) of bin `i`
    pub fn bin(&self, i: usize) -> (f64, f64) {
        (self.edges[i], self.edges[i + 1])
    }

    /// Width of bin `i`
    pub fn bin_width(&self, i: usize) -> f64 {
        self.edges[i + 1] - self.edges[i]
    }

    /// Index of the bin containing `x`, using half-open bins [e_i, e_{i+1}).
    /// `None` if `x` is below `lower()`, at or above `upper()`, or NaN.
    pub fn locate(&self, x: f64) -> Option<usize> {
        if !(x >= self.lower() && x < self.upper()) {
            return None; // also catches NaN, since NaN comparisons are false
        }
        // Number of edges <= x is at least 1 here, so subtracting 1 is safe
        Some(self.edges.partition_point(|&e| e <= x) - 1)
    }

    /// Like [`locate`](Self::locate), but on the *closed* range
    /// [lower(), upper()]: the top edge belongs to the last bin.
    ///
    /// Use this when the whole domain is closed (a slab including its right
    /// face, an energy grid including its top edge). Interior edges still
    /// belong to the bin above, exactly as in `locate`.
    pub fn locate_closed(&self, x: f64) -> Option<usize> {
        if x == self.upper() {
            Some(self.n_bins() - 1)
        } else {
            self.locate(x)
        }
    }

    /// Clip `x` into [lower(), upper()]. NaN stays NaN.
    pub fn clamp(&self, x: f64) -> f64 {
        x.clamp(self.lower(), self.upper())
    }

    /// The bins overlapped by the interval between `a` and `b`, as
    /// `(bin index, overlap length)` in ascending bin order.
    ///
    /// - Order of `a` and `b` does not matter.
    /// - The interval is clipped to [lower(), upper()] first.
    /// - Only overlaps of positive length are yielded, so an interval that
    ///   sits on an edge, is empty (`a == b`), lies outside, or contains NaN
    ///   yields nothing.
    /// - The lengths add up to the clipped interval length (up to rounding).
    pub fn overlaps(&self, a: f64, b: f64) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let (lo, hi) = (self.clamp(lo), self.clamp(hi));
        // Bins that can overlap: from the one containing `lo` up to the one
        // containing `hi`. If anything is NaN or the interval is empty the
        // range is empty.
        let range = if lo < hi {
            let first = self.locate_closed(lo).unwrap_or(0);
            let last = self.locate_closed(hi).unwrap_or(0);
            first..last + 1
        } else {
            0..0
        };
        range.filter_map(move |i| {
            let (e0, e1) = self.bin(i);
            let len = hi.min(e1) - lo.max(e0);
            (len > 0.0).then_some((i, len))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    fn bins() -> Bins {
        // Three uneven bins: [0, 1), [1, 1.5), [1.5, 4)
        Bins::new(vec![0.0, 1.0, 1.5, 4.0]).unwrap()
    }

    #[test]
    fn helper_reports_first_offending_index() {
        assert_eq!(first_non_increasing(&[0.0, 1.0, 2.0]), None);
        assert_eq!(first_non_increasing(&[0.0, 1.0, 1.0, 2.0]), Some(2));
        assert_eq!(first_non_increasing(&[0.0, 2.0, 1.0]), Some(2));
        assert_eq!(first_non_increasing::<f64>(&[]), None);
        assert!(is_strictly_increasing(&[1, 2, 3]));
        assert!(!is_strictly_increasing(&[1, 1]));
    }

    #[test]
    fn rejects_too_few_edges() {
        assert_eq!(Bins::new(vec![]), Err(BinsError::TooFewEdges { found: 0 }));
        assert_eq!(
            Bins::new(vec![1.0]),
            Err(BinsError::TooFewEdges { found: 1 })
        );
    }

    #[test]
    fn rejects_non_finite_edges() {
        assert_eq!(
            Bins::new(vec![0.0, f64::NAN, 1.0]),
            Err(BinsError::NonFinite { index: 1 })
        );
        assert_eq!(
            Bins::new(vec![0.0, f64::INFINITY]),
            Err(BinsError::NonFinite { index: 1 })
        );
        assert_eq!(
            Bins::new(vec![f64::NEG_INFINITY, 0.0]),
            Err(BinsError::NonFinite { index: 0 })
        );
    }

    #[test]
    fn rejects_repeated_or_decreasing_edges() {
        assert_eq!(
            Bins::new(vec![0.0, 1.0, 1.0]),
            Err(BinsError::NotIncreasing { index: 2 })
        );
        assert_eq!(
            Bins::new(vec![2.0, 1.0]),
            Err(BinsError::NotIncreasing { index: 1 })
        );
    }

    #[test]
    fn negative_edges_are_allowed() {
        // Domain rules like "energies are positive" belong to wrapper types
        let b = Bins::new(vec![-1.0, 0.0, 1.0]).unwrap();
        assert_eq!(b.locate(-0.5), Some(0));
    }

    #[test]
    fn sizes_and_edges() {
        let b = bins();
        assert_eq!(b.n_bins(), 3);
        assert_eq!(b.edges(), &[0.0, 1.0, 1.5, 4.0]);
        assert_eq!(b.lower(), 0.0);
        assert_eq!(b.upper(), 4.0);
        assert_eq!(b.span(), 4.0);
        assert_eq!(b.bin(1), (1.0, 1.5));
        assert_eq!(b.bin_width(1), 0.5);
        assert_eq!(b.bin_width(2), 2.5);
    }

    #[test]
    fn locate_inside_bins() {
        let b = bins();
        assert_eq!(b.locate(0.5), Some(0));
        assert_eq!(b.locate(1.2), Some(1));
        assert_eq!(b.locate(3.9), Some(2));
    }

    #[test]
    fn locate_at_edges_is_half_open() {
        let b = bins();
        assert_eq!(b.locate(0.0), Some(0)); // lower edge is inside
        assert_eq!(b.locate(1.0), Some(1)); // interior edge belongs to the bin above
        assert_eq!(b.locate(1.5), Some(2));
        assert_eq!(b.locate(4.0), None); // upper edge is outside
    }

    #[test]
    fn locate_outside_or_nan_is_none() {
        let b = bins();
        assert_eq!(b.locate(-1e-12), None);
        assert_eq!(b.locate(10.0), None);
        assert_eq!(b.locate(f64::NAN), None);
    }

    #[test]
    fn locate_closed_includes_top_edge_only() {
        let b = bins();
        assert_eq!(b.locate_closed(4.0), Some(2)); // top edge -> last bin
        assert_eq!(b.locate_closed(0.0), Some(0));
        assert_eq!(b.locate_closed(1.0), Some(1)); // interior edges unchanged
        assert_eq!(b.locate_closed(4.0 + 1e-12), None);
        assert_eq!(b.locate_closed(f64::NAN), None);
    }

    fn collect(b: &Bins, x0: f64, x1: f64) -> Vec<(usize, f64)> {
        b.overlaps(x0, x1).collect()
    }

    #[test]
    fn overlaps_within_one_bin() {
        assert_eq!(collect(&bins(), 0.25, 0.75), vec![(0, 0.5)]);
    }

    #[test]
    fn overlaps_across_bins_in_ascending_order() {
        let b = bins();
        assert_eq!(collect(&b, 0.5, 2.0), vec![(0, 0.5), (1, 0.5), (2, 0.5)]);
        // Reversed endpoints give the same answer
        assert_eq!(collect(&b, 2.0, 0.5), collect(&b, 0.5, 2.0));
    }

    #[test]
    fn overlaps_whole_domain_including_top_edge() {
        let b = bins();
        assert_eq!(collect(&b, 0.0, 4.0), vec![(0, 1.0), (1, 0.5), (2, 2.5)]);
    }

    #[test]
    fn overlaps_starting_or_ending_on_interior_edges() {
        let b = bins();
        assert_eq!(collect(&b, 1.0, 1.5), vec![(1, 0.5)]); // no zero-length bits
        assert_eq!(collect(&b, 0.5, 1.0), vec![(0, 0.5)]);
    }

    #[test]
    fn overlaps_degenerate_cases_are_empty() {
        let b = bins();
        assert!(collect(&b, 1.0, 1.0).is_empty()); // empty interval on an edge
        assert!(collect(&b, 0.3, 0.3).is_empty());
        assert!(collect(&b, 5.0, 6.0).is_empty()); // outside
        assert!(collect(&b, 4.0, 4.0).is_empty());
        assert!(collect(&b, f64::NAN, 1.0).is_empty());
    }

    #[test]
    fn overlaps_are_clipped_to_the_domain() {
        assert_eq!(collect(&bins(), -1.0, 0.5), vec![(0, 0.5)]);
        assert_eq!(collect(&bins(), 3.0, 9.0), vec![(2, 1.0)]);
    }

    #[test]
    fn overlap_lengths_sum_to_interval_length() {
        let b = bins();
        for i in 0..=20 {
            for j in 0..=20 {
                let (x0, x1) = (0.2 * i as f64, 0.2 * j as f64);
                let total: f64 = b.overlaps(x0, x1).map(|(_, l)| l).sum();
                assert!(is_close(total, (x1 - x0).abs(), Tol::EXACT), "{x0} {x1}");
            }
        }
    }

    #[test]
    fn error_messages_are_readable() {
        assert_eq!(
            BinsError::NotIncreasing { index: 3 }.to_string(),
            "bin edge 3 is not larger than bin edge 2"
        );
    }
}
