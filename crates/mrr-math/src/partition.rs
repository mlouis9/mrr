//! Labelled partitions of an interval, built incrementally.
//!
//! A [`Partition<L>`] splits `[start, end]` into consecutive intervals, each
//! carrying a label (a material, an overlay cell, a tally bin, ...). It is the
//! per-path counterpart of [`Bins`]: `Bins` is a fixed, validated grid built
//! once; a `Partition` is rebuilt for every ray, into a reused buffer, by
//! pushing one interval at a time.
//!
//! # Invariants
//!
//! Every method preserves these, so consumers never re-check them:
//!
//! 1. `edges` is strictly increasing and finite, and starts at `start`.
//! 2. Every interval is at least `min_len` long (no slivers).
//! 3. `labels.len() == edges.len() - 1`. An empty partition (`edges == [start]`,
//!    no labels) is valid, e.g. for a zero-length path.
//!
//! # Sliver rule
//!
//! [`push`](Partition::push) is the single place where near-coincident edges
//! are resolved. An end closer than `min_len` to the current end is
//! *absorbed*: the last interval is stretched to it (or, if the partition is
//! still empty, nothing is added). Labels are never re-derived from
//! positions. [`merge`](Partition::merge) relies on this rule, so two layers
//! whose edges differ by rounding error produce no slivers.
//!
//! Adjacent intervals with equal labels are **not** coalesced: an edge between
//! two pieces of the same material may still be a real face (e.g. a lattice
//! cell boundary), which matters wherever edges, not only labels, are used.
//!
//! # Interval convention
//!
//! Same as `Bins`: intervals are half-open, `[e_k, e_{k+1})`. `locate(end)` is
//! `None`; `locate_closed(end)` gives the last interval.

use std::fmt;

use crate::Bins;

/// Why a `Partition` could not be built from existing data
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionError {
    /// One label per bin is required
    LabelCount { expected: usize, got: usize },
    /// Bin `index` is shorter than the partition's `min_len`
    TooShort {
        index: usize,
        width: f64,
        min_len: f64,
    },
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PartitionError::LabelCount { expected, got } => {
                write!(f, "partition needs {expected} labels, got {got}")
            }
            PartitionError::TooShort {
                index,
                width,
                min_len,
            } => write!(
                f,
                "interval {index} has width {width:e}, below the minimum {min_len:e}"
            ),
        }
    }
}

impl std::error::Error for PartitionError {}

/// One labelled interval `[lo, hi)` of a partition
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span<L> {
    pub lo: f64,
    pub hi: f64,
    pub label: L,
}

impl<L> Span<L> {
    pub fn length(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        0.5 * (self.lo + self.hi)
    }
}

/// A partition of `[start, end]` into labelled intervals. See the module docs
/// for the invariants and the sliver rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Partition<L> {
    edges: Vec<f64>,
    labels: Vec<L>,
    min_len: f64,
}

impl<L: Copy> Partition<L> {
    /// Empty partition starting at `start`. Intervals shorter than `min_len`
    /// are never created (`min_len = 0` only forbids zero-length intervals).
    pub fn new(start: f64, min_len: f64) -> Self {
        assert!(
            start.is_finite(),
            "partition start must be finite, got {start}"
        );
        assert!(
            min_len.is_finite() && min_len >= 0.0,
            "min_len must be finite and non-negative, got {min_len}"
        );
        Self {
            edges: vec![start],
            labels: Vec::new(),
            min_len,
        }
    }

    /// Empty the partition and restart it at `start`, keeping the allocation
    /// and `min_len`. Use this to reuse one buffer across many rays.
    pub fn reset(&mut self, start: f64) {
        assert!(
            start.is_finite(),
            "partition start must be finite, got {start}"
        );
        self.edges.clear();
        self.edges.push(start);
        self.labels.clear();
    }

    /// Extend the partition to `end` with `label`.
    ///
    /// - `end` at least `min_len` beyond the current end: a new interval.
    /// - `end` within `min_len` of the current end: absorbed (see module docs);
    ///   `label` is dropped.
    ///
    /// Panics if `end` is not finite or lies more than `min_len` *before* the
    /// current end: that can only be a bug in the code producing the edges.
    pub fn push(&mut self, end: f64, label: L) {
        assert!(end.is_finite(), "partition edge must be finite, got {end}");
        let current = self.end();
        assert!(
            end >= current - self.min_len,
            "partition edge {end} goes backwards (current end {current})"
        );
        if end - current >= self.min_len && end > current {
            self.edges.push(end);
            self.labels.push(label);
        } else if end > current && !self.labels.is_empty() {
            // Absorb: stretch the last interval. Keeps `end()` exact.
            *self.edges.last_mut().expect("edges is never empty") = end;
        }
        // Otherwise (end <= current, or a sliver at the very start): nothing.
    }

    /// Partition with the edges of `bins` and one label per bin.
    pub fn from_bins(bins: &Bins, labels: Vec<L>, min_len: f64) -> Result<Self, PartitionError> {
        if labels.len() != bins.n_bins() {
            return Err(PartitionError::LabelCount {
                expected: bins.n_bins(),
                got: labels.len(),
            });
        }
        if let Some(index) = (0..bins.n_bins()).find(|&i| bins.bin_width(i) < min_len) {
            return Err(PartitionError::TooShort {
                index,
                width: bins.bin_width(index),
                min_len,
            });
        }
        let mut p = Self::new(bins.lower(), min_len);
        p.edges.extend_from_slice(&bins.edges()[1..]);
        p.labels = labels;
        Ok(p)
    }

    pub fn min_len(&self) -> f64 {
        self.min_len
    }

    pub fn start(&self) -> f64 {
        self.edges[0]
    }

    /// End of the last interval (`start()` if empty)
    pub fn end(&self) -> f64 {
        *self.edges.last().expect("edges is never empty")
    }

    /// Total length covered, `end() - start()`
    pub fn length(&self) -> f64 {
        self.end() - self.start()
    }

    /// Number of intervals
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// All edges, `len() + 1` of them
    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    pub fn labels(&self) -> &[L] {
        &self.labels
    }

    /// Interval `k`
    pub fn span(&self, k: usize) -> Span<L> {
        Span {
            lo: self.edges[k],
            hi: self.edges[k + 1],
            label: self.labels[k],
        }
    }

    /// Intervals in increasing order
    pub fn iter(&self) -> impl Iterator<Item = Span<L>> + '_ {
        self.edges
            .windows(2)
            .zip(&self.labels)
            .map(|(e, &label)| Span {
                lo: e[0],
                hi: e[1],
                label,
            })
    }

    /// Index of the interval containing `s` (half-open), or `None` outside
    /// `[start, end)`, for NaN, or if empty.
    pub fn index_of(&self, s: f64) -> Option<usize> {
        if !(s >= self.start() && s < self.end()) {
            return None;
        }
        Some(self.edges.partition_point(|&e| e <= s) - 1)
    }

    /// Label at `s` (half-open intervals)
    pub fn locate(&self, s: f64) -> Option<L> {
        self.index_of(s).map(|k| self.labels[k])
    }

    /// Like `locate`, but `end()` belongs to the last interval
    pub fn locate_closed(&self, s: f64) -> Option<L> {
        if s == self.end() && !self.is_empty() {
            self.labels.last().copied()
        } else {
            self.locate(s)
        }
    }

    /// Common refinement of `self` and `other`: every edge of either, with
    /// labels paired. `out` is reset and keeps `self.min_len()`.
    ///
    /// Both partitions must cover the same range up to `min_len` (they come
    /// from different layers segmenting the same path). Edges of the two
    /// inputs closer than `min_len` produce no sliver: the later one is
    /// absorbed by `push`.
    pub fn merge<M: Copy>(&self, other: &Partition<M>, out: &mut Partition<(L, M)>) {
        let tol = self.min_len.max(other.min_len);
        assert!(
            (self.start() - other.start()).abs() <= tol && (self.end() - other.end()).abs() <= tol,
            "merging partitions of different ranges: [{}, {}] vs [{}, {}]",
            self.start(),
            self.end(),
            other.start(),
            other.end()
        );
        out.min_len = self.min_len;
        out.reset(self.start());
        if self.is_empty() || other.is_empty() {
            return;
        }

        // Walk both edge lists; (i, j) index the current interval of each.
        let (a, b) = (&self.edges, &other.edges);
        let (mut i, mut j) = (0, 0);
        while i < self.len() && j < other.len() {
            let (ea, eb) = (a[i + 1], b[j + 1]);
            let next = ea.min(eb);
            out.push(next, (self.labels[i], other.labels[j]));
            // Advance every input whose current interval ends at (or within
            // tol of) `next`, so a near-coincident edge is consumed once.
            if ea <= next + tol {
                i += 1;
            }
            if eb <= next + tol {
                j += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_close;

    /// The partitions' `min_len` (a property of the data, not a test tolerance)
    const TOL: f64 = 1e-10;

    fn p(start: f64, pieces: &[(f64, u32)]) -> Partition<u32> {
        let mut part = Partition::new(start, TOL);
        for &(end, label) in pieces {
            part.push(end, label);
        }
        part
    }

    #[test]
    fn new_is_empty() {
        let part: Partition<u32> = Partition::new(2.0, TOL);
        assert!(part.is_empty());
        assert_eq!(part.len(), 0);
        assert_eq!(part.start(), 2.0);
        assert_eq!(part.end(), 2.0);
        assert_eq!(part.length(), 0.0);
        assert_eq!(part.iter().count(), 0);
        assert_eq!(part.locate(2.0), None);
        assert_eq!(part.locate_closed(2.0), None);
    }

    #[test]
    fn push_builds_intervals() {
        let part = p(0.0, &[(1.0, 7), (2.5, 8)]);
        assert_eq!(part.len(), 2);
        assert_eq!(part.edges(), &[0.0, 1.0, 2.5]);
        assert_eq!(part.labels(), &[7, 8]);
        assert_eq!(
            part.span(1),
            Span {
                lo: 1.0,
                hi: 2.5,
                label: 8
            }
        );
        let lengths: Vec<f64> = part.iter().map(|s| s.length()).collect();
        assert_eq!(lengths, vec![1.0, 1.5]);
    }

    #[test]
    fn push_absorbs_slivers() {
        // Just past the current end: stretch the last interval, drop label
        let part = p(0.0, &[(1.0, 1), (1.0 + 1e-12, 2), (2.0, 3)]);
        assert_eq!(part.labels(), &[1, 3]);
        assert_eq!(part.edges(), &[0.0, 1.0 + 1e-12, 2.0]);
        // Equal or slightly before the current end: ignored
        let part = p(0.0, &[(1.0, 1), (1.0, 2), (1.0 - 1e-12, 3)]);
        assert_eq!(part.labels(), &[1]);
        assert_eq!(part.end(), 1.0);
    }

    #[test]
    fn sliver_at_start_is_dropped() {
        let part = p(0.0, &[(1e-12, 1), (1.0, 2)]);
        assert_eq!(part.labels(), &[2]);
        assert_eq!(part.edges(), &[0.0, 1.0]);
    }

    #[test]
    fn zero_min_len_only_forbids_empty_intervals() {
        let mut part = Partition::new(0.0, 0.0);
        part.push(1e-300, 1);
        part.push(1e-300, 2);
        assert_eq!(part.labels(), &[1]);
    }

    #[test]
    #[should_panic(expected = "goes backwards")]
    fn push_backwards_panics() {
        p(0.0, &[(1.0, 1), (0.5, 2)]);
    }

    #[test]
    #[should_panic(expected = "must be finite")]
    fn push_nan_panics() {
        p(0.0, &[(f64::NAN, 1)]);
    }

    #[test]
    fn reset_keeps_min_len_and_empties() {
        let mut part = p(0.0, &[(1.0, 1), (2.0, 2)]);
        part.reset(5.0);
        assert!(part.is_empty());
        assert_eq!(part.start(), 5.0);
        assert_eq!(part.min_len(), TOL);
        part.push(6.0, 9);
        assert_eq!(part.labels(), &[9]);
    }

    #[test]
    fn locate_is_half_open() {
        let part = p(0.0, &[(1.0, 1), (2.0, 2)]);
        assert_eq!(part.locate(0.0), Some(1));
        assert_eq!(part.locate(0.5), Some(1));
        assert_eq!(part.locate(1.0), Some(2)); // interior edge -> interval above
        assert_eq!(part.locate(2.0), None); // end is outside
        assert_eq!(part.locate_closed(2.0), Some(2));
        assert_eq!(part.locate(-0.1), None);
        assert_eq!(part.locate(f64::NAN), None);
        assert_eq!(part.index_of(1.5), Some(1));
    }

    #[test]
    fn from_bins_checks_labels_and_widths() {
        let bins = Bins::new(vec![0.0, 1.0, 3.0]).unwrap();
        let part = Partition::from_bins(&bins, vec![4u32, 5], TOL).unwrap();
        assert_eq!(part, p(0.0, &[(1.0, 4), (3.0, 5)]));
        assert_eq!(
            Partition::from_bins(&bins, vec![4u32], TOL),
            Err(PartitionError::LabelCount {
                expected: 2,
                got: 1
            })
        );
        let thin = Bins::new(vec![0.0, 1e-12, 1.0]).unwrap();
        assert!(matches!(
            Partition::from_bins(&thin, vec![1u32, 2], TOL),
            Err(PartitionError::TooShort { index: 0, .. })
        ));
    }

    #[test]
    fn merge_pairs_labels_on_common_refinement() {
        let a = p(0.0, &[(1.0, 1), (3.0, 2)]);
        let b = p(0.0, &[(2.0, 10), (3.0, 20)]);
        let mut out = Partition::new(0.0, TOL);
        a.merge(&b, &mut out);
        assert_eq!(out.edges(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(out.labels(), &[(1, 10), (2, 10), (2, 20)]);
    }

    #[test]
    fn merge_with_coincident_and_near_coincident_edges() {
        let a = p(0.0, &[(1.0, 1), (2.0, 2), (3.0, 3)]);
        let b = p(0.0, &[(1.0, 10), (2.0 + 1e-13, 20), (3.0, 30)]);
        let mut out = Partition::new(0.0, TOL);
        a.merge(&b, &mut out);
        // No sliver between 2.0 and 2.0 + 1e-13
        assert_eq!(out.len(), 3);
        assert_eq!(out.labels(), &[(1, 10), (2, 20), (3, 30)]);
        assert_eq!(out.end(), 3.0);
    }

    #[test]
    fn merge_with_single_interval_is_identity_up_to_labels() {
        let a = p(0.0, &[(0.4, 1), (1.7, 2), (2.0, 3)]);
        let whole = p(0.0, &[(2.0, 0u32)]);
        let mut out = Partition::new(0.0, TOL);
        a.merge(&whole, &mut out);
        assert_eq!(out.edges(), a.edges());
        let left: Vec<u32> = out.labels().iter().map(|&(l, _)| l).collect();
        assert_eq!(left, a.labels());
    }

    #[test]
    fn merge_of_empty_is_empty() {
        let a: Partition<u32> = Partition::new(1.0, TOL);
        let b: Partition<u32> = Partition::new(1.0, TOL);
        // A non-empty buffer is reset by merge
        let mut out = p(0.0, &[(5.0, 1)]);
        let mut out_pairs = Partition::new(0.0, TOL);
        a.merge(&b, &mut out_pairs);
        assert!(out_pairs.is_empty());
        assert_eq!(out_pairs.start(), 1.0);
        out.reset(0.0);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "different ranges")]
    fn merge_of_different_ranges_panics() {
        let a = p(0.0, &[(1.0, 1)]);
        let b = p(0.0, &[(2.0, 1)]);
        let mut out = Partition::new(0.0, TOL);
        a.merge(&b, &mut out);
    }

    /// Small deterministic generator, so the property test needs no extra
    /// dependency and cannot fail at random.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Random partition of [0, len] with `n` cuts, some of them duplicated
    /// from `shared` (to create exactly and nearly coincident edges)
    fn random_partition(
        rng: &mut Lcg,
        len: f64,
        n: usize,
        shared: &[f64],
        label0: u32,
    ) -> Partition<u32> {
        let mut cuts: Vec<f64> = (0..n).map(|_| rng.next() * len).collect();
        for &s in shared {
            let r = rng.next();
            if r < 0.3 {
                cuts.push(s); // exactly coincident
            } else if r < 0.5 {
                cuts.push(s + (rng.next() - 0.5) * 1e-12); // nearly coincident
            }
        }
        cuts.retain(|&c| c > 0.0 && c < len);
        cuts.sort_by(f64::total_cmp);
        let mut part = Partition::new(0.0, TOL);
        for (k, &c) in cuts.iter().enumerate() {
            part.push(c, label0 + k as u32);
        }
        part.push(len, label0 + cuts.len() as u32);
        part
    }

    #[test]
    fn merge_property_random_partitions() {
        let mut rng = Lcg(12345);
        let mut out = Partition::new(0.0, TOL);
        for _ in 0..500 {
            let len = 0.5 + 3.0 * rng.next();
            let (na, nb) = ((rng.next() * 8.0) as usize, (rng.next() * 8.0) as usize);
            let a = random_partition(&mut rng, len, na, &[], 0);
            let shared = a.edges().to_vec();
            let b = random_partition(&mut rng, len, nb, &shared, 1000);
            a.merge(&b, &mut out);

            // Invariants
            assert_eq!(out.start(), 0.0);
            assert_eq!(out.end(), len);
            assert_eq!(out.labels().len() + 1, out.edges().len());
            for s in out.iter() {
                assert!(s.length() >= TOL, "sliver {s:?}");
                // Labels agree with each input at the midpoint
                let mid = s.midpoint();
                assert_eq!(Some(s.label.0), a.locate(mid), "a at {mid}");
                assert_eq!(Some(s.label.1), b.locate(mid), "b at {mid}");
            }
            // Every input edge appears in the output, up to the tolerance
            for &e in a.edges().iter().chain(b.edges()) {
                assert!(
                    out.edges().iter().any(|&o| (o - e).abs() <= TOL),
                    "edge {e} lost"
                );
            }
            // Total length preserved
            let total: f64 = out.iter().map(|s| s.length()).sum();
            assert_close(total, len);
        }
    }
}
