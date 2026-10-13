use crate::traits::{Emission, Gather, Medium};
use std::ops::Range;

pub(crate) const RTOL: f64 = 1e-12;
pub(crate) const ATOL: f64 = 1e-14;

/// Minimal 1D medium on [0, width) that accepts every sample
pub(crate) struct TestSlab {
    pub width: f64,
    pub n_groups: usize,
    /// Test acceptance range for u\in[0,1) for rejection sampling
    accept: Range<f64>,
}

impl TestSlab {
    pub(crate) fn new(width: f64, n_groups: usize, accept: Range<f64>) -> Self {
        Self {
            width,
            n_groups,
            accept,
        }
    }
}

impl Medium for TestSlab {
    type Point = f64;
    const SAMPLE_DIM: usize = 1;
    fn n_groups(&self) -> usize {
        self.n_groups
    }
    fn optical_depth(&self, _: &f64, _: &f64) -> f64 {
        0.0
    }
    fn volume(&self) -> f64 {
        self.width * (self.accept.end - self.accept.start)
    }
    fn map_uniform(&self, u: &[f64]) -> Option<f64> {
        if self.accept.contains(&u[0]) {
            Some(u[0] * self.width)
        } else {
            None
        }
    }
}

/// Gather that just returns a constant block for ach atom
pub(crate) struct ConstGather {
    block: Vec<f64>,
}

impl ConstGather {
    pub(crate) fn new(block: Vec<f64>) -> Self {
        Self { block }
    }
}

impl<M: Medium> Gather<M> for ConstGather {
    fn gather(
        &self,
        _medium: &M,
        _src_atoms: &crate::atoms::Atoms<<M as Medium>::Point>,
        _target_point: <M as Medium>::Point,
    ) -> Vec<f64> {
        self.block.clone()
    }
}

/// Emission that returns the flux moments it was supplied
pub(crate) struct IdentityEmission;

impl IdentityEmission {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<M: Medium> Emission<M> for IdentityEmission {
    fn compute_emission(&self, _medium: &M, target_moments: &[f64]) -> Vec<f64> {
        target_moments.to_vec()
    }
}

// Closeness assertions used throughout the tests.
//
// Two values are "close" when |a - b| <= atol + rtol * |b| (the rule numpy's
// `isclose` uses): `rtol` covers values of ordinary size, `atol` covers values
// near zero, where any relative test would fail on round-off alone.
// `#[track_caller]` makes a failure report the line in the *test* that called
// the helper, not a line inside the helper.

/// Assert `a` is close to the expected value `b` with the default tolerances
#[track_caller]
pub(crate) fn assert_close(a: f64, b: f64) {
    assert_close_tol(a, b, RTOL, ATOL);
}

/// For the few cases that need a looser tolerance like statistical checks
#[track_caller]
pub(crate) fn assert_close_tol(a: f64, b: f64, rtol: f64, atol: f64) {
    let diff = (a - b).abs();
    let tol = atol + rtol * b.abs();
    assert!(
        diff <= tol,
        "not close: actual {a:e}, expected {b:e} (|diff| = {diff:e} > tol = {tol:e})"
    );
}

/// Vector version: same length, and every entry close to its expected value
#[track_caller]
pub(crate) fn assert_all_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - b).abs();
        let tol = ATOL + RTOL * b.abs();
        assert!(
            diff <= tol,
            "not close at index {i}: actual {a:e}, expected {b:e} (|diff| = {diff:e} > tol = {tol:e})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_values_pass() {
        assert_close(1.0, 1.0 + 1e-13);
        assert_close(0.0, 1e-15); // near zero: absolute tolerance applies
        assert_all_close(&[1.0, 0.0], &[1.0 + 1e-13, 1e-15]);
    }

    #[test]
    #[should_panic(expected = "not close")]
    fn far_values_fail() {
        assert_close(1.0, 1.0 + 1e-6);
    }

    #[test]
    #[should_panic(expected = "not close at index 1")]
    fn slice_reports_failing_index() {
        assert_all_close(&[1.0, 2.0], &[1.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn slice_length_mismatch_fails() {
        assert_all_close(&[1.0], &[1.0, 2.0]);
    }
}
