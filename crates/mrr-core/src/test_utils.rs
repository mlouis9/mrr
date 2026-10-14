use crate::traits::{Emission, Emitted, Gather, Geometry};
use std::ops::Range;

pub(crate) const RTOL: f64 = 1e-12;
pub(crate) const ATOL: f64 = 1e-14;

/// Minimal 1D geometry on [0, width) that accepts every sample
pub(crate) struct TestSlab {
    pub width: f64,
    /// Test acceptance range for u\in[0,1) for rejection sampling
    accept: Range<f64>,
}

impl TestSlab {
    pub(crate) fn new(width: f64, accept: Range<f64>) -> Self {
        Self { width, accept }
    }
}

impl Geometry for TestSlab {
    type Point = f64;
    const SAMPLE_DIM: usize = 1;
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

impl<G: Geometry> Gather<G> for ConstGather {
    fn gather(
        &self,
        _geometry: &G,
        _src_atoms: &crate::atoms::Atoms<<G as Geometry>::Point>,
        _target_point: <G as Geometry>::Point,
    ) -> Vec<f64> {
        self.block.clone()
    }
}

/// Emission that returns fixed multiples of the flux as its scattering and
/// fission parts. If the gathered flux integrates to V, then
/// k = fission * V / (V - scatter * V) = fission / (1 - scatter).
pub(crate) struct ScaledEmission {
    scatter: f64,
    fission: f64,
}

impl ScaledEmission {
    pub(crate) fn new(scatter: f64, fission: f64) -> Self {
        Self { scatter, fission }
    }
}

impl<G: Geometry> Emission<G> for ScaledEmission {
    fn compute_emission(&self, _geometry: &G, _point: G::Point, flux_moments: &[f64]) -> Emitted {
        Emitted {
            scattering: flux_moments.iter().map(|m| self.scatter * m).collect(),
            fission: flux_moments.iter().map(|m| self.fission * m).collect(),
        }
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
