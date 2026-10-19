//! Shared floating-point comparison for tests across the workspace.
//!
//! Two values are close when `|a - b| <= atol + rtol * |b|`, with `b` the
//! expected value (the rule numpy's `isclose` uses). `rtol` covers values of
//! ordinary size; `atol` covers expected values at or near zero, where any
//! relative test fails on round-off alone.
//!
//! # Which tolerance to use
//!
//! | Preset            | rtol    | atol    | Use when the two sides differ by …                     |
//! |-------------------|---------|---------|--------------------------------------------------------|
//! | [`Tol::EXACT`]    | 1e-12   | 1e-14   | a handful of roundings: the same formula evaluated two ways, sums of a few terms, values from closed forms |
//! | [`Tol::NUMERICAL`]| 1e-10   | 1e-12   | an independent *numerical method*: quadrature, series vs continued fraction, recurrences run over many steps |
//!
//! f64 has a unit round-off of about 1.1e-16, so `EXACT` leaves room for
//! roughly 10⁴ roundings of relative size ε. That is generous for
//! formula-vs-formula checks yet still catches any genuine bug, which shows
//! up as an O(1) or at least O(1e-6) discrepancy. `NUMERICAL` allows for the
//! truncation error of a well-converged approximation.
//!
//! Anything looser (statistical tests, benchmark data quoted to a few
//! digits) is not a preset: write the tolerance at the call site with a
//! comment explaining where it comes from.
//!
//! This module is compiled in normal builds (it is tiny and has no
//! dependencies) so that other crates' tests can use it; `#[cfg(test)]` would
//! hide it from them. Add `mrr-math` as a `[dev-dependencies]` entry in
//! crates that only need it for tests.

/// Relative and absolute tolerance pair
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tol {
    pub rtol: f64,
    pub atol: f64,
}

impl Tol {
    /// Same quantity computed two ways with a few roundings (the default)
    pub const EXACT: Tol = Tol {
        rtol: 1e-12,
        atol: 1e-14,
    };

    /// Compared against an independent numerical approximation
    pub const NUMERICAL: Tol = Tol {
        rtol: 1e-10,
        atol: 1e-12,
    };

    pub const fn new(rtol: f64, atol: f64) -> Tol {
        Tol { rtol, atol }
    }

    /// Allowed difference when the expected value is `expected`
    pub fn bound(&self, expected: f64) -> f64 {
        self.atol + self.rtol * expected.abs()
    }
}

// Presets get stricter-to-looser in order (checked at compile time)
const _: () =
    assert!(Tol::EXACT.rtol < Tol::NUMERICAL.rtol && Tol::EXACT.atol < Tol::NUMERICAL.atol);

/// Default relative tolerance (`Tol::EXACT.rtol`)
pub const RTOL: f64 = Tol::EXACT.rtol;
/// Default absolute tolerance (`Tol::EXACT.atol`)
pub const ATOL: f64 = Tol::EXACT.atol;

/// `true` if `actual` is within `tol` of `expected`. NaN is never close.
pub fn is_close(actual: f64, expected: f64, tol: Tol) -> bool {
    (actual - expected).abs() <= tol.bound(expected)
}

/// Assert `actual` is close to `expected` with [`Tol::EXACT`]
#[track_caller]
pub fn assert_close(actual: f64, expected: f64) {
    assert_close_tol(actual, expected, Tol::EXACT);
}

/// Assert `actual` is close to `expected` with an explicit tolerance
#[track_caller]
pub fn assert_close_tol(actual: f64, expected: f64, tol: Tol) {
    let diff = (actual - expected).abs();
    let bound = tol.bound(expected);
    assert!(
        diff <= bound,
        "not close: actual {actual:e}, expected {expected:e} (|diff| = {diff:e} > tol = {bound:e})"
    );
}

/// Slice version of [`assert_close`]: same length, every entry close
#[track_caller]
pub fn assert_all_close(actual: &[f64], expected: &[f64]) {
    assert_all_close_tol(actual, expected, Tol::EXACT);
}

/// Slice version of [`assert_close_tol`]
#[track_caller]
pub fn assert_all_close_tol(actual: &[f64], expected: &[f64], tol: Tol) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - b).abs();
        let bound = tol.bound(b);
        assert!(
            diff <= bound,
            "not close at index {i}: actual {a:e}, expected {b:e} (|diff| = {diff:e} > tol = {bound:e})"
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
        assert_close_tol(1.0, 1.0 + 1e-11, Tol::NUMERICAL);
    }

    #[test]
    fn is_close_rejects_nan_and_far_values() {
        assert!(!is_close(f64::NAN, 1.0, Tol::EXACT));
        assert!(!is_close(1.0, f64::NAN, Tol::EXACT));
        assert!(!is_close(1.0, 1.0 + 1e-6, Tol::NUMERICAL));
        assert!(is_close(1e6, 1e6 + 1e-7, Tol::EXACT)); // relative part scales
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
