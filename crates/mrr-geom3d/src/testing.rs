//! Test helpers for 3D types, built on the workspace-wide rule in
//! `mrr_math::testing` (re-exported here, so geometry tests need one import).
//!
//! Vectors and matrices are compared component by component with the same
//! `|a - b| <= atol + rtol * |b|` rule. Compiled in normal builds so that
//! backend crates (e.g. `mrr-csg`) can use it in their own tests.

pub use mrr_math::testing::*;

use crate::{DMat3, Vec3};

/// `true` if every component of `actual` is close to `expected`
pub fn vec_is_close(actual: Vec3, expected: Vec3, tol: Tol) -> bool {
    (0..3).all(|i| is_close(actual[i], expected[i], tol))
}

/// Assert two vectors are close with [`Tol::EXACT`]
#[track_caller]
pub fn assert_vec_close(actual: Vec3, expected: Vec3) {
    assert_vec_close_tol(actual, expected, Tol::EXACT);
}

/// Assert two vectors are close with an explicit tolerance
#[track_caller]
pub fn assert_vec_close_tol(actual: Vec3, expected: Vec3, tol: Tol) {
    assert!(
        vec_is_close(actual, expected, tol),
        "vectors not close: actual {actual:?}, expected {expected:?} (tol {tol:?})"
    );
}

/// Assert two matrices are close (column by column) with [`Tol::EXACT`]
#[track_caller]
pub fn assert_mat_close(actual: DMat3, expected: DMat3) {
    let close = (0..3).all(|c| vec_is_close(actual.col(c), expected.col(c), Tol::EXACT));
    assert!(
        close,
        "matrices not close:\nactual   {actual:?}\nexpected {expected:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_vectors_and_matrices_pass() {
        assert_vec_close(
            Vec3::new(1.0, 0.0, -2.0),
            Vec3::new(1.0 + 1e-13, 1e-15, -2.0),
        );
        assert_mat_close(DMat3::IDENTITY * (1.0 + 1e-13), DMat3::IDENTITY);
    }

    #[test]
    #[should_panic(expected = "vectors not close")]
    fn far_vectors_fail() {
        assert_vec_close(Vec3::X, Vec3::Y);
    }

    #[test]
    #[should_panic(expected = "matrices not close")]
    fn far_matrices_fail() {
        assert_mat_close(DMat3::IDENTITY, DMat3::ZERO);
    }
}
