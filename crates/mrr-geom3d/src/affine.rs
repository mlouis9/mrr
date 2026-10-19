//! Affine maps y = A x + b in 3D.
//!
//! Used for boundary images: a reflection in a plane, a periodic translation,
//! and compositions of these (the image transforms `g` of the method of
//! images). Points transform with the full map; directions only with the
//! linear part A.

use crate::{DMat3, Vec3};

/// An affine map y = A x + b
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine {
    /// The linear part A
    pub linear: DMat3,
    /// The translation b
    pub translation: Vec3,
}

impl Affine {
    /// The identity map
    pub const IDENTITY: Affine = Affine {
        linear: DMat3::IDENTITY,
        translation: Vec3::ZERO,
    };

    pub fn new(linear: DMat3, translation: Vec3) -> Self {
        Self {
            linear,
            translation,
        }
    }

    /// x ↦ x + t
    pub fn translation(t: Vec3) -> Self {
        Self::new(DMat3::IDENTITY, t)
    }

    /// Reflection in the plane through `point` with normal `normal` (any
    /// non-zero length): x ↦ x − 2((x − p)·n)n, i.e. A = I − 2nnᵀ, b = 2(p·n)n.
    pub fn reflection(point: Vec3, normal: Vec3) -> Self {
        let n = normal.normalize();
        assert!(
            n.is_finite(),
            "reflection normal must be non-zero, got {normal}"
        );
        let a = DMat3::IDENTITY - 2.0 * outer(n, n);
        Self::new(a, 2.0 * point.dot(n) * n)
    }

    /// Map a point
    pub fn apply(&self, x: Vec3) -> Vec3 {
        self.linear * x + self.translation
    }

    /// Map a direction (or any displacement): the translation does not apply
    pub fn apply_dir(&self, v: Vec3) -> Vec3 {
        self.linear * v
    }

    /// `self ∘ inner`: apply `inner` first, then `self`.
    ///
    /// A(A_i x + b_i) + b = (A A_i) x + (A b_i + b)
    pub fn compose(&self, inner: &Self) -> Self {
        Self::new(
            self.linear * inner.linear,
            self.linear * inner.translation + self.translation,
        )
    }

    /// The inverse map, x = A⁻¹ y − A⁻¹ b. Panics if A is singular.
    pub fn inverse(&self) -> Self {
        let det = self.linear.determinant();
        assert!(
            det.abs() > f64::EPSILON,
            "affine map is singular (det A = {det:e})"
        );
        let inv = self.linear.inverse();
        Self::new(inv, -(inv * self.translation))
    }

    /// `true` if A is orthogonal (AᵀA = I) to within `tol` per entry. Every
    /// composition of reflections, rotations and translations is.
    pub fn is_orthogonal(&self, tol: f64) -> bool {
        (self.linear.transpose() * self.linear).abs_diff_eq(DMat3::IDENTITY, tol)
    }

    /// Inverse of a map with orthogonal A, using A⁻¹ = Aᵀ: exact (no
    /// division), cheaper, and never singular. For boundary image transforms.
    pub fn inverse_orthogonal(&self) -> Self {
        debug_assert!(
            self.is_orthogonal(1e-12),
            "inverse_orthogonal on a non-orthogonal map: {self:?}"
        );
        let at = self.linear.transpose();
        Self::new(at, -(at * self.translation))
    }
}

impl Default for Affine {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Outer product u vᵀ
fn outer(u: Vec3, v: Vec3) -> DMat3 {
    DMat3::from_cols(u * v.x, u * v.y, u * v.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    /// A map with A ≠ A⁻¹, A not symmetric, and b ≠ 0, so that a wrong
    /// inverse or a swapped composition order cannot pass by accident.
    fn generic() -> Affine {
        let a = DMat3::from_rotation_z(0.3)
            * DMat3::from_rotation_x(-0.7)
            * DMat3::from_diagonal(Vec3::new(1.0, 2.0, 0.5));
        Affine::new(a, Vec3::new(1.0, -2.0, 0.5))
    }

    /// Reflection in the plane x = 1
    fn mirror_x1() -> Affine {
        Affine::reflection(Vec3::new(1.0, 0.0, 0.0), Vec3::X)
    }

    fn points() -> [Vec3; 4] {
        [
            Vec3::ZERO,
            Vec3::new(0.3, -1.2, 2.0),
            Vec3::new(-5.0, 4.0, 0.1),
            Vec3::new(1.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn identity_does_nothing() {
        for x in points() {
            assert_eq!(Affine::IDENTITY.apply(x), x);
            assert_eq!(Affine::IDENTITY.apply_dir(x), x);
        }
        assert_eq!(Affine::default(), Affine::IDENTITY);
    }

    #[test]
    fn inverse_undoes_apply_on_both_sides() {
        let m = generic();
        let inv = m.inverse();
        for x in points() {
            assert_vec_close(inv.apply(m.apply(x)), x);
            assert_vec_close(m.apply(inv.apply(x)), x);
        }
    }

    #[test]
    fn composing_with_inverse_gives_identity() {
        let m = generic();
        for id in [m.compose(&m.inverse()), m.inverse().compose(&m)] {
            assert_mat_close(id.linear, DMat3::IDENTITY);
            assert_vec_close(id.translation, Vec3::ZERO);
        }
    }

    #[test]
    fn compose_applies_inner_first() {
        let (f, g) = (generic(), mirror_x1());
        for x in points() {
            assert_vec_close(f.compose(&g).apply(x), f.apply(g.apply(x)));
            assert_vec_close(g.compose(&f).apply(x), g.apply(f.apply(x)));
        }
        // The two orders differ, so the test can detect a swap
        let x = Vec3::new(0.3, -1.2, 2.0);
        assert!(!vec_is_close(
            f.compose(&g).apply(x),
            g.compose(&f).apply(x),
            Tol::EXACT
        ));
    }

    #[test]
    fn compose_is_associative() {
        let (f, g, h) = (
            generic(),
            mirror_x1(),
            Affine::translation(Vec3::new(0.0, 3.0, -1.0)),
        );
        let left = f.compose(&g).compose(&h);
        let right = f.compose(&g.compose(&h));
        assert_mat_close(left.linear, right.linear);
        assert_vec_close(left.translation, right.translation);
    }

    #[test]
    fn apply_dir_ignores_translation() {
        let m = generic();
        for v in points() {
            assert_vec_close(m.apply_dir(v), m.apply(v) - m.apply(Vec3::ZERO));
        }
        assert_eq!(Affine::translation(Vec3::ONE).apply_dir(Vec3::X), Vec3::X);
    }

    #[test]
    fn reflection_fixes_its_plane_and_flips_the_normal() {
        let r = mirror_x1();
        assert_vec_close(
            r.apply(Vec3::new(1.0, 2.0, -3.0)),
            Vec3::new(1.0, 2.0, -3.0),
        );
        assert_vec_close(
            r.apply(Vec3::new(0.0, 2.0, -3.0)),
            Vec3::new(2.0, 2.0, -3.0),
        );
        assert_vec_close(
            r.apply_dir(Vec3::new(1.0, 1.0, 0.0)),
            Vec3::new(-1.0, 1.0, 0.0),
        );
    }

    #[test]
    fn reflection_is_an_involution_for_any_plane() {
        let r = Affine::reflection(Vec3::new(0.2, -1.0, 3.0), Vec3::new(1.0, 2.0, -0.5));
        let rr = r.compose(&r);
        assert_mat_close(rr.linear, DMat3::IDENTITY);
        assert_vec_close(rr.translation, Vec3::ZERO);
        assert!(r.is_orthogonal(1e-12));
        assert_close(r.linear.determinant(), -1.0);
    }

    #[test]
    fn reflection_normal_length_does_not_matter() {
        let p = Vec3::new(0.0, 1.0, 0.0);
        let (a, b) = (
            Affine::reflection(p, Vec3::Y),
            Affine::reflection(p, 7.0 * Vec3::Y),
        );
        assert_mat_close(a.linear, b.linear);
        assert_vec_close(a.translation, b.translation);
    }

    #[test]
    fn two_parallel_mirrors_make_a_translation() {
        // Reflect in x = 0, then in x = 1: a translation by +2 along x
        let t = mirror_x1().compose(&Affine::reflection(Vec3::ZERO, Vec3::X));
        assert_mat_close(t.linear, DMat3::IDENTITY);
        assert_vec_close(t.translation, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn orthogonal_inverse_matches_general_inverse() {
        let m = mirror_x1()
            .compose(&Affine::reflection(Vec3::new(0.0, 0.5, 0.0), Vec3::Y))
            .compose(&Affine::translation(Vec3::new(4.0, 0.0, -2.0)));
        assert!(m.is_orthogonal(1e-12));
        let (a, b) = (m.inverse_orthogonal(), m.inverse());
        assert_mat_close(a.linear, b.linear);
        assert_vec_close(a.translation, b.translation);
        assert!(!generic().is_orthogonal(1e-12));
    }

    #[test]
    #[should_panic(expected = "singular")]
    fn singular_inverse_panics() {
        Affine::new(DMat3::from_diagonal(Vec3::new(1.0, 0.0, 1.0)), Vec3::ZERO).inverse();
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_normal_panics() {
        Affine::reflection(Vec3::ZERO, Vec3::ZERO);
    }
}
