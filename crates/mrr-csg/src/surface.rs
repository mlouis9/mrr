//! Surfaces: planes, spheres and axis-aligned cylinders.
//!
//! Every surface has a signed function `eval(p)` that is negative on one side
//! (the *negative half-space*: below a plane, inside a sphere or cylinder) and
//! positive on the other. Here `eval` is the **signed distance** to the
//! surface, so `|eval(p)| <= tol::LEN` means "on the surface" in the same
//! length units as every other tolerance.
//!
//! [`Surface::sense`] answers which side a point is on, with the shared tie
//! rule: a point on the surface takes the side its direction of travel points
//! into (the sign of `normal · dir`). [`Surface::distance`] gives the nearest
//! crossing ahead of a ray, ignoring roots within `tol::LEN` of the origin, so
//! a walker standing on (or just past) a surface does not find it again.

use mrr_geom3d::{Vec3, tol};

/// Coordinate axis of a cylinder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis {
    /// The two other axes, in increasing order
    fn others(self) -> (usize, usize) {
        match self {
            Axis::X => (1, 2),
            Axis::Y => (0, 2),
            Axis::Z => (0, 1),
        }
    }
}

/// A surface. Construct through the functions below, which validate input
/// and normalize plane normals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Surface {
    /// `n · p = d` with `|n| = 1`; negative side is `n · p < d`
    Plane { n: Vec3, d: f64 },
    /// `|p - c| = r`; negative side is inside
    Sphere { c: Vec3, r: f64 },
    /// Infinite cylinder of radius `r` along `axis` through `c` (the coordinate
    /// of `c` along `axis` is ignored); negative side is inside
    Cylinder { axis: Axis, c: Vec3, r: f64 },
}

impl Surface {
    /// Plane `normal · p = offset · |normal|`, i.e. the plane through
    /// `offset · n̂` with unit normal `n̂ = normal / |normal|`
    pub fn plane(normal: Vec3, offset: f64) -> Self {
        let len = normal.length();
        assert!(
            len > 0.0 && len.is_finite(),
            "plane normal must be non-zero and finite"
        );
        assert!(offset.is_finite(), "plane offset must be finite");
        Surface::Plane {
            n: normal / len,
            d: offset,
        }
    }

    /// Plane `x[axis] = value`, negative side below
    pub fn axis_plane(axis: Axis, value: f64) -> Self {
        let mut n = Vec3::ZERO;
        n[axis as usize] = 1.0;
        Self::plane(n, value)
    }

    pub fn sphere(c: Vec3, r: f64) -> Self {
        assert!(c.is_finite() && r > 0.0 && r.is_finite(), "invalid sphere");
        Surface::Sphere { c, r }
    }

    pub fn cylinder(axis: Axis, c: Vec3, r: f64) -> Self {
        assert!(
            c.is_finite() && r > 0.0 && r.is_finite(),
            "invalid cylinder"
        );
        Surface::Cylinder { axis, c, r }
    }

    /// Signed distance to the surface: negative on the negative side
    pub fn eval(&self, p: Vec3) -> f64 {
        match *self {
            Surface::Plane { n, d } => n.dot(p) - d,
            Surface::Sphere { c, r } => (p - c).length() - r,
            Surface::Cylinder { axis, c, r } => {
                let (i, j) = axis.others();
                (p[i] - c[i]).hypot(p[j] - c[j]) - r
            }
        }
    }

    /// Outward unit normal (pointing to the positive side) at `p`. At the
    /// centre of a sphere or on a cylinder's axis it is undefined; returns zero.
    pub fn normal(&self, p: Vec3) -> Vec3 {
        match *self {
            Surface::Plane { n, .. } => n,
            Surface::Sphere { c, .. } => (p - c).normalize_or_zero(),
            Surface::Cylinder { axis, c, .. } => {
                let (i, j) = axis.others();
                let mut v = Vec3::ZERO;
                v[i] = p[i] - c[i];
                v[j] = p[j] - c[j];
                v.normalize_or_zero()
            }
        }
    }

    /// `true` if `p` is on the positive side.
    ///
    /// Off the surface (`|eval| > tol::LEN`) this is the sign of `eval`. On it,
    /// the direction decides: positive if `dir` points to the positive side
    /// (`normal · dir > 0`). A direction tangent to the surface falls back to
    /// the sign of `eval` (with 0 counting as positive).
    pub fn sense(&self, p: Vec3, dir: Vec3) -> bool {
        let f = self.eval(p);
        if f.abs() > tol::LEN {
            return f > 0.0;
        }
        let along = self.normal(p).dot(dir);
        if along != 0.0 { along > 0.0 } else { f >= 0.0 }
    }

    /// Distance along `origin + t·dir` (`dir` a unit vector) to the nearest
    /// crossing with `t > tol::LEN`, or `None` if the ray never crosses.
    ///
    /// Tangent rays (a double root) do not cross and return `None`.
    pub fn distance(&self, origin: Vec3, dir: Vec3) -> Option<f64> {
        match *self {
            Surface::Plane { n, d } => {
                let denom = n.dot(dir);
                if denom == 0.0 {
                    return None;
                }
                let t = (d - n.dot(origin)) / denom;
                (t > tol::LEN).then_some(t)
            }
            Surface::Sphere { c, r } => {
                let o = origin - c;
                // t² + 2 b t + q = 0
                let b = o.dot(dir);
                let q = o.length_squared() - r * r;
                smallest_root_ahead(1.0, b, q)
            }
            Surface::Cylinder { axis, c, r } => {
                let (i, j) = axis.others();
                let (ox, oy) = (origin[i] - c[i], origin[j] - c[j]);
                let (dx, dy) = (dir[i], dir[j]);
                let a = dx * dx + dy * dy;
                if a == 0.0 {
                    return None; // parallel to the axis
                }
                let b = ox * dx + oy * dy;
                let q = ox * ox + oy * oy - r * r;
                smallest_root_ahead(a, b, q)
            }
        }
    }
}

/// Smallest root `t > tol::LEN` of `a t² + 2 b t + q = 0` (`a > 0`), using the
/// cancellation-free form of the quadratic formula. Tangency (zero
/// discriminant) is not a crossing.
fn smallest_root_ahead(a: f64, b: f64, q: f64) -> Option<f64> {
    let disc = b * b - a * q;
    if disc <= 0.0 {
        return None;
    }
    // k = -(b + sign(b) √disc) never cancels and is non-zero for disc > 0
    // (f64::signum(0.0) = 1); the roots are k / a and q / k
    let k = -(b + b.signum() * disc.sqrt());
    let (r1, r2) = (k / a, q / k);
    let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
    if lo > tol::LEN {
        Some(lo)
    } else if hi > tol::LEN {
        Some(hi)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_geom3d::testing::*;

    fn unit_sphere() -> Surface {
        Surface::sphere(Vec3::ZERO, 1.0)
    }

    #[test]
    fn eval_is_signed_distance() {
        let p = Surface::axis_plane(Axis::X, 2.0);
        assert_close(p.eval(Vec3::new(5.0, 1.0, 1.0)), 3.0);
        assert_close(p.eval(Vec3::new(-1.0, 0.0, 0.0)), -3.0);
        // Oblique plane n̂ = (1,1,0)/√2 at offset √2: passes through (1,1,0)
        let q = Surface::plane(Vec3::new(1.0, 1.0, 0.0), 2f64.sqrt());
        assert_close(q.eval(Vec3::new(1.0, 1.0, 0.0)), 0.0);
        assert_close(q.eval(Vec3::ZERO), -2f64.sqrt());
        assert_close(unit_sphere().eval(Vec3::new(0.0, 3.0, 0.0)), 2.0);
        assert_close(unit_sphere().eval(Vec3::ZERO), -1.0);
        let cyl = Surface::cylinder(Axis::Z, Vec3::new(1.0, 1.0, 99.0), 0.5);
        assert_close(cyl.eval(Vec3::new(1.0, 2.0, -7.0)), 0.5); // axis coordinate ignored
    }

    #[test]
    fn normals_point_to_the_positive_side() {
        assert_eq!(
            Surface::axis_plane(Axis::Y, 0.0).normal(Vec3::ZERO),
            Vec3::Y
        );
        assert_vec_close(unit_sphere().normal(Vec3::new(0.0, 0.0, 2.0)), Vec3::Z);
        let cyl = Surface::cylinder(Axis::X, Vec3::ZERO, 1.0);
        assert_vec_close(cyl.normal(Vec3::new(5.0, 0.0, -3.0)), -Vec3::Z);
        assert_eq!(unit_sphere().normal(Vec3::ZERO), Vec3::ZERO);
    }

    #[test]
    fn sense_off_and_on_the_surface() {
        let s = unit_sphere();
        assert!(!s.sense(Vec3::new(0.5, 0.0, 0.0), Vec3::X));
        assert!(s.sense(Vec3::new(1.5, 0.0, 0.0), -Vec3::X));
        // On the surface: the direction decides
        let on = Vec3::new(1.0, 0.0, 0.0);
        assert!(s.sense(on, Vec3::X)); // heading out
        assert!(!s.sense(on, -Vec3::X)); // heading in
        // Within tol::LEN counts as on the surface
        let near = Vec3::new(1.0 + 0.5 * tol::LEN, 0.0, 0.0);
        assert!(!s.sense(near, -Vec3::X));
        // Tangent direction falls back to the sign of eval
        assert!(s.sense(on, Vec3::Y));
    }

    #[test]
    fn sense_on_surface_matches_a_small_step() {
        // The tie rule must agree with where a tiny step along dir lands
        let surfaces = [
            Surface::plane(Vec3::new(1.0, -2.0, 0.5), 0.3),
            unit_sphere(),
            Surface::cylinder(Axis::Y, Vec3::new(0.2, 0.0, -0.1), 0.7),
        ];
        let dirs = [
            Vec3::new(0.3, 0.4, -0.866),
            Vec3::new(-0.6, 0.0, 0.8),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.9, 0.1, 0.3).normalize(),
        ];
        for s in surfaces {
            // A point on the surface: project a point along the normal
            let start = Vec3::new(0.4, 0.3, 0.2);
            let on = start - s.eval(start) * s.normal(start);
            assert!(s.eval(on).abs() < 1e-12);
            for d in dirs {
                let step = on + 1e-6 * d;
                if s.normal(on).dot(d).abs() > 1e-3 {
                    assert_eq!(s.sense(on, d), s.eval(step) > 0.0, "{s:?} dir {d:?}");
                }
            }
        }
    }

    #[test]
    fn plane_distance() {
        let p = Surface::axis_plane(Axis::Z, 2.0);
        assert_close(p.distance(Vec3::ZERO, Vec3::Z).unwrap(), 2.0);
        let d = Vec3::new(0.0, 0.6, 0.8);
        assert_close(p.distance(Vec3::ZERO, d).unwrap(), 2.5);
        assert_eq!(p.distance(Vec3::ZERO, -Vec3::Z), None); // behind
        assert_eq!(p.distance(Vec3::ZERO, Vec3::X), None); // parallel
        assert_eq!(p.distance(Vec3::new(0.0, 0.0, 2.0), Vec3::Z), None); // on it
    }

    #[test]
    fn sphere_chords() {
        let s = unit_sphere();
        // From outside: near side
        assert_close(s.distance(Vec3::new(-3.0, 0.0, 0.0), Vec3::X).unwrap(), 2.0);
        // From inside: far side
        assert_close(s.distance(Vec3::ZERO, Vec3::X).unwrap(), 1.0);
        assert_close(s.distance(Vec3::new(0.5, 0.0, 0.0), -Vec3::X).unwrap(), 1.5);
        // Offset chord: impact parameter 0.6 -> half chord 0.8
        let o = Vec3::new(-2.0, 0.6, 0.0);
        assert_close(s.distance(o, Vec3::X).unwrap(), 2.0 - 0.8);
        // Miss, and pointing away
        assert_eq!(s.distance(Vec3::new(-2.0, 1.5, 0.0), Vec3::X), None);
        assert_eq!(s.distance(Vec3::new(-2.0, 0.0, 0.0), -Vec3::X), None);
    }

    #[test]
    fn sphere_from_its_own_surface() {
        let s = unit_sphere();
        let on = Vec3::new(1.0, 0.0, 0.0);
        // Heading inward: the far side, a full diameter away
        assert_close(s.distance(on, -Vec3::X).unwrap(), 2.0);
        // Heading outward: never crosses again
        assert_eq!(s.distance(on, Vec3::X), None);
        // Just past the surface after a bump, heading in: still the far side
        let bumped = Vec3::new(1.0 - tol::BUMP, 0.0, 0.0);
        assert_close_tol(
            s.distance(bumped, -Vec3::X).unwrap(),
            2.0 - tol::BUMP,
            Tol::NUMERICAL,
        );
    }

    #[test]
    fn grazing_and_tangent_rays() {
        let s = unit_sphere();
        // Exactly tangent: no crossing
        assert_eq!(s.distance(Vec3::new(-2.0, 1.0, 0.0), Vec3::X), None);
        // Nearly tangent from inside: short chord near the surface still found
        let o = Vec3::new(0.0, 1.0 - 1e-8, 0.0);
        let t = s.distance(o, Vec3::X).unwrap();
        let exact = (1.0 - (1.0 - 1e-8f64).powi(2)).sqrt();
        assert_close_tol(t, exact, Tol::new(1e-6, 0.0));
    }

    #[test]
    fn cylinder_chords() {
        let c = Surface::cylinder(Axis::Z, Vec3::new(1.0, 0.0, 0.0), 0.5);
        // Across the diameter, perpendicular to the axis
        assert_close(c.distance(Vec3::new(-1.0, 0.0, 3.0), Vec3::X).unwrap(), 1.5);
        // Oblique: only the radial component counts
        let d = Vec3::new(0.6, 0.0, 0.8);
        assert_close(c.distance(Vec3::new(0.0, 0.0, 0.0), d).unwrap(), 0.5 / 0.6);
        // From inside, far wall
        assert_close(c.distance(Vec3::new(1.0, 0.0, 0.0), Vec3::Y).unwrap(), 0.5);
        // Parallel to the axis
        assert_eq!(c.distance(Vec3::new(1.0, 0.0, 0.0), Vec3::Z), None);
    }

    #[test]
    fn random_rays_land_on_the_surface() {
        let surfaces = [
            Surface::plane(Vec3::new(0.3, -1.0, 2.0), 0.4),
            Surface::sphere(Vec3::new(0.1, 0.2, -0.3), 0.8),
            Surface::cylinder(Axis::X, Vec3::new(0.0, 0.3, 0.1), 0.6),
        ];
        let mut state = 3u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for s in surfaces {
            for _ in 0..500 {
                let o = Vec3::new(rnd(), rnd(), rnd()) * 3.0;
                let d = Vec3::new(rnd(), rnd(), rnd()).normalize();
                if let Some(t) = s.distance(o, d) {
                    assert!(t > tol::LEN);
                    assert!(s.eval(o + t * d).abs() < 1e-12, "{s:?}");
                    // No crossing strictly between: sense constant on (0, t)
                    let a = s.eval(o + 0.25 * t * d).signum();
                    let b = s.eval(o + 0.75 * t * d).signum();
                    assert_eq!(a, b);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_plane_normal_panics() {
        Surface::plane(Vec3::ZERO, 1.0);
    }
}
