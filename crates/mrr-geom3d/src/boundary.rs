//! The outer boundary of the problem: an axis-aligned box with a boundary
//! condition on each face, and the method of images it implies.
//!
//! # Boundary conditions
//!
//! Each of the six faces is [`Bc::Vacuum`], [`Bc::Reflective`] or
//! [`Bc::Periodic`] (periodic faces come in pairs on an axis). The walker uses
//! [`BoundaryBox::face_hit`] to find where a straight ray leaves the box and
//! [`BoundaryBox::cross`] to apply the condition there: stop (vacuum), mirror
//! the direction (reflective) or jump to the opposite face (periodic).
//!
//! # The method of images (unfolding)
//!
//! A reflected path is a straight line in *unfolded* space: tile space with
//! copies of the box, mirroring the copy each time a reflective face is
//! crossed (translating it, for a periodic face). On each axis `a`, with
//! `l = lower[a]`, `L = size[a]` and `u = p[a] - l`, copy `t ∈ ℤ` occupies
//! `[l + tL, l + (t+1)L]` and the image of `p` in it is
//!
//! ```text
//! reflective-type axis:  x'_a = l + tL + (u      if t even,
//!                                         L - u  if t odd)
//! periodic axis:         x'_a = p_a + tL
//! ```
//!
//! `t = [t_x, t_y, t_z]` indexes the image; `|t_a|` is the number of times the
//! path crosses a face of axis `a`. Facts used throughout:
//!
//! - **Image ↔ physical path.** A straight segment from `x` to an image
//!   `x' = g·y` is the unfolded form of exactly one physical path from `x` to
//!   `y` that reflects specularly at the box faces (the angle of incidence
//!   equals the angle of reflection because the unfolding is a mirror). Its
//!   length is `|x' − x|` and its reflection points are the crossings of the
//!   segment with the tiling planes `l + mL`, folded back into the box
//!   ([`crossings`](BoundaryBox::crossings)).
//! - **Sum over images, not over face sequences.** For perpendicular mirrors
//!   the order of hits ("x then y" vs "y then x") is set by geometry, not
//!   chosen, so each `t` is one path. Summing over `t` counts every path once.
//! - **The map is affine and orthogonal.** `g = transform(t)` has linear part
//!   `diag(±1)` (−1 on reflective-type axes with odd `t_a`) and a translation,
//!   so directions transform by the sign flips alone: the emission direction
//!   at the source is the arrival direction pulled back through `g`.
//! - **Admissibility.** Along a reflective-type axis, an image with `t_a > 0`
//!   crosses the faces upper, lower, upper, … (`|t_a|` crossings), and with
//!   `t_a < 0` lower, upper, lower, …. The image exists only if every face it
//!   crosses is reflective ([`admissible`](BoundaryBox::admissible)); any
//!   vacuum face on the way means the physical path would have left the
//!   domain. Periodic axes admit every `t_a`; an axis with two vacuum faces
//!   admits only `t_a = 0`.
//!
//! Mixed conditions work per face, e.g. a reflective lower `z` face (a
//! symmetry plane) with a vacuum upper face admits `t_z ∈ {0, −1}` only.

use std::fmt;

use crate::{Affine, DMat3, Vec3, tol};

/// Condition on one face of the box
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bc {
    /// Particles leave and never return
    Vacuum,
    /// Specular reflection (a symmetry plane)
    Reflective,
    /// Leaving through this face re-enters through the opposite one
    Periodic,
}

/// Which face of an axis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Lower = 0,
    Upper = 1,
}

/// Why a `BoundaryBox` could not be built
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryError {
    /// A corner coordinate is NaN or infinite
    NonFinite,
    /// `upper[axis] <= lower[axis]`
    EmptyAxis { axis: usize },
    /// Only one face of `axis` is periodic
    UnpairedPeriodic { axis: usize },
}

impl fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BoundaryError::NonFinite => write!(f, "box corners must be finite"),
            BoundaryError::EmptyAxis { axis } => {
                write!(f, "box has zero or negative size along axis {axis}")
            }
            BoundaryError::UnpairedPeriodic { axis } => write!(
                f,
                "periodic boundary on only one face of axis {axis}; both faces must be periodic"
            ),
        }
    }
}

impl std::error::Error for BoundaryError {}

/// Where a ray from inside the box first reaches its boundary. Several faces
/// are hit at once when the ray passes through an edge or corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceHit {
    /// Distance along the ray to the face(s)
    pub dist: f64,
    /// `faces[axis]` is the face of that axis hit at `dist`, if any
    pub faces: [Option<Side>; 3],
}

impl FaceHit {
    /// The (axis, side) pairs hit, in axis order
    pub fn iter(&self) -> impl Iterator<Item = (usize, Side)> + '_ {
        (0..3).filter_map(|a| self.faces[a].map(|side| (a, side)))
    }
}

/// Result of applying the boundary conditions where a ray meets the box
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Crossed {
    /// At least one face hit is vacuum: the walk ends here
    Exit,
    /// Continue from `point` along `dir` (reflected and/or translated)
    Continue { point: Vec3, dir: Vec3 },
}

/// One crossing of a box face by an unfolded segment (see [`BoundaryBox::crossings`])
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crossing {
    /// Path length from the segment's start
    pub s: f64,
    pub axis: usize,
    /// The physical face crossed
    pub side: Side,
    pub bc: Bc,
    /// The crossing point folded back into the box (it lies on the face)
    pub point: Vec3,
}

/// Axis-aligned box `[lower, upper]` with a condition on each face
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryBox {
    lower: Vec3,
    size: Vec3,
    /// `faces[axis][side as usize]`
    faces: [[Bc; 2]; 3],
}

impl BoundaryBox {
    /// Box from corners and a condition per face, `faces[axis] = [lower, upper]`
    pub fn new(lower: Vec3, upper: Vec3, faces: [[Bc; 2]; 3]) -> Result<Self, BoundaryError> {
        if !(lower.is_finite() && upper.is_finite()) {
            return Err(BoundaryError::NonFinite);
        }
        for axis in 0..3 {
            if upper[axis] <= lower[axis] {
                return Err(BoundaryError::EmptyAxis { axis });
            }
            let [lo, hi] = faces[axis];
            if (lo == Bc::Periodic) != (hi == Bc::Periodic) {
                return Err(BoundaryError::UnpairedPeriodic { axis });
            }
        }
        Ok(Self {
            lower,
            size: upper - lower,
            faces,
        })
    }

    /// Box with the same condition on all six faces
    pub fn uniform(lower: Vec3, upper: Vec3, bc: Bc) -> Result<Self, BoundaryError> {
        Self::new(lower, upper, [[bc; 2]; 3])
    }

    pub fn lower(&self) -> Vec3 {
        self.lower
    }

    pub fn upper(&self) -> Vec3 {
        self.lower + self.size
    }

    pub fn size(&self) -> Vec3 {
        self.size
    }

    pub fn volume(&self) -> f64 {
        self.size.x * self.size.y * self.size.z
    }

    pub fn bc(&self, axis: usize, side: Side) -> Bc {
        self.faces[axis][side as usize]
    }

    fn is_periodic(&self, axis: usize) -> bool {
        self.faces[axis][0] == Bc::Periodic
    }

    /// `true` if `p` is in the closed box
    pub fn contains(&self, p: Vec3) -> bool {
        let (lo, hi) = (self.lower, self.upper());
        (0..3).all(|a| p[a] >= lo[a] && p[a] <= hi[a])
    }

    /// Euclidean distance from `p` to the closed box (0 inside)
    pub fn distance_to(&self, p: Vec3) -> f64 {
        let (lo, hi) = (self.lower, self.upper());
        let outside = Vec3::new(
            (lo.x - p.x).max(0.0).max(p.x - hi.x),
            (lo.y - p.y).max(0.0).max(p.y - hi.y),
            (lo.z - p.z).max(0.0).max(p.z - hi.z),
        );
        outside.length()
    }

    // ------------------------------------------------------------------
    // Rays meeting the boundary (used by walkers)
    // ------------------------------------------------------------------

    /// Nearest face reached by the ray `origin + d·dir` (`origin` inside the
    /// box, `dir` non-zero). Every face reached within `tol::LEN` of the
    /// nearest is reported, so edges and corners are handled in one step.
    pub fn face_hit(&self, origin: Vec3, dir: Vec3) -> FaceHit {
        debug_assert!(
            self.distance_to(origin) <= tol::LEN,
            "face_hit from outside the box: {origin:?}"
        );
        let (lo, hi) = (self.lower, self.upper());
        let mut dists = [f64::INFINITY; 3];
        let mut sides = [Side::Lower; 3];
        for a in 0..3 {
            if dir[a] > 0.0 {
                dists[a] = ((hi[a] - origin[a]) / dir[a]).max(0.0);
                sides[a] = Side::Upper;
            } else if dir[a] < 0.0 {
                dists[a] = ((lo[a] - origin[a]) / dir[a]).max(0.0);
                sides[a] = Side::Lower;
            }
        }
        let dist = dists.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(dist.is_finite(), "face_hit with a zero direction");
        let mut faces = [None; 3];
        for a in 0..3 {
            if dists[a] <= dist + tol::LEN {
                faces[a] = Some(sides[a]);
            }
        }
        FaceHit { dist, faces }
    }

    /// Apply the boundary conditions at the face(s) of `hit`, for a ray that
    /// has just reached `point` moving along `dir`.
    ///
    /// - Any vacuum face hit: [`Crossed::Exit`].
    /// - Reflective: the direction component along that axis is negated, and
    ///   the point is snapped exactly onto the face.
    /// - Periodic: the point moves to the opposite face; direction unchanged.
    ///
    /// At an edge or corner every face is applied at once (reflecting in two
    /// perpendicular faces = reversing both components), which is the limit of
    /// hitting them in either order.
    pub fn cross(&self, point: Vec3, dir: Vec3, hit: &FaceHit) -> Crossed {
        if hit.iter().any(|(a, side)| self.bc(a, side) == Bc::Vacuum) {
            return Crossed::Exit;
        }
        let (lo, hi) = (self.lower, self.upper());
        let (mut point, mut dir) = (point, dir);
        for (a, side) in hit.iter() {
            let face = if side == Side::Upper { hi[a] } else { lo[a] };
            match self.bc(a, side) {
                Bc::Reflective => {
                    point[a] = face;
                    dir[a] = -dir[a];
                }
                Bc::Periodic => {
                    point[a] = if side == Side::Upper { lo[a] } else { hi[a] };
                }
                Bc::Vacuum => unreachable!("vacuum handled above"),
            }
        }
        Crossed::Continue { point, dir }
    }

    // ------------------------------------------------------------------
    // Method of images
    // ------------------------------------------------------------------

    /// Image of `p` in copy `t` of the box (see the module docs). Defined for
    /// every `t`; use [`admissible`](Self::admissible) to know whether it
    /// corresponds to a physical path.
    pub fn image(&self, t: [i32; 3], p: Vec3) -> Vec3 {
        let mut out = p;
        for a in 0..3 {
            let (l, len, tf) = (self.lower[a], self.size[a], t[a] as f64);
            out[a] = if self.is_periodic(a) {
                p[a] + tf * len
            } else {
                let u = p[a] - l;
                l + tf * len + if t[a] % 2 == 0 { u } else { len - u }
            };
        }
        out
    }

    /// The affine map `g` with `g(p) = image(t, p)` for all `p`. Its linear part
    /// is `diag(±1)`, so it is orthogonal and `inverse_orthogonal` applies.
    pub fn transform(&self, t: [i32; 3]) -> Affine {
        let mut signs = Vec3::ONE;
        let mut shift = Vec3::ZERO;
        for a in 0..3 {
            let (l, len, tf) = (self.lower[a], self.size[a], t[a] as f64);
            if self.is_periodic(a) || t[a] % 2 == 0 {
                shift[a] = tf * len;
            } else {
                // l + tL + L − (p − l) = −p + 2l + (t+1)L
                signs[a] = -1.0;
                shift[a] = 2.0 * l + (tf + 1.0) * len;
            }
        }
        Affine::new(DMat3::from_diagonal(signs), shift)
    }

    /// `true` if image `t` corresponds to a physical path: every face crossed
    /// on the way is reflective (or the axis is periodic).
    pub fn admissible(&self, t: [i32; 3]) -> bool {
        (0..3).all(|a| {
            if self.is_periodic(a) || t[a] == 0 {
                return true;
            }
            // First face crossed, then alternating
            let (first, second) = if t[a] > 0 {
                (Side::Upper, Side::Lower)
            } else {
                (Side::Lower, Side::Upper)
            };
            let n = t[a].unsigned_abs();
            self.bc(a, first) == Bc::Reflective && (n < 2 || self.bc(a, second) == Bc::Reflective)
        })
    }

    /// Every admissible `t` whose image of `p` lies within distance `r` of the
    /// box, appended to `out` in no particular order. `t = [0, 0, 0]` is always
    /// included for `p` inside the box.
    ///
    /// Use: points `x` inside the box with `|x − image(t, p)| < r` are exactly
    /// the points within *path* distance `r` of `p` via `t`. A radius query
    /// over atoms therefore runs once per returned `t`, centred on
    /// `image(t, p)`.
    pub fn images_within(&self, p: Vec3, r: f64, out: &mut Vec<[i32; 3]>) {
        assert!(
            r >= 0.0 && r.is_finite(),
            "radius must be finite and non-negative, got {r}"
        );
        // Candidate range per axis: copies whose slab lies within r of the box
        let mut range = [(0i32, 0i32); 3];
        for (a, rng) in range.iter_mut().enumerate() {
            let any_reflective = self.faces[a].contains(&Bc::Reflective);
            if self.is_periodic(a) || any_reflective {
                let k = (r / self.size[a]).floor() as i32 + 1;
                *rng = (-k, k);
            }
        }
        for tx in range[0].0..=range[0].1 {
            for ty in range[1].0..=range[1].1 {
                for tz in range[2].0..=range[2].1 {
                    let t = [tx, ty, tz];
                    if self.admissible(t)
                        && self.distance_to(self.image(t, p)) < r.max(f64::MIN_POSITIVE)
                    {
                        out.push(t);
                    }
                }
            }
        }
    }

    /// Map an unfolded point back into the box: the triangle wave on
    /// reflective-type axes, modulo the size on periodic ones. `fold` inverts
    /// `image`: `fold(image(t, p)) == p` for `p` in the box.
    pub fn fold(&self, y: Vec3) -> Vec3 {
        let mut out = y;
        for a in 0..3 {
            let (l, len) = (self.lower[a], self.size[a]);
            out[a] = if self.is_periodic(a) {
                l + (y[a] - l).rem_euclid(len)
            } else {
                let r = (y[a] - l).rem_euclid(2.0 * len); // in [0, 2L)
                l + len - (r - len).abs()
            };
        }
        out
    }

    /// The boundary crossings of the straight unfolded segment from `x` (in the
    /// box) to `xp` (e.g. an image `image(t, y)`), in order along the segment,
    /// appended to `out`. Each crossing is folded back into the box, so its
    /// `point` lies on the physical face crossed.
    ///
    /// For an admissible image this is exactly the sequence of reflections
    /// (and periodic jumps) of the physical path from `x` to `fold(xp)`:
    /// `Σ|t_a|` crossings. Crossings at the same `s` (through an edge or
    /// corner) are both reported; their relative order is immaterial.
    pub fn crossings(&self, x: Vec3, xp: Vec3, out: &mut Vec<Crossing>) {
        let delta = xp - x;
        let len = delta.length();
        if len == 0.0 {
            return;
        }
        let dir = delta / len;
        let start = out.len();
        for a in 0..3 {
            if dir[a] == 0.0 {
                continue;
            }
            let (l, size) = (self.lower[a], self.size[a]);
            let (lo_c, hi_c) = if x[a] < xp[a] {
                (x[a], xp[a])
            } else {
                (xp[a], x[a])
            };
            // Tiling planes l + mL strictly between x and xp on this axis
            let m_lo = ((lo_c - l) / size).floor() as i64 + 1;
            let m_hi = ((hi_c - l) / size).ceil() as i64 - 1;
            for m in m_lo..=m_hi {
                let plane = l + m as f64 * size;
                if !(plane > lo_c && plane < hi_c) {
                    continue;
                }
                let s = (plane - x[a]) / dir[a];
                let side = if self.is_periodic(a) {
                    // Direction never flips on a periodic axis
                    if dir[a] > 0.0 {
                        Side::Upper
                    } else {
                        Side::Lower
                    }
                } else if m.rem_euclid(2) == 0 {
                    Side::Lower
                } else {
                    Side::Upper
                };
                let mut point = self.fold(x + s * dir);
                // Snap exactly onto the face crossed
                point[a] = if side == Side::Upper { l + size } else { l };
                out.push(Crossing {
                    s,
                    axis: a,
                    side,
                    bc: self.bc(a, side),
                    point,
                });
            }
        }
        out[start..].sort_by(|p, q| p.s.total_cmp(&q.s));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    const LO: Vec3 = Vec3::new(-1.0, 0.0, 2.0);
    const HI: Vec3 = Vec3::new(2.0, 1.5, 3.0);

    fn reflective() -> BoundaryBox {
        BoundaryBox::uniform(LO, HI, Bc::Reflective).unwrap()
    }

    /// x reflective, y periodic, z: reflective below (symmetry plane), vacuum above
    fn mixed() -> BoundaryBox {
        BoundaryBox::new(
            LO,
            HI,
            [
                [Bc::Reflective, Bc::Reflective],
                [Bc::Periodic, Bc::Periodic],
                [Bc::Reflective, Bc::Vacuum],
            ],
        )
        .unwrap()
    }

    /// Small deterministic generator for property tests
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn point_in(&mut self, b: &BoundaryBox) -> Vec3 {
            b.lower() + Vec3::new(self.next(), self.next(), self.next()) * b.size()
        }
        fn t(&mut self, k: i32) -> [i32; 3] {
            let mut pick = || (self.next() * (2 * k + 1) as f64).floor() as i32 - k;
            [pick(), pick(), pick()]
        }
    }

    // ---------------- construction and basic queries ----------------

    #[test]
    fn construction_validates() {
        assert_eq!(
            BoundaryBox::uniform(LO, Vec3::new(2.0, 0.0, 3.0), Bc::Vacuum),
            Err(BoundaryError::EmptyAxis { axis: 1 })
        );
        assert_eq!(
            BoundaryBox::uniform(Vec3::splat(f64::NAN), HI, Bc::Vacuum),
            Err(BoundaryError::NonFinite)
        );
        let bad = [
            [Bc::Vacuum; 2],
            [Bc::Periodic, Bc::Reflective],
            [Bc::Vacuum; 2],
        ];
        assert_eq!(
            BoundaryBox::new(LO, HI, bad),
            Err(BoundaryError::UnpairedPeriodic { axis: 1 })
        );
        assert!(
            BoundaryError::UnpairedPeriodic { axis: 1 }
                .to_string()
                .contains("axis 1")
        );
    }

    #[test]
    fn sizes_contains_and_distance() {
        let b = reflective();
        assert_eq!(b.size(), Vec3::new(3.0, 1.5, 1.0));
        assert_close(b.volume(), 4.5);
        assert_eq!(b.upper(), HI);
        assert!(b.contains(LO) && b.contains(HI) && b.contains(Vec3::new(0.0, 1.0, 2.5)));
        assert!(!b.contains(Vec3::new(0.0, 1.6, 2.5)));
        assert_eq!(b.distance_to(Vec3::new(0.0, 1.0, 2.5)), 0.0);
        assert_close(b.distance_to(Vec3::new(5.0, 1.0, 2.5)), 3.0);
        assert_close(b.distance_to(Vec3::new(5.0, 5.5, 2.5)), 5.0); // 3-4-5 corner
        assert_eq!(mixed().bc(2, Side::Upper), Bc::Vacuum);
    }

    // ---------------- face_hit and cross ----------------

    #[test]
    fn face_hit_along_each_axis() {
        let b = reflective();
        let c = Vec3::new(0.5, 0.75, 2.5); // centre
        let h = b.face_hit(c, Vec3::X);
        assert_close(h.dist, 1.5);
        assert_eq!(h.faces, [Some(Side::Upper), None, None]);
        let h = b.face_hit(c, -Vec3::Z);
        assert_close(h.dist, 0.5);
        assert_eq!(h.iter().collect::<Vec<_>>(), vec![(2, Side::Lower)]);
    }

    #[test]
    fn face_hit_through_an_edge_and_a_corner() {
        let b = BoundaryBox::uniform(Vec3::ZERO, Vec3::ONE, Bc::Reflective).unwrap();
        let c = Vec3::splat(0.5);
        let h = b.face_hit(c, Vec3::new(1.0, 1.0, 0.0).normalize());
        assert_eq!(h.faces, [Some(Side::Upper), Some(Side::Upper), None]);
        let h = b.face_hit(c, -Vec3::ONE.normalize());
        assert_eq!(h.faces, [Some(Side::Lower); 3]);
        assert_close(h.dist, 0.5 * 3f64.sqrt());
    }

    #[test]
    fn face_hit_from_a_face_moving_inward_is_the_far_face() {
        let b = reflective();
        let h = b.face_hit(Vec3::new(-1.0, 0.75, 2.5), Vec3::X);
        assert_close(h.dist, 3.0);
    }

    #[test]
    fn cross_applies_each_condition() {
        let b = mixed();
        let p = Vec3::new(0.5, 0.75, 2.5);
        // Reflective x: direction component flips, point snapped to face
        let d = Vec3::new(0.6, 0.0, 0.8);
        let h = b.face_hit(Vec3::new(0.5, 0.75, 2.0), Vec3::X);
        let at = Vec3::new(2.0 - 1e-15, 0.75, 2.0);
        assert_eq!(
            b.cross(at, d, &h),
            Crossed::Continue {
                point: Vec3::new(2.0, 0.75, 2.0),
                dir: Vec3::new(-0.6, 0.0, 0.8)
            }
        );
        // Periodic y: jump to the opposite face, direction unchanged
        let h = b.face_hit(p, Vec3::Y);
        let at = Vec3::new(0.5, 1.5, 2.5);
        assert_eq!(
            b.cross(at, Vec3::Y, &h),
            Crossed::Continue {
                point: Vec3::new(0.5, 0.0, 2.5),
                dir: Vec3::Y
            }
        );
        // Vacuum upper z
        let h = b.face_hit(p, Vec3::Z);
        assert_eq!(
            b.cross(Vec3::new(0.5, 0.75, 3.0), Vec3::Z, &h),
            Crossed::Exit
        );
    }

    #[test]
    fn cross_at_a_corner_reflects_all_components() {
        let b = BoundaryBox::uniform(Vec3::ZERO, Vec3::ONE, Bc::Reflective).unwrap();
        let d = Vec3::ONE.normalize();
        let h = b.face_hit(Vec3::splat(0.5), d);
        assert_eq!(
            b.cross(Vec3::ONE, d, &h),
            Crossed::Continue {
                point: Vec3::ONE,
                dir: -d
            }
        );
        // A corner touching any vacuum face exits
        let v = BoundaryBox::new(
            Vec3::ZERO,
            Vec3::ONE,
            [
                [Bc::Reflective; 2],
                [Bc::Reflective; 2],
                [Bc::Reflective, Bc::Vacuum],
            ],
        )
        .unwrap();
        assert_eq!(v.cross(Vec3::ONE, d, &h), Crossed::Exit);
    }

    // ---------------- images ----------------

    #[test]
    fn image_examples_from_the_formula() {
        let b = reflective(); // x: l = -1, L = 3
        let p = Vec3::new(0.0, 0.5, 2.25); // u_x = 1
        // t_x = 1: reflection in the upper face x = 2 -> 2 + (2 - 0) = 4
        assert_close(b.image([1, 0, 0], p).x, 4.0);
        // t_x = -1: reflection in the lower face x = -1 -> -2
        assert_close(b.image([-1, 0, 0], p).x, -2.0);
        // t_x = ±2: translations by ±2L = ±6
        assert_close(b.image([2, 0, 0], p).x, 6.0);
        assert_close(b.image([-2, 0, 0], p).x, -6.0);
        // Periodic axis: plain translation
        assert_close(mixed().image([0, 3, 0], p).y, 0.5 + 4.5);
        assert_eq!(b.image([0, 0, 0], p), p);
    }

    #[test]
    fn transform_matches_image_and_is_orthogonal() {
        let mut rng = Lcg(1);
        for b in [reflective(), mixed()] {
            for _ in 0..200 {
                let (t, p) = (rng.t(4), rng.point_in(&b));
                let g = b.transform(t);
                assert_vec_close(g.apply(p), b.image(t, p));
                assert!(g.is_orthogonal(1e-12));
                assert_vec_close(g.inverse_orthogonal().apply(b.image(t, p)), p);
            }
        }
    }

    #[test]
    fn direction_maps_by_sign_flips() {
        let g = reflective().transform([1, -2, 3]);
        assert_eq!(
            g.apply_dir(Vec3::new(1.0, 2.0, 3.0)),
            Vec3::new(-1.0, 2.0, -3.0)
        );
    }

    #[test]
    fn fold_inverts_image() {
        let mut rng = Lcg(2);
        for b in [reflective(), mixed()] {
            for _ in 0..500 {
                let (t, p) = (rng.t(5), rng.point_in(&b));
                let y = b.image(t, p);
                // Folding introduces a few roundings per tile crossed
                assert_vec_close_tol(b.fold(y), p, Tol::new(1e-12, 1e-12));
            }
        }
    }

    #[test]
    fn fold_keeps_points_in_the_box() {
        let b = mixed();
        let mut rng = Lcg(3);
        for _ in 0..500 {
            let y = Vec3::new(
                20.0 * rng.next() - 10.0,
                20.0 * rng.next() - 10.0,
                20.0 * rng.next() - 10.0,
            );
            let f = b.fold(y);
            assert!(b.distance_to(f) <= 1e-12, "{f:?}");
        }
    }

    #[test]
    fn admissibility_follows_the_faces_crossed() {
        let b = mixed();
        // z: lower reflective, upper vacuum -> only 0 and -1
        let z_ok: Vec<i32> = (-3..=3).filter(|&t| b.admissible([0, 0, t])).collect();
        assert_eq!(z_ok, vec![-1, 0]);
        // x: both reflective -> everything; y: periodic -> everything
        assert!((-3..=3).all(|t| b.admissible([t, t, 0])));
        // A fully vacuum box admits only the identity
        let v = BoundaryBox::uniform(LO, HI, Bc::Vacuum).unwrap();
        assert!(v.admissible([0, 0, 0]));
        assert!(!v.admissible([1, 0, 0]) && !v.admissible([0, -1, 0]));
        // Reflective upper, vacuum lower: only 0 and +1
        let up = BoundaryBox::new(
            LO,
            HI,
            [
                [Bc::Vacuum, Bc::Reflective],
                [Bc::Vacuum; 2],
                [Bc::Vacuum; 2],
            ],
        )
        .unwrap();
        let ok: Vec<i32> = (-3..=3).filter(|&t| up.admissible([t, 0, 0])).collect();
        assert_eq!(ok, vec![0, 1]);
    }

    #[test]
    fn images_within_matches_brute_force() {
        let mut rng = Lcg(4);
        let mut found = Vec::new();
        for b in [reflective(), mixed()] {
            for _ in 0..50 {
                let p = rng.point_in(&b);
                let r = 4.0 * rng.next();
                found.clear();
                b.images_within(p, r, &mut found);
                // No duplicates
                let mut sorted = found.clone();
                sorted.sort();
                sorted.dedup();
                assert_eq!(sorted.len(), found.len());
                // Brute force over a large cube of t
                let mut expected = Vec::new();
                for tx in -8..=8 {
                    for ty in -8..=8 {
                        for tz in -8..=8 {
                            let t = [tx, ty, tz];
                            if b.admissible(t) && b.distance_to(b.image(t, p)) < r {
                                expected.push(t);
                            }
                        }
                    }
                }
                expected.sort();
                assert_eq!(sorted, expected, "p = {p:?}, r = {r}");
                assert!(found.contains(&[0, 0, 0]));
            }
        }
    }

    #[test]
    fn images_within_respects_vacuum_faces() {
        // Fully vacuum box: only the identity, however large r
        let v = BoundaryBox::uniform(LO, HI, Bc::Vacuum).unwrap();
        let mut found = Vec::new();
        v.images_within(Vec3::new(0.0, 0.5, 2.5), 100.0, &mut found);
        assert_eq!(found, vec![[0, 0, 0]]);
    }

    // ---------------- crossings (reflection points) ----------------

    /// Path through the folded crossings from x to the target fold(xp)
    fn folded_length(x: Vec3, target: Vec3, crossings: &[Crossing]) -> f64 {
        let mut pts = vec![x];
        pts.extend(crossings.iter().map(|c| c.point));
        pts.push(target);
        pts.windows(2).map(|w| (w[1] - w[0]).length()).sum()
    }

    #[test]
    fn single_reflection_example() {
        // Unit box, x = (0.25, 0.5, 0.5) to the image of y = (0.75, 0.5, 0.5)
        // in the upper x face (t = [1,0,0]): x' = 1.25, path 0.25 -> 1 -> 0.75
        let b = BoundaryBox::uniform(Vec3::ZERO, Vec3::ONE, Bc::Reflective).unwrap();
        let (x, y) = (Vec3::new(0.25, 0.5, 0.5), Vec3::new(0.75, 0.5, 0.5));
        let xp = b.image([1, 0, 0], y);
        let mut c = Vec::new();
        b.crossings(x, xp, &mut c);
        assert_eq!(c.len(), 1);
        assert_eq!(
            (c[0].axis, c[0].side, c[0].bc),
            (0, Side::Upper, Bc::Reflective)
        );
        assert_close(c[0].s, 0.75);
        assert_vec_close(c[0].point, Vec3::new(1.0, 0.5, 0.5));
        assert_close(folded_length(x, y, &c), 1.0);
    }

    #[test]
    fn corner_image_reflects_once_on_each_wall() {
        // 2D corner: image across both x = 0 and y = 0 (t = [-1, -1, 0])
        let b = BoundaryBox::uniform(Vec3::ZERO, Vec3::ONE, Bc::Reflective).unwrap();
        let (x, y) = (Vec3::new(0.2, 0.6, 0.5), Vec3::new(0.7, 0.3, 0.5));
        let xp = b.image([-1, -1, 0], y);
        let mut c = Vec::new();
        b.crossings(x, xp, &mut c);
        assert_eq!(c.len(), 2);
        let axes: Vec<_> = c.iter().map(|k| (k.axis, k.side)).collect();
        assert!(axes.contains(&(0, Side::Lower)) && axes.contains(&(1, Side::Lower)));
        assert!(c[0].s <= c[1].s);
        assert_close(folded_length(x, y, &c), (xp - x).length());
    }

    #[test]
    fn crossings_reproduce_reflected_paths() {
        let b = reflective();
        let mut rng = Lcg(5);
        let mut c = Vec::new();
        for _ in 0..500 {
            let (x, y, t) = (rng.point_in(&b), rng.point_in(&b), rng.t(3));
            let xp = b.image(t, y);
            c.clear();
            b.crossings(x, xp, &mut c);
            let n: i32 = t.iter().map(|v| v.abs()).sum();
            assert_eq!(c.len() as i32, n, "t = {t:?}");
            for (k, cr) in c.iter().enumerate() {
                // Each point lies on the face it reports, inside the box
                let face = if cr.side == Side::Upper {
                    b.upper()
                } else {
                    b.lower()
                };
                assert_close(cr.point[cr.axis], face[cr.axis]);
                assert!(b.distance_to(cr.point) <= 1e-12);
                assert!(cr.s >= 0.0 && cr.s <= (xp - x).length());
                if k > 0 {
                    assert!(cr.s >= c[k - 1].s);
                }
            }
            // The broken path through the reflection points has the unfolded length
            assert_close_tol(
                folded_length(x, y, &c),
                (xp - x).length(),
                Tol::new(1e-11, 1e-11),
            );
        }
    }

    #[test]
    fn crossings_report_periodic_jumps_and_vacuum_faces() {
        let b = mixed();
        let x = Vec3::new(0.5, 0.2, 2.5);
        let mut c = Vec::new();
        // Two periodic jumps through the upper y face
        b.crossings(x, b.image([0, 2, 0], x), &mut c);
        assert_eq!(c.len(), 2);
        assert!(
            c.iter()
                .all(|k| k.axis == 1 && k.side == Side::Upper && k.bc == Bc::Periodic)
        );
        // An inadmissible image crosses the vacuum face, which is reported
        c.clear();
        let t = [0, 0, 1];
        assert!(!b.admissible(t));
        b.crossings(x, b.image(t, x), &mut c);
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].side, c[0].bc), (Side::Upper, Bc::Vacuum));
    }

    #[test]
    fn coincident_points_have_no_crossings() {
        let mut c = Vec::new();
        let p = Vec3::new(0.0, 0.5, 2.5);
        reflective().crossings(p, p, &mut c);
        assert!(c.is_empty());
    }
}
