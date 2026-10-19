//! Conformance suite: checks every [`Walker`] implementation must pass.
//!
//! Each check panics (with `#[track_caller]`) on failure, so backends call
//! them from their own tests, typically through [`run_suite`]:
//!
//! ```ignore
//! #[test]
//! fn model_conforms() {
//!     mrr_geom3d::conformance::run_suite(&my_model(), 200, 1);
//! }
//! ```
//!
//! What is checked:
//! - [`check_walk`]: one walk is self-consistent (length, coverage, labels
//!   agree with `material_at`, legs stay in the box, `Ended` recorded).
//! - [`check_reversible`]: a walk without reflections, reversed, gives the same
//!   intervals in reverse order.
//! - [`check_image`]: walking toward an image of a node reproduces the method
//!   of images exactly: it reaches the node, reflects at the points
//!   `BoundaryBox::crossings` predicts, arrives with the flipped direction, or
//!   exits at the first vacuum face for an inadmissible image.

use crate::testing::*;
use crate::{Ended, MaterialId, Partition, Path, Vec3, Walker, new_partition};

/// Positions and lengths along a path agree to this (cm): far above the
/// rounding of a few reflections and crossings, far below anything physical
pub const PATH_TOL: f64 = 1e-9;

/// Small deterministic generator, so suites are reproducible
#[derive(Debug, Clone)]
pub struct Lcg(pub u64);

impl Lcg {
    /// Uniform in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform direction on the unit sphere
    pub fn direction(&mut self) -> Vec3 {
        let mu = 2.0 * self.next_f64() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * self.next_f64();
        let s = (1.0 - mu * mu).sqrt();
        Vec3::new(s * phi.cos(), s * phi.sin(), mu)
    }

    /// Uniform point in the box `[lower, lower + size]`
    pub fn point_in(&mut self, lower: Vec3, size: Vec3) -> Vec3 {
        lower + Vec3::new(self.next_f64(), self.next_f64(), self.next_f64()) * size
    }

    /// Integer in `[-k, k]`
    pub fn int(&mut self, k: i32) -> i32 {
        (self.next_f64() * (2 * k + 1) as f64).floor() as i32 - k
    }
}

/// Walk and check the result is self-consistent. Returns the path and the
/// material partition for further checks.
#[track_caller]
pub fn check_walk<W: Walker>(
    w: &W,
    origin: Vec3,
    dir: Vec3,
    s_max: f64,
) -> (Path, Partition<MaterialId>) {
    let mut path = Path::new(origin, dir);
    let mut mats = new_partition();
    let ended = w.walk(origin, dir, s_max, &mut path, &mut mats);
    let ctx = format!("walk from {origin:?} along {dir:?} for {s_max}");

    // The end state is recorded and the length matches it
    assert_eq!(
        path.ended(),
        Some(ended),
        "{ctx}: Ended not recorded in path"
    );
    let expected = match ended {
        Ended::Reached => s_max,
        Ended::VacuumExit { s } => {
            assert!(s <= s_max + PATH_TOL, "{ctx}: exit at s = {s} beyond s_max");
            s
        }
    };
    assert!(
        (path.length() - expected).abs() <= PATH_TOL,
        "{ctx}: path length {} but expected {expected} ({ended:?})",
        path.length()
    );

    // The material partition covers exactly the path
    assert_eq!(mats.start(), 0.0, "{ctx}: partition does not start at 0");
    if path.is_empty() {
        assert!(mats.is_empty(), "{ctx}: intervals on an empty path");
    } else {
        assert!(
            (mats.end() - path.s_end()).abs() <= PATH_TOL,
            "{ctx}: partition ends at {} but the path at {}",
            mats.end(),
            path.s_end()
        );
    }

    // Labels agree with point location at each interval's midpoint
    for span in mats.iter() {
        let mid = span.midpoint();
        let k = path.leg_index(mid).expect("midpoint is on the path");
        let leg = path.legs()[k];
        let point = leg.point_at(mid);
        assert_eq!(
            w.material_at(point, leg.dir),
            Some(span.label),
            "{ctx}: interval {span:?} disagrees with material_at({point:?})"
        );
    }

    // Legs stay in the box and are unit-speed
    let b = w.boundary();
    for leg in path.legs() {
        assert!(
            (leg.dir.length() - 1.0).abs() <= 1e-12,
            "{ctx}: non-unit leg {leg:?}"
        );
        for p in [leg.origin, leg.point_at(leg.s_end)] {
            assert!(
                b.distance_to(p) <= PATH_TOL,
                "{ctx}: leg leaves the box at {p:?}"
            );
        }
    }
    (path, mats)
}

/// A walk without reflections from `a` to `b`, reversed, gives the same
/// intervals in reverse order. Skipped (returns `false`) if the walk reflects
/// or exits.
#[track_caller]
pub fn check_reversible<W: Walker>(w: &W, a: Vec3, b: Vec3) -> bool {
    let len = (b - a).length();
    if len < 1e-6 {
        return false;
    }
    let dir = (b - a) / len;
    let (pf, mf) = check_walk(w, a, dir, len);
    let (pb, mb) = check_walk(w, b, -dir, len);
    let straight = |p: &Path| p.legs().len() == 1 && p.ended() == Some(Ended::Reached);
    if !(straight(&pf) && straight(&pb)) {
        return false;
    }
    let fwd: Vec<_> = mf.iter().collect();
    let bwd: Vec<_> = mb.iter().collect();
    assert_eq!(
        fwd.len(),
        bwd.len(),
        "reversed walk {a:?} <-> {b:?} has a different interval count"
    );
    for (f, r) in fwd.iter().zip(bwd.iter().rev()) {
        assert_eq!(
            f.label, r.label,
            "reversed walk {a:?} <-> {b:?}: labels differ"
        );
        assert!(
            (f.length() - r.length()).abs() <= PATH_TOL,
            "reversed walk {a:?} <-> {b:?}: interval lengths {} vs {}",
            f.length(),
            r.length()
        );
    }
    true
}

/// Walking from `x` toward image `t` of `y` reproduces the method of images.
#[track_caller]
pub fn check_image<W: Walker>(w: &W, x: Vec3, y: Vec3, t: [i32; 3]) {
    let b = w.boundary();
    let xp = b.image(t, y);
    let len = (xp - x).length();
    if len < 1e-6 {
        return;
    }
    let dir = (xp - x) / len;
    let (path, _) = check_walk(w, x, dir, len);
    let mut crossings = Vec::new();
    b.crossings(x, xp, &mut crossings);
    let ctx = format!("image t = {t:?} of {y:?} from {x:?}");

    if !b.admissible(t) {
        let first_vacuum = crossings
            .iter()
            .find(|c| c.bc == crate::Bc::Vacuum)
            .expect("an inadmissible image crosses a vacuum face");
        match path.ended() {
            Some(Ended::VacuumExit { s }) => assert!(
                (s - first_vacuum.s).abs() <= PATH_TOL,
                "{ctx}: exited at s = {s}, expected {}",
                first_vacuum.s
            ),
            other => panic!("{ctx}: inadmissible image should exit, got {other:?}"),
        }
        return;
    }

    assert_eq!(
        path.ended(),
        Some(Ended::Reached),
        "{ctx}: did not reach the node"
    );
    assert_vec_close_tol(path.end_point(), y, Tol::new(0.0, PATH_TOL));

    // One leg boundary per distinct crossing event (edges/corners count once)
    let mut events: Vec<&crate::Crossing> = Vec::new();
    for c in &crossings {
        if events.last().is_none_or(|e| c.s - e.s > PATH_TOL) {
            events.push(c);
        }
    }
    assert_eq!(
        path.legs().len(),
        events.len() + 1,
        "{ctx}: {} legs for {} boundary events",
        path.legs().len(),
        events.len()
    );
    for (leg, event) in path.legs().iter().zip(&events) {
        assert!(
            (leg.s_end - event.s).abs() <= PATH_TOL,
            "{ctx}: leg ends at s = {}, crossing at {}",
            leg.s_end,
            event.s
        );
        assert_vec_close_tol(
            leg.point_at(leg.s_end),
            event.point,
            Tol::new(0.0, PATH_TOL),
        );
    }

    // Arrival direction: the unfolded direction mapped by the image's sign flips
    let arrival = b.transform(t).apply_dir(dir);
    assert_vec_close_tol(path.last_dir(), arrival, Tol::new(0.0, 1e-12));
}

/// Random walks, reversals and images. `n` sets the number of each.
#[track_caller]
pub fn run_suite<W: Walker>(w: &W, n: usize, seed: u64) {
    let mut rng = Lcg(seed);
    let b = w.boundary();
    let (lo, size) = (b.lower(), b.size());
    let diag = size.length();
    for _ in 0..n {
        // Arbitrary walks, long enough to reflect several times
        let o = rng.point_in(lo, size);
        let d = rng.direction();
        let s_max = 3.0 * diag * rng.next_f64();
        check_walk(w, o, d, s_max);

        // Reversal between two interior points
        let (a, c) = (rng.point_in(lo, size), rng.point_in(lo, size));
        check_reversible(w, a, c);

        // Images up to two copies away on each axis
        let (x, y) = (rng.point_in(lo, size), rng.point_in(lo, size));
        let t = [rng.int(2), rng.int(2), rng.int(2)];
        check_image(w, x, y, t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_uniform_enough() {
        let mut r = Lcg(1);
        let mean: f64 = (0..10_000).map(|_| r.next_f64()).sum::<f64>() / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02);
        for _ in 0..100 {
            assert!((r.direction().length() - 1.0).abs() < 1e-12);
            let k = r.int(2);
            assert!((-2..=2).contains(&k));
        }
    }
}
