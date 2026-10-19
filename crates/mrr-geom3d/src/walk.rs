//! The necessary types for expressing a walk in 3D geometry

use crate::{Vec3, tol};

/// An entire traced path, including boundary reflections, represented as separate legs
#[derive(Debug, Clone, Default)]
pub struct Path {
    legs: Vec<Leg>,
    ended: Option<Ended>,
}

/// A portion of a path up to a boundary interaction
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Leg {
    pub origin: Vec3,
    pub dir: Vec3,
    pub s_start: f64,
    pub s_end: f64,
}

impl Leg {
    /// Length of the leg, `s_end - s_start`
    pub fn length(&self) -> f64 {
        self.s_end - self.s_start
    }

    /// Point at path length `s` along this leg: `origin + (s - s_start) dir`.
    ///
    /// `s` must lie in `[s_start, s_end]`, up to `tol::LEN` of slack at each
    /// end because path lengths are computed values (checked in debug builds).
    pub fn point_at(&self, s: f64) -> Vec3 {
        let (start, end) = (self.s_start, self.s_end);
        debug_assert!(
            s >= start - tol::LEN && s <= end + tol::LEN,
            "s = {s} not in leg [{start}, {end}]"
        );
        self.origin + (s - start) * self.dir
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ended {
    VacuumExit { s: f64 },
    Reached,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    fn leg() -> Leg {
        Leg {
            origin: Vec3::new(1.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.6, 0.8),
            s_start: 2.0,
            s_end: 7.0,
        }
    }

    #[test]
    fn length_is_s_range() {
        assert_close(leg().length(), 5.0);
    }

    #[test]
    fn point_at_hits_both_ends_and_interior() {
        let l = leg();
        assert_vec_close(l.point_at(2.0), l.origin);
        assert_vec_close(l.point_at(7.0), Vec3::new(1.0, 3.0, 4.0));
        assert_vec_close(l.point_at(4.5), Vec3::new(1.0, 1.5, 2.0));
    }

    #[test]
    fn point_at_allows_rounding_slack_at_the_ends() {
        let l = leg();
        let _ = l.point_at(2.0 - 0.5 * tol::LEN);
        let _ = l.point_at(7.0 + 0.5 * tol::LEN);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "not in leg")]
    fn point_outside_leg_panics_in_debug() {
        leg().point_at(7.1);
    }
}
