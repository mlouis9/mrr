//! The route of one walk through the geometry.
//!
//! A [`Path`] is a sequence of straight [`Leg`]s parametrized by the total path
//! length `s` from the start. A new leg begins wherever the walk changes
//! direction or jumps (reflection, periodic translation). Only the walker
//! builds paths; every layer segments a finished path one leg at a time.
//!
//! Positions are continuous across a reflection but not across a periodic
//! translation, so only continuity in `s` is enforced between legs.

use crate::{MaterialId, Partition, Segment, Vec3, tol};

/// The route taken by one walk: where and in which direction it started,
/// straight legs contiguous in `s`, and how the walk ended.
///
/// A path always has a start, set by [`begin`](Path::begin), so every summary
/// (`s_end`, `length`, `last_dir`, `end_point`) is defined. A walk of zero
/// length (coincident points, or a vacuum exit at the start) is simply a
/// path with no legs, whose end equals its start. Reuse one `Path` across
/// many walks: `begin` keeps the allocation.
#[derive(Debug, Clone)]
pub struct Path {
    origin: Vec3,
    dir: Vec3,
    s0: f64,
    legs: Vec<Leg>,
    ended: Option<Ended>,
}

impl Path {
    /// A path starting at `origin` heading along `dir` (a unit vector), at
    /// `s = 0`, with no legs yet
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        let mut path = Self {
            origin,
            dir,
            s0: 0.0,
            legs: Vec::new(),
            ended: None,
        };
        path.begin(origin, dir);
        path
    }

    /// Start a new walk from `origin` along `dir` at `s = 0`, discarding any
    /// previous legs and end state but keeping the allocation
    pub fn begin(&mut self, origin: Vec3, dir: Vec3) {
        debug_assert!(
            (dir.length() - 1.0).abs() <= 1e-12,
            "path direction is not a unit vector: {dir:?}"
        );
        self.origin = origin;
        self.dir = dir;
        self.s0 = 0.0;
        self.legs.clear();
        self.ended = None;
    }

    /// Append a leg. In debug builds checks that it has a unit direction and
    /// non-negative length, that it starts where the path currently ends (in
    /// `s` to within `tol::LEN`; the first leg also at the path's origin), and
    /// that the path is not finished.
    pub fn push_leg(&mut self, leg: Leg) {
        debug_assert!(self.ended.is_none(), "push_leg on a finished path");
        debug_assert!(
            (leg.dir.length() - 1.0).abs() <= 1e-12,
            "leg direction is not a unit vector: {:?}",
            leg.dir
        );
        debug_assert!(
            leg.s_end >= leg.s_start,
            "leg has negative length: [{}, {}]",
            leg.s_start,
            leg.s_end
        );
        debug_assert!(
            (leg.s_start - self.s_end()).abs() <= tol::LEN,
            "leg starts at s = {} but the path ends at s = {}",
            leg.s_start,
            self.s_end()
        );
        debug_assert!(
            !self.legs.is_empty() || (leg.origin - self.origin).length() <= tol::LEN,
            "first leg starts at {:?}, not at the path origin {:?}",
            leg.origin,
            self.origin
        );
        self.legs.push(leg);
    }

    /// Record how the walk ended. Called once, by the walker.
    pub fn finish(&mut self, ended: Ended) {
        debug_assert!(self.ended.is_none(), "path finished twice");
        self.ended = Some(ended);
    }

    pub fn legs(&self) -> &[Leg] {
        &self.legs
    }

    /// How the walk ended, or `None` while it is still being built
    pub fn ended(&self) -> Option<Ended> {
        self.ended
    }

    /// `true` if no legs have been pushed (a zero-length walk)
    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    /// `s` at the start (always 0 for paths built with `begin`)
    pub fn s_start(&self) -> f64 {
        self.s0
    }

    /// `s` at the end of the last leg, or the start if there are none
    pub fn s_end(&self) -> f64 {
        self.legs.last().map_or(self.s0, |l| l.s_end)
    }

    /// Total length walked
    pub fn length(&self) -> f64 {
        self.s_end() - self.s0
    }

    /// Direction at the start: the emission direction
    pub fn first_dir(&self) -> Vec3 {
        self.dir
    }

    /// Direction at the end: the arrival direction (the start direction if
    /// there are no legs)
    pub fn last_dir(&self) -> Vec3 {
        self.legs.last().map_or(self.dir, |l| l.dir)
    }

    pub fn start_point(&self) -> Vec3 {
        self.origin
    }

    /// End of the last leg, or the start if there are none
    pub fn end_point(&self) -> Vec3 {
        self.legs
            .last()
            .map_or(self.origin, |l| l.point_at(l.s_end))
    }

    /// Index of the leg containing `s`. Legs are half-open `[s_start, s_end)`
    /// (at a reflection point the later leg wins, as in the tie policy), except
    /// that the path's final `s_end` belongs to the last leg. `None` outside
    /// the path, for NaN, or if there are no legs.
    pub fn leg_index(&self, s: f64) -> Option<usize> {
        if self.legs.is_empty() || !(s >= self.s0 && s <= self.s_end()) {
            return None;
        }
        // Number of legs that end at or before s, capped at the last leg
        let k = self.legs.partition_point(|l| l.s_end <= s);
        Some(k.min(self.legs.len() - 1))
    }

    /// Point at path length `s`, or `None` outside the path (a path with no
    /// legs has only its start point, at `s_start`)
    pub fn point_at(&self, s: f64) -> Option<Vec3> {
        if self.legs.is_empty() {
            return (s == self.s0).then_some(self.origin);
        }
        self.leg_index(s).map(|k| self.legs[k].point_at(s))
    }
}

/// One straight piece of a path, covering `s ∈ [s_start, s_end]`.
/// `origin` is the point at `s_start`; `dir` is a unit vector.
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

/// How a walk ended
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ended {
    /// Left the domain through a vacuum face at path length `s`
    VacuumExit { s: f64 },
    /// Walked the full requested length
    Reached,
}

/// Convert a material partition into `(material, length)` segments, for
/// `mrr_xs::optical_depths`. `out` is cleared first.
pub fn material_segments(materials: &Partition<MaterialId>, out: &mut Vec<Segment>) {
    out.clear();
    out.extend(materials.iter().map(|span| Segment {
        material: span.label,
        length: span.length(),
    }));
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

    /// A path from (0,0,0) up +z for 2, reflected at z = 2 back down for 1.5
    fn bounce() -> Path {
        let mut p = Path::new(Vec3::ZERO, Vec3::Z);
        p.push_leg(Leg {
            origin: Vec3::ZERO,
            dir: Vec3::Z,
            s_start: 0.0,
            s_end: 2.0,
        });
        p.push_leg(Leg {
            origin: Vec3::new(0.0, 0.0, 2.0),
            dir: -Vec3::Z,
            s_start: 2.0,
            s_end: 3.5,
        });
        p.finish(Ended::Reached);
        p
    }

    #[test]
    fn path_without_legs_ends_where_it_starts() {
        let start = Vec3::new(1.0, 2.0, 3.0);
        let p = Path::new(start, Vec3::Y);
        assert!(p.is_empty());
        assert_eq!(p.length(), 0.0);
        assert_eq!((p.s_start(), p.s_end()), (0.0, 0.0));
        assert_eq!((p.first_dir(), p.last_dir()), (Vec3::Y, Vec3::Y));
        assert_eq!((p.start_point(), p.end_point()), (start, start));
        assert_eq!(p.point_at(0.0), Some(start));
        assert_eq!(p.point_at(0.1), None);
        assert_eq!(p.leg_index(0.0), None);
        assert_eq!(p.ended(), None);
    }

    #[test]
    fn two_leg_path_summary() {
        let p = bounce();
        assert_eq!(p.legs().len(), 2);
        assert_close(p.length(), 3.5);
        assert_eq!(p.first_dir(), Vec3::Z);
        assert_eq!(p.last_dir(), -Vec3::Z);
        assert_eq!(p.start_point(), Vec3::ZERO);
        assert_vec_close(p.end_point(), Vec3::new(0.0, 0.0, 0.5));
        assert_eq!(p.ended(), Some(Ended::Reached));
    }

    #[test]
    fn point_at_picks_the_right_leg() {
        let p = bounce();
        assert_vec_close(p.point_at(1.0).unwrap(), Vec3::new(0.0, 0.0, 1.0));
        assert_vec_close(p.point_at(3.0).unwrap(), Vec3::new(0.0, 0.0, 1.0));
        // The reflection point belongs to the later leg; both agree on position
        assert_eq!(p.leg_index(2.0), Some(1));
        assert_vec_close(p.point_at(2.0).unwrap(), Vec3::new(0.0, 0.0, 2.0));
        // Both ends are inside; beyond them is not
        assert_eq!(p.leg_index(0.0), Some(0));
        assert_eq!(p.leg_index(3.5), Some(1));
        assert_eq!(p.point_at(3.6), None);
        assert_eq!(p.point_at(-0.1), None);
        assert_eq!(p.point_at(f64::NAN), None);
    }

    #[test]
    fn begin_resets_for_reuse() {
        let mut p = bounce();
        p.begin(leg().origin, leg().dir);
        assert!(p.is_empty());
        assert_eq!(p.ended(), None);
        assert_eq!(p.first_dir(), leg().dir);
        p.push_leg(Leg {
            s_start: 0.0,
            s_end: 5.0,
            ..leg()
        });
        assert_close(p.length(), 5.0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "but the path ends at")]
    fn discontinuous_legs_panic_in_debug() {
        let mut p = Path::new(leg().origin, leg().dir);
        p.push_leg(Leg {
            s_start: 0.0,
            s_end: 5.0,
            ..leg()
        });
        p.push_leg(Leg {
            s_start: 6.0,
            s_end: 7.0,
            ..leg()
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "not at the path origin")]
    fn first_leg_must_start_at_origin_in_debug() {
        let mut p = Path::new(Vec3::ZERO, Vec3::Z);
        p.push_leg(Leg {
            s_start: 0.0,
            s_end: 1.0,
            ..leg()
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "not a unit vector")]
    fn non_unit_direction_panics_in_debug() {
        let mut p = Path::new(Vec3::ZERO, Vec3::Z);
        p.push_leg(Leg {
            origin: Vec3::ZERO,
            dir: Vec3::new(0.0, 0.0, 2.0),
            s_start: 0.0,
            s_end: 1.0,
        });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "finished path")]
    fn pushing_after_finish_panics_in_debug() {
        let mut p = bounce();
        p.push_leg(leg());
    }

    #[test]
    fn material_segments_are_interval_lengths() {
        let mut p = Partition::new(0.0, tol::LEN);
        p.push(1.5, MaterialId(0));
        p.push(2.0, MaterialId(1));
        let mut segs = vec![Segment {
            material: MaterialId(9),
            length: 9.0,
        }];
        material_segments(&p, &mut segs);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].material, MaterialId(1));
        assert_close(segs[0].length, 1.5);
        assert_close(segs[1].length, 0.5);
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
