//! `Walker` for the CSG model: trace a straight path through the cells,
//! reflecting / wrapping / stopping at the outer box.
//!
//! # The loop
//!
//! From the current point `p` in cell `c`, moving along `d` with `s` walked:
//!
//! 1. Candidate events: the nearest crossing of one of `c`'s surfaces, the
//!    nearest box face (`BoundaryBox::face_hit`) and the remaining budget
//!    `s_max - s`.
//! 2. Advance to the nearest, recording `c`'s material up to it.
//! 3. Resolve, in this order (so ties are deterministic):
//!    - **budget**: the walk is complete, even if it ends on a face;
//!    - **box face** (also when a surface is hit at the same point): close the
//!      leg and apply the boundary condition (`BoundaryBox::cross`); vacuum
//!      ends the walk, reflective/periodic start a new leg;
//!    - **surface**: nothing to do but re-locate.
//! 4. Re-locate the cell at `p` with the (possibly new) direction. The
//!    direction-aware tie rule puts a point on a surface in the cell it is
//!    heading into, so `p` and `s` stay exact: no bump is applied to them.
//!
//! Positions are recomputed from the start of the current leg
//! (`leg.origin + (s - leg.s_start) d`) rather than accumulated step by step,
//! so rounding does not grow with the number of internal crossings.
//!
//! # Robustness
//!
//! - A cell is left through one of its own surfaces (cells are intersections
//!   of half-spaces, hence convex), so only those are candidates.
//! - If re-location at `p` fails (a grazing crossing where `p` is off the
//!   surface by more than `tol::LEN`), it is retried a `tol::BUMP` ahead.
//! - Zero-progress steps are counted; after `MAX_STALLS` in a row the walk is
//!   forced `BUMP` forward. A walk can therefore not loop forever.
//! - A point inside the box but in no cell is a modelling error and panics
//!   with its location.

use mrr_geom3d::{
    BoundaryBox, Crossed, Ended, Geometry, Leg, MaterialId, Partition, Path, Vec3, Walker, tol,
};

use crate::{CellId, Model};

/// Consecutive zero-length steps tolerated before forcing progress
const MAX_STALLS: usize = 4;

/// The core geometry view of the model: the domain is the boundary box,
/// sampled uniformly (every sample lands inside, so none is rejected).
impl Geometry for Model {
    type Point = Vec3;
    const SAMPLE_DIM: usize = 3;

    fn volume(&self) -> f64 {
        Model::boundary(self).volume()
    }

    fn map_uniform(&self, u: &[f64]) -> Option<Vec3> {
        let b = Model::boundary(self);
        Some(b.lower() + Vec3::new(u[0], u[1], u[2]) * b.size())
    }
}

impl Walker for Model {
    fn boundary(&self) -> &BoundaryBox {
        Model::boundary(self)
    }

    fn material_at(&self, p: Vec3, dir: Vec3) -> Option<MaterialId> {
        Model::material_at(self, p, dir)
    }

    fn walk(
        &self,
        origin: Vec3,
        dir: Vec3,
        s_max: f64,
        path: &mut Path,
        materials: &mut Partition<MaterialId>,
    ) -> Ended {
        path.begin(origin, dir);
        materials.reset(0.0);
        let ended = self.trace(origin, dir, s_max, path, materials);
        path.finish(ended);
        ended
    }
}

impl Model {
    /// Nearest crossing of one of `cell`'s surfaces ahead of `p` (infinite if none)
    fn nearest_surface(&self, cell: CellId, p: Vec3, d: Vec3) -> f64 {
        self.cell(cell)
            .region()
            .iter()
            .filter_map(|h| self.surface(h.surface).distance(p, d))
            .fold(f64::INFINITY, f64::min)
    }

    /// Cell at `p` heading along `d`, retrying a bump ahead for grazing cases
    fn relocate(&self, p: Vec3, d: Vec3) -> CellId {
        self.find_cell(p, d)
            .or_else(|| self.find_cell(p + tol::BUMP * d, d))
            .unwrap_or_else(|| panic!("no cell at {p:?} (heading {d:?}): gap in the model"))
    }

    fn trace(
        &self,
        origin: Vec3,
        dir: Vec3,
        s_max: f64,
        path: &mut Path,
        materials: &mut Partition<MaterialId>,
    ) -> Ended {
        let bbox = Model::boundary(self);
        debug_assert!(
            bbox.distance_to(origin) <= tol::LEN,
            "walk starts outside the boundary box at {origin:?}"
        );
        if s_max <= 0.0 {
            return Ended::Reached;
        }
        if !bbox.contains(origin) {
            return Ended::VacuumExit { s: 0.0 };
        }

        let mut d = dir;
        let mut s = 0.0;
        // Start of the current leg: positions are recomputed from it
        let (mut leg_origin, mut leg_s0) = (origin, 0.0);
        let mut p = origin;
        let mut cell = self.relocate(p, d);
        let mut stalls = 0;

        loop {
            let budget = s_max - s;
            let d_surf = self.nearest_surface(cell, p, d);
            let hit = bbox.face_hit(p, d);
            let mut step = d_surf.min(hit.dist).min(budget).max(0.0);

            // Never spin in place: force progress after repeated empty steps
            if step < tol::LEN {
                stalls += 1;
                if stalls > MAX_STALLS {
                    step = tol::BUMP.min(budget);
                    stalls = 0;
                }
            } else {
                stalls = 0;
            }

            materials.push(s + step, self.cell(cell).material());
            s += step;
            p = leg_origin + (s - leg_s0) * d;

            // 1. Budget used up: done (takes precedence over a face here)
            if step >= budget {
                close_leg(path, leg_origin, d, leg_s0, s);
                return Ended::Reached;
            }

            // 2. Box face reached (possibly together with a surface)
            if hit.dist <= step + tol::LEN {
                close_leg(path, leg_origin, d, leg_s0, s);
                match bbox.cross(p, d, &hit) {
                    Crossed::Exit => return Ended::VacuumExit { s },
                    Crossed::Continue { point, dir } => {
                        p = point;
                        d = dir;
                        (leg_origin, leg_s0) = (p, s);
                    }
                }
            }

            // 3. Surface crossed, or new direction: locate the next cell
            cell = self.relocate(p, d);
        }
    }
}

/// Append the leg `[s0, s]` starting at `origin` along `dir` (empty legs, e.g.
/// a reflection exactly at the start, are skipped)
fn close_leg(path: &mut Path, origin: Vec3, dir: Vec3, s0: f64, s: f64) {
    if s > s0 {
        path.push_leg(Leg {
            origin,
            dir,
            s_start: s0,
            s_end: s,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Axis, Surface};
    use mrr_geom3d::conformance::{check_image, check_walk, run_suite};
    use mrr_geom3d::testing::*;
    use mrr_geom3d::{Bc, BoundaryBox, Segment, material_segments, new_partition};

    const FUEL: MaterialId = MaterialId(0);
    const WATER: MaterialId = MaterialId(1);
    const R: f64 = 0.41;
    const HALF_PITCH: f64 = 0.63;

    /// Pin cell: fuel cylinder along z in water, box with the given conditions
    fn pin_cell(bc: Bc) -> Model {
        let h = HALF_PITCH;
        let bbox = BoundaryBox::uniform(Vec3::new(-h, -h, 0.0), Vec3::new(h, h, 2.0), bc).unwrap();
        let mut m = Model::new(bbox);
        let cyl = m.add_surface(Surface::cylinder(Axis::Z, Vec3::ZERO, R));
        m.add_cell(vec![-cyl], FUEL).unwrap();
        m.add_cell(vec![cyl.positive()], WATER).unwrap();
        m
    }

    /// Sphere of fuel in a vacuum box of water
    fn sphere_in_box() -> Model {
        let bbox = BoundaryBox::uniform(Vec3::splat(-2.0), Vec3::splat(2.0), Bc::Vacuum).unwrap();
        let mut m = Model::new(bbox);
        let s = m.add_surface(Surface::sphere(Vec3::ZERO, 1.0));
        m.add_cell(vec![-s], FUEL).unwrap();
        m.add_cell(vec![s.positive()], WATER).unwrap();
        m
    }

    /// Slab x < 0.3 fuel, else water; x reflective, y periodic, z: reflective
    /// below, vacuum above
    fn mixed_slab() -> Model {
        let bbox = BoundaryBox::new(
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, 3.0),
            [
                [Bc::Reflective, Bc::Reflective],
                [Bc::Periodic, Bc::Periodic],
                [Bc::Reflective, Bc::Vacuum],
            ],
        )
        .unwrap();
        let mut m = Model::new(bbox);
        let x = m.add_surface(Surface::axis_plane(Axis::X, 0.3));
        m.add_cell(vec![-x], FUEL).unwrap();
        m.add_cell(vec![x.positive()], WATER).unwrap();
        m
    }

    fn walk(m: &Model, o: Vec3, d: Vec3, s: f64) -> (Path, Partition<MaterialId>, Ended) {
        let mut path = Path::new(o, d);
        let mut mats = new_partition();
        let e = m.walk(o, d, s, &mut path, &mut mats);
        (path, mats, e)
    }

    // ---------------- worked examples ----------------

    #[test]
    fn straight_chord_through_the_pin() {
        // Along x through the centre, from the left face to the right face
        let m = pin_cell(Bc::Vacuum);
        let o = Vec3::new(-HALF_PITCH, 0.0, 1.0);
        let (path, mats, e) = walk(&m, o, Vec3::X, 2.0 * HALF_PITCH);
        assert_eq!(e, Ended::Reached);
        assert_eq!(mats.labels(), &[WATER, FUEL, WATER]);
        let w = HALF_PITCH - R;
        assert_all_close(mats.edges(), &[0.0, w, w + 2.0 * R, 2.0 * HALF_PITCH]);
        assert_eq!(path.legs().len(), 1);
    }

    #[test]
    fn vacuum_exit_reports_where() {
        let m = pin_cell(Bc::Vacuum);
        let (path, mats, e) = walk(&m, Vec3::new(0.0, 0.0, 1.0), Vec3::X, 10.0);
        assert_eq!(e, Ended::VacuumExit { s: HALF_PITCH });
        assert_eq!(path.ended(), Some(e));
        assert_close(path.length(), HALF_PITCH);
        assert_eq!(mats.labels(), &[FUEL, WATER]);
        assert_all_close(mats.edges(), &[0.0, R, HALF_PITCH]);
    }

    #[test]
    fn reflection_returns_through_the_pin() {
        // From the centre along +x: hit the reflective face, come back through
        // the fuel, out the other side, reflect again
        let m = pin_cell(Bc::Reflective);
        let (path, mats, e) = walk(&m, Vec3::new(0.0, 0.0, 1.0), Vec3::X, 4.0 * HALF_PITCH);
        assert_eq!(e, Ended::Reached);
        assert_eq!(path.legs().len(), 3);
        assert_eq!(path.last_dir(), Vec3::X);
        assert_eq!(
            mats.labels(),
            &[FUEL, WATER, WATER, FUEL, WATER, WATER, FUEL]
        );
        // Total fuel length: R + 2R + R
        let fuel: f64 = mats
            .iter()
            .filter(|s| s.label == FUEL)
            .map(|s| s.length())
            .sum();
        assert_close(fuel, 4.0 * R);
        assert_vec_close(path.end_point(), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn sphere_chord_through_the_centre() {
        let m = sphere_in_box();
        let (_, mats, e) = walk(&m, Vec3::new(-2.0, 0.0, 0.0), Vec3::X, 4.0);
        assert_eq!(e, Ended::Reached);
        assert_eq!(mats.labels(), &[WATER, FUEL, WATER]);
        assert_all_close(mats.edges(), &[0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn sphere_offset_chord_length() {
        // Impact parameter 0.6 -> chord 2·0.8 through the fuel
        let m = sphere_in_box();
        let (_, mats, _) = walk(&m, Vec3::new(-2.0, 0.6, 0.0), Vec3::X, 4.0);
        let fuel: f64 = mats
            .iter()
            .filter(|s| s.label == FUEL)
            .map(|s| s.length())
            .sum();
        assert_close(fuel, 1.6);
    }

    #[test]
    fn periodic_wrap_keeps_direction() {
        let m = mixed_slab();
        // Along +y from y = 1.5 for 1.0: wraps at y = 2 to y = 0, ends at y = 0.5
        let o = Vec3::new(0.5, 1.5, 1.0);
        let (path, _, e) = walk(&m, o, Vec3::Y, 1.0);
        assert_eq!(e, Ended::Reached);
        assert_eq!(path.legs().len(), 2);
        assert_eq!(path.last_dir(), Vec3::Y);
        assert_vec_close(path.end_point(), Vec3::new(0.5, 0.5, 1.0));
    }

    #[test]
    fn symmetry_plane_then_vacuum() {
        // Down along -z: reflect at z = 0, then up to the vacuum face z = 3
        let m = mixed_slab();
        let (path, _, e) = walk(&m, Vec3::new(0.5, 1.0, 1.0), -Vec3::Z, 10.0);
        assert_eq!(e, Ended::VacuumExit { s: 4.0 });
        assert_eq!(path.legs().len(), 2);
        assert_eq!(path.last_dir(), Vec3::Z);
    }

    #[test]
    fn walk_ending_exactly_on_a_face_does_not_reflect() {
        let m = pin_cell(Bc::Reflective);
        let (path, _, e) = walk(&m, Vec3::new(0.0, 0.0, 1.0), Vec3::X, HALF_PITCH);
        assert_eq!(e, Ended::Reached);
        assert_eq!(path.legs().len(), 1);
        assert_eq!(path.last_dir(), Vec3::X);
    }

    #[test]
    fn start_on_a_surface_and_on_a_face() {
        let m = pin_cell(Bc::Reflective);
        // On the fuel surface heading out: first interval is water
        let (_, mats, _) = walk(&m, Vec3::new(R, 0.0, 1.0), Vec3::X, 0.1);
        assert_eq!(mats.labels(), &[WATER]);
        // On the fuel surface heading in: fuel, the full diameter
        let (_, mats, _) = walk(&m, Vec3::new(R, 0.0, 1.0), -Vec3::X, 2.0 * R);
        assert_eq!(mats.labels(), &[FUEL]);
        // On a reflective face heading out: reflect immediately, no empty leg
        let (path, _, e) = walk(&m, Vec3::new(HALF_PITCH, 0.0, 1.0), Vec3::X, 0.1);
        assert_eq!(e, Ended::Reached);
        assert_eq!(path.legs().len(), 1);
        assert_eq!(path.first_dir(), Vec3::X);
        assert_eq!(path.last_dir(), -Vec3::X);
    }

    #[test]
    fn corner_hit_reflects_both_components() {
        // Diagonal in the xy plane through the box corner
        let m = pin_cell(Bc::Reflective);
        let d = Vec3::new(1.0, 1.0, 0.0).normalize();
        let o = Vec3::new(0.5, 0.5, 1.0);
        let to_corner = (HALF_PITCH - 0.5) * 2f64.sqrt();
        let (path, _, e) = walk(&m, o, d, 2.0 * to_corner);
        assert_eq!(e, Ended::Reached);
        assert_eq!(path.legs().len(), 2);
        assert_vec_close(path.last_dir(), -d);
        assert_vec_close(path.end_point(), o);
    }

    #[test]
    fn zero_length_walk() {
        let m = pin_cell(Bc::Vacuum);
        let (path, mats, e) = walk(&m, Vec3::new(0.0, 0.0, 1.0), Vec3::X, 0.0);
        assert_eq!(e, Ended::Reached);
        assert!(path.is_empty() && mats.is_empty());
    }

    #[test]
    fn segments_feed_optical_depth() {
        // Σ_t: fuel 0.5, water 1.2; chord through the pin centre
        let m = pin_cell(Bc::Vacuum);
        let (_, mats, _) = walk(
            &m,
            Vec3::new(-HALF_PITCH, 0.0, 1.0),
            Vec3::X,
            2.0 * HALF_PITCH,
        );
        let mut segs: Vec<Segment> = Vec::new();
        material_segments(&mats, &mut segs);
        let sigma = |id: MaterialId| if id == FUEL { 0.5 } else { 1.2 };
        let tau: f64 = segs.iter().map(|s| sigma(s.material) * s.length).sum();
        assert_close(tau, 0.5 * 2.0 * R + 1.2 * 2.0 * (HALF_PITCH - R));
    }

    #[test]
    fn reused_buffers_are_reset() {
        let m = pin_cell(Bc::Reflective);
        let mut path = Path::new(Vec3::ZERO, Vec3::X);
        let mut mats = new_partition();
        m.walk(Vec3::new(0.0, 0.0, 1.0), Vec3::X, 5.0, &mut path, &mut mats);
        m.walk(Vec3::new(0.0, 0.0, 1.0), Vec3::Y, 0.2, &mut path, &mut mats);
        assert_eq!(path.legs().len(), 1);
        assert_eq!(mats.labels(), &[FUEL]);
        assert_close(mats.end(), 0.2);
    }

    #[test]
    #[should_panic(expected = "gap in the model")]
    fn gap_in_the_model_panics() {
        let bbox = BoundaryBox::uniform(Vec3::ZERO, Vec3::ONE, Bc::Vacuum).unwrap();
        let mut m = Model::new(bbox);
        let x = m.add_surface(Surface::axis_plane(Axis::X, 0.5));
        m.add_cell(vec![-x], FUEL).unwrap(); // x > 0.5 is in no cell
        let mut path = Path::new(Vec3::ZERO, Vec3::X);
        let mut mats = new_partition();
        m.walk(Vec3::new(0.1, 0.5, 0.5), Vec3::X, 1.0, &mut path, &mut mats);
    }

    // ---------------- method of images ----------------

    #[test]
    fn single_reflection_image_by_hand() {
        let m = pin_cell(Bc::Reflective);
        let (x, y) = (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.5, 0.0, 1.0));
        check_image(&m, x, y, [1, 0, 0]);
        // Reflected at x = 0.63: 0.63 + 0.13 = 0.76
        let xp = m.boundary().image([1, 0, 0], y);
        let (path, _) = check_walk(&m, x, (xp - x).normalize(), (xp - x).length());
        assert_close(path.length(), 0.76);
        assert_eq!(path.last_dir(), -Vec3::X);
    }

    // ---------------- core Geometry ----------------

    #[test]
    fn model_is_a_core_geometry() {
        let m = pin_cell(Bc::Reflective);
        assert_close(m.volume(), (2.0 * HALF_PITCH).powi(2) * 2.0);
        assert_eq!(<Model as Geometry>::SAMPLE_DIM, 3);
        assert_eq!(m.map_uniform(&[0.0, 0.0, 0.0]), Some(m.boundary().lower()));
        let p = m.map_uniform(&[0.5, 0.5, 0.25]).unwrap();
        assert_vec_close(p, Vec3::new(0.0, 0.0, 0.5));
        // Samples land in a cell, so the driver's points are always valid
        assert_eq!(Walker::material_at(&m, p, Vec3::X), Some(FUEL));
    }

    // ---------------- conformance ----------------

    #[test]
    fn conformance_reflective_pin_cell() {
        run_suite(&pin_cell(Bc::Reflective), 300, 1);
    }

    #[test]
    fn conformance_vacuum_pin_cell() {
        run_suite(&pin_cell(Bc::Vacuum), 300, 2);
    }

    #[test]
    fn conformance_periodic_pin_cell() {
        run_suite(&pin_cell(Bc::Periodic), 300, 3);
    }

    #[test]
    fn conformance_sphere_in_vacuum_box() {
        run_suite(&sphere_in_box(), 300, 4);
    }

    #[test]
    fn conformance_mixed_boundaries() {
        run_suite(&mixed_slab(), 300, 5);
    }
}
