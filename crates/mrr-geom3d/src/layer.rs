//! Layers: labelled partitions of space, applied along paths.
//!
//! A layer splits space into labelled regions (materials, overlay grid cells,
//! tally bins, ...). Given one straight [`Leg`], it appends the labelled
//! intervals of that leg to a [`Partition`] in increasing `s`. Reflections
//! need no special handling: [`Layer::segment_path`] simply segments each leg
//! of a [`Path`] in turn. Different layers' partitions of the same path are
//! combined with [`Partition::merge`].
//!
//! # Contract for implementors
//!
//! - `segment_leg` **appends** to `out` with `Partition::push`, covering the
//!   whole leg `[s_start, s_end]` in increasing `s`. Regions outside the
//!   layer's domain must still be covered, so layers with a bounded domain use
//!   an `Option` label (`None` outside).
//! - Labels come from the layer's own structure (which edge was crossed), never
//!   by re-locating a midpoint. Slivers are absorbed by `Partition::push`.
//! - `locate(p, dir)` uses the direction-aware tie rule of [`crate::tol`]: a
//!   point exactly on an edge belongs to the region `dir` points into. It must
//!   agree with the label `segment_leg` gives the interval starting at `p`.

use mrr_math::{Bins, Partition};

use crate::traits::Layer;
use crate::walk::Leg;
use crate::{Vec3, tol};

/// A new empty partition starting at `s = 0` with the shared `tol::LEN`.
/// Every partition a layer fills should be created this way (or reset from
/// one), so that all merges use the same sliver tolerance.
pub fn new_partition<L: Copy>() -> Partition<L> {
    Partition::new(0.0, tol::LEN)
}

/// Bins along one coordinate axis: region `i` is the slab
/// `edges[i] <= x[axis] < edges[i+1]` (infinite in the other two directions).
///
/// Labels are bin indices, `None` outside `[lower, upper]`. Useful as an axial
/// overlay, and as the simplest real layer for testing paths and merges.
#[derive(Debug, Clone)]
pub struct AxisBins {
    axis: usize,
    bins: Bins,
}

impl AxisBins {
    /// `axis` is 0, 1 or 2 for x, y, z
    pub fn new(axis: usize, bins: Bins) -> Self {
        assert!(axis < 3, "axis must be 0, 1 or 2, got {axis}");
        Self { axis, bins }
    }

    pub fn axis(&self) -> usize {
        self.axis
    }

    pub fn bins(&self) -> &Bins {
        &self.bins
    }

    /// Bin containing coordinate `x` when moving with velocity component `v`
    /// along the axis. Moving up (or not moving): half-open `[e_i, e_{i+1})`.
    /// Moving down: `(e_i, e_{i+1}]`, so a point on an edge belongs to the bin
    /// below it, which is where it is heading.
    fn locate_coord(&self, x: f64, v: f64) -> Option<usize> {
        if v < 0.0 {
            let edges = self.bins.edges();
            if !(x > self.bins.lower() && x <= self.bins.upper()) {
                return None;
            }
            // Number of edges strictly below x, minus one
            Some(edges.partition_point(|&e| e < x) - 1)
        } else {
            self.bins.locate(x)
        }
    }
}

impl Layer for AxisBins {
    type Label = Option<usize>;

    fn locate(&self, p: Vec3, dir: Vec3) -> Option<Option<usize>> {
        // Every point gets a label; `None` inside means "outside the bins"
        Some(self.locate_coord(p[self.axis], dir[self.axis]))
    }

    fn segment_leg(&self, leg: &Leg, out: &mut Partition<Option<usize>>) {
        // Along the axis the leg is x(s) = x0 + v (s - s_start)
        let (x0, v) = (leg.origin[self.axis], leg.dir[self.axis]);
        let x1 = x0 + v * leg.length();
        let edges = self.bins.edges();
        let s_at = |k: usize| leg.s_start + (edges[k] - x0) / v;

        // Label at the start; an edge exactly at x0 is resolved by direction
        let mut label = self.locate_coord(x0, v);

        // Cross every edge strictly between x0 and x1 (an edge exactly at x1
        // is not crossed within this leg), closing the current interval at
        // each and switching to the bin just entered.
        if v > 0.0 {
            // edges with x0 < e < x1, ascending; entering the bin above edge k
            let lo = edges.partition_point(|&e| e <= x0);
            let hi = edges.partition_point(|&e| e < x1);
            for k in lo..hi {
                out.push(s_at(k), label);
                label = (k < self.bins.n_bins()).then_some(k);
            }
        } else if v < 0.0 {
            // edges with x1 < e < x0, descending; entering the bin below edge k
            let lo = edges.partition_point(|&e| e <= x1);
            let hi = edges.partition_point(|&e| e < x0);
            for k in (lo..hi).rev() {
                out.push(s_at(k), label);
                label = k.checked_sub(1);
            }
        }
        out.push(leg.s_end, label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;
    use crate::walk::{Ended, Path};

    /// z bins [0,1) [1,1.5) [1.5,4)
    fn zbins() -> AxisBins {
        AxisBins::new(2, Bins::new(vec![0.0, 1.0, 1.5, 4.0]).unwrap())
    }

    fn leg(origin: Vec3, dir: Vec3, length: f64) -> Leg {
        Leg {
            origin,
            dir: dir.normalize(),
            s_start: 0.0,
            s_end: length,
        }
    }

    fn segment(layer: &AxisBins, leg: &Leg) -> Partition<Option<usize>> {
        let mut out = new_partition();
        out.reset(leg.s_start);
        layer.segment_leg(leg, &mut out);
        out
    }

    /// Every interval's label agrees with `locate` at its midpoint, and the
    /// partition covers the leg exactly
    #[track_caller]
    fn check_consistent(layer: &AxisBins, leg: &Leg) {
        let part = segment(layer, leg);
        assert_close(part.start(), leg.s_start);
        assert_close(part.end(), leg.s_end);
        for span in part.iter() {
            let mid = leg.point_at(span.midpoint());
            assert_eq!(
                Some(span.label),
                layer.locate(mid, leg.dir),
                "span {span:?} of leg {leg:?}"
            );
        }
    }

    #[test]
    fn locate_breaks_ties_by_direction() {
        let z = zbins();
        let at = |zz: f64| Vec3::new(0.0, 0.0, zz);
        assert_eq!(z.locate(at(1.0), Vec3::Z), Some(Some(1))); // moving up: bin above
        assert_eq!(z.locate(at(1.0), -Vec3::Z), Some(Some(0))); // moving down: bin below
        assert_eq!(z.locate(at(1.0), Vec3::X), Some(Some(1))); // not moving in z: half-open
        assert_eq!(z.locate(at(0.0), -Vec3::Z), Some(None)); // leaving through the bottom
        assert_eq!(z.locate(at(4.0), -Vec3::Z), Some(Some(2))); // entering through the top
        assert_eq!(z.locate(at(4.0), Vec3::Z), Some(None));
        assert_eq!(z.locate(at(-1.0), Vec3::Z), Some(None));
    }

    #[test]
    fn upward_leg_through_all_bins() {
        let part = segment(&zbins(), &leg(Vec3::new(0.0, 0.0, -0.5), Vec3::Z, 5.0));
        assert_eq!(part.edges(), &[0.0, 0.5, 1.5, 2.0, 4.5, 5.0]);
        assert_eq!(part.labels(), &[None, Some(0), Some(1), Some(2), None]);
    }

    #[test]
    fn downward_leg_reverses_the_order() {
        let part = segment(&zbins(), &leg(Vec3::new(0.0, 0.0, 3.0), -Vec3::Z, 2.5));
        assert_eq!(part.labels(), &[Some(2), Some(1), Some(0)]);
        assert_all_close(part.edges(), &[0.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn oblique_leg_scales_by_the_direction_cosine() {
        // dir.z = 0.5, so each unit of z costs 2 units of s
        let dir = Vec3::new(3f64.sqrt() / 2.0, 0.0, 0.5);
        let part = segment(&zbins(), &leg(Vec3::new(0.0, 0.0, 0.5), dir, 3.0));
        assert_all_close(part.edges(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(part.labels(), &[Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn leg_parallel_to_the_planes_has_one_interval() {
        let part = segment(&zbins(), &leg(Vec3::new(0.0, 0.0, 1.2), Vec3::X, 10.0));
        assert_eq!(part.labels(), &[Some(1)]);
        // Exactly on an edge: half-open rule
        let part = segment(&zbins(), &leg(Vec3::new(0.0, 0.0, 1.0), Vec3::Y, 2.0));
        assert_eq!(part.labels(), &[Some(1)]);
    }

    #[test]
    fn legs_starting_and_ending_on_edges() {
        let z = zbins();
        // Starts on z = 1 moving up, ends on z = 1.5: one bin, no sliver
        let part = segment(&z, &leg(Vec3::new(0.0, 0.0, 1.0), Vec3::Z, 0.5));
        assert_eq!(part.labels(), &[Some(1)]);
        // Starts on z = 1.5 moving down, ends on z = 0
        let part = segment(&z, &leg(Vec3::new(0.0, 0.0, 1.5), -Vec3::Z, 1.5));
        assert_eq!(part.labels(), &[Some(1), Some(0)]);
        assert_all_close(part.edges(), &[0.0, 0.5, 1.5]);
    }

    #[test]
    fn zero_length_leg_adds_nothing() {
        let part = segment(&zbins(), &leg(Vec3::new(0.0, 0.0, 1.0), Vec3::Z, 0.0));
        assert!(part.is_empty());
    }

    #[test]
    fn random_legs_are_consistent_with_locate() {
        // Deterministic pseudo-random legs, including starts on edges and
        // directions with tiny or zero z components
        let z = zbins();
        let mut state = 7u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let edges = [0.0, 1.0, 1.5, 4.0];
        for i in 0..2000 {
            let z0 = if i % 5 == 0 {
                edges[i % 4]
            } else {
                -1.0 + 6.0 * rnd()
            };
            let dz = match i % 7 {
                0 => 0.0,
                1 => 1e-9 * (rnd() - 0.5),
                _ => rnd() - 0.5,
            };
            let dir = Vec3::new(rnd() - 0.5, rnd() - 0.5, dz);
            if dir.length() == 0.0 {
                continue;
            }
            check_consistent(&z, &leg(Vec3::new(0.0, 0.0, z0), dir, 8.0 * rnd()));
        }
    }

    #[test]
    fn segment_path_follows_reflections() {
        // Up from z = 0.5 to the plane z = 3, reflect, back down to z = 0.25
        let mut path = Path::new(Vec3::new(0.0, 0.0, 0.5), Vec3::Z);
        path.push_leg(Leg {
            origin: Vec3::new(0.0, 0.0, 0.5),
            dir: Vec3::Z,
            s_start: 0.0,
            s_end: 2.5,
        });
        path.push_leg(Leg {
            origin: Vec3::new(0.0, 0.0, 3.0),
            dir: -Vec3::Z,
            s_start: 2.5,
            s_end: 5.25,
        });
        path.finish(Ended::Reached);

        let mut out = new_partition();
        zbins().segment_path(&path, &mut out);
        // Up: bin 0 (0.5), bin 1 (0.5), bin 2 (1.5); down: bin 2 (1.5), bin 1 (0.5), bin 0 (0.75).
        // The two bin-2 intervals meet at the reflection and stay separate.
        assert_eq!(
            out.labels(),
            &[Some(0), Some(1), Some(2), Some(2), Some(1), Some(0)]
        );
        assert_all_close(out.edges(), &[0.0, 0.5, 1.0, 2.5, 4.0, 4.5, 5.25]);
        let total: f64 = out.iter().map(|s| s.length()).sum();
        assert_close(total, path.length());
    }

    #[test]
    fn empty_path_gives_empty_partition() {
        let mut out = new_partition::<Option<usize>>();
        zbins().segment_path(&Path::new(Vec3::ZERO, Vec3::Z), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn merging_two_layers_matches_brute_force() {
        let z = zbins();
        let x = AxisBins::new(0, Bins::new(vec![-1.0, 0.0, 0.7, 2.0]).unwrap());
        let dir = Vec3::new(0.6, 0.0, 0.8);
        let l = leg(Vec3::new(-0.8, 0.0, 0.1), dir, 4.0);
        let (pz, px) = (segment(&z, &l), segment(&x, &l));
        let mut both = new_partition();
        pz.merge(&px, &mut both);
        assert_close(both.end(), 4.0);
        for span in both.iter() {
            let mid = l.point_at(span.midpoint());
            assert_eq!(Some(span.label.0), z.locate(mid, dir));
            assert_eq!(Some(span.label.1), x.locate(mid, dir));
        }
        // Every edge of either input appears in the merge
        for &e in pz.edges().iter().chain(px.edges()) {
            assert!(both.edges().iter().any(|&o| (o - e).abs() <= tol::LEN));
        }
    }

    #[test]
    #[should_panic(expected = "axis must be")]
    fn bad_axis_panics() {
        AxisBins::new(3, Bins::new(vec![0.0, 1.0]).unwrap());
    }
}
