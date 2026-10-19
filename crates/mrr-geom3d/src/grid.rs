//! Uniform Cartesian grid: the overlay mesh of the spread proposal (§8.3).
//!
//! The grid covers the box `[lower, lower + n·cell]` with `n[0]·n[1]·n[2]`
//! cells of equal, exactly known volume `|B_m|`. Cell `(i, j, k)` has flat
//! index `m = i + n0·(j + n1·k)`. As a [`Layer`] its label is `Option<usize>`:
//! `None` outside the grid, so a path that leaves it is still fully covered
//! (the spread density is 0 there).
//!
//! The grid is the product of three [`AxisBins`], and it inherits their
//! direction-aware tie rule axis by axis. `segment_leg` segments the leg with
//! each axis and merges the three partitions with `Partition::merge`, so all
//! sliver handling (a leg through a grid edge or corner crosses two or three
//! planes at once) lives in `Partition::push`. A one-pass Amanatides–Woo
//! traversal would avoid the temporary partitions; if it is ever written for
//! speed, test it against this version.

use std::fmt;

use mrr_math::{Bins, Partition};

use crate::layer::AxisBins;
use crate::traits::Layer;
use crate::walk::Leg;
use crate::{Vec3, tol};

/// Why a `CartesianGrid` could not be built
#[derive(Debug, Clone, PartialEq)]
pub enum GridError {
    /// `lower` or `cell` is not finite
    NonFinite,
    /// A cell size is zero or negative
    NonPositiveCell { axis: usize },
    /// An axis has no cells
    NoCells { axis: usize },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GridError::NonFinite => write!(f, "grid corner and cell size must be finite"),
            GridError::NonPositiveCell { axis } => {
                write!(f, "grid cell size must be positive along axis {axis}")
            }
            GridError::NoCells { axis } => write!(f, "grid has no cells along axis {axis}"),
        }
    }
}

impl std::error::Error for GridError {}

/// Uniform Cartesian grid of `n[0] × n[1] × n[2]` cells of size `cell`
#[derive(Debug, Clone)]
pub struct CartesianGrid {
    lower: Vec3,
    cell: Vec3,
    n: [usize; 3],
    axes: [AxisBins; 3],
}

impl CartesianGrid {
    /// Grid with corner `lower`, cell size `cell` and `n` cells per axis
    pub fn new(lower: Vec3, cell: Vec3, n: [usize; 3]) -> Result<Self, GridError> {
        if !(lower.is_finite() && cell.is_finite()) {
            return Err(GridError::NonFinite);
        }
        for a in 0..3 {
            if cell[a] <= 0.0 {
                return Err(GridError::NonPositiveCell { axis: a });
            }
            if n[a] == 0 {
                return Err(GridError::NoCells { axis: a });
            }
        }
        let axis = |a: usize| {
            let edges = (0..=n[a]).map(|i| lower[a] + i as f64 * cell[a]).collect();
            AxisBins::new(a, Bins::new(edges).expect("uniform edges are increasing"))
        };
        Ok(Self {
            lower,
            cell,
            n,
            axes: [axis(0), axis(1), axis(2)],
        })
    }

    /// Grid spanning the box `[lower, upper]` with `n` cells per axis
    pub fn from_box(lower: Vec3, upper: Vec3, n: [usize; 3]) -> Result<Self, GridError> {
        let counts = Vec3::new(n[0] as f64, n[1] as f64, n[2] as f64);
        Self::new(lower, (upper - lower) / counts, n)
    }

    pub fn lower(&self) -> Vec3 {
        self.lower
    }

    pub fn upper(&self) -> Vec3 {
        Vec3::new(
            self.axes[0].bins().upper(),
            self.axes[1].bins().upper(),
            self.axes[2].bins().upper(),
        )
    }

    pub fn cell_size(&self) -> Vec3 {
        self.cell
    }

    pub fn shape(&self) -> [usize; 3] {
        self.n
    }

    pub fn n_cells(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }

    /// Volume of every cell (exact: the product of the cell sizes)
    pub fn cell_volume(&self) -> f64 {
        self.cell.x * self.cell.y * self.cell.z
    }

    /// Flat index of cell `(i, j, k)`
    pub fn index(&self, ijk: [usize; 3]) -> usize {
        debug_assert!(
            (0..3).all(|a| ijk[a] < self.n[a]),
            "cell {ijk:?} out of range"
        );
        ijk[0] + self.n[0] * (ijk[1] + self.n[1] * ijk[2])
    }

    /// `(i, j, k)` of flat index `m`
    pub fn ijk(&self, m: usize) -> [usize; 3] {
        assert!(m < self.n_cells(), "cell {m} out of range");
        let (n0, n1) = (self.n[0], self.n[1]);
        [m % n0, (m / n0) % n1, m / (n0 * n1)]
    }

    /// Corners `(lower, upper)` of cell `m`
    pub fn cell_bounds(&self, m: usize) -> (Vec3, Vec3) {
        let ijk = self.ijk(m);
        let mut lo = Vec3::ZERO;
        let mut hi = Vec3::ZERO;
        for a in 0..3 {
            (lo[a], hi[a]) = self.axes[a].bins().bin(ijk[a]);
        }
        (lo, hi)
    }

    /// Cell containing `p`, ties broken by `dir` on each axis (see
    /// [`AxisBins`]); `None` outside the grid
    pub fn cell_of(&self, p: Vec3, dir: Vec3) -> Option<usize> {
        let i = self.axes[0].locate_coord(p.x, dir.x)?;
        let j = self.axes[1].locate_coord(p.y, dir.y)?;
        let k = self.axes[2].locate_coord(p.z, dir.z)?;
        Some(self.index([i, j, k]))
    }

    fn combine(&self, ijk: (Option<usize>, Option<usize>, Option<usize>)) -> Option<usize> {
        match ijk {
            (Some(i), Some(j), Some(k)) => Some(self.index([i, j, k])),
            _ => None,
        }
    }
}

impl Layer for CartesianGrid {
    type Label = Option<usize>;

    fn locate(&self, p: Vec3, dir: Vec3) -> Option<Option<usize>> {
        // Every point gets a label; `None` inside means "outside the grid"
        Some(self.cell_of(p, dir))
    }

    fn segment_leg(&self, leg: &Leg, out: &mut Partition<Option<usize>>) {
        // One partition per axis, then their common refinement. (A closure
        // cannot build these: it would be fixed to a single label type.)
        let start = leg.s_start;
        let mut px = Partition::new(start, tol::LEN);
        let mut py = Partition::new(start, tol::LEN);
        let mut pz = Partition::new(start, tol::LEN);
        self.axes[0].segment_leg(leg, &mut px);
        self.axes[1].segment_leg(leg, &mut py);
        self.axes[2].segment_leg(leg, &mut pz);

        let mut pxy = Partition::new(start, tol::LEN);
        let mut pxyz = Partition::new(start, tol::LEN);
        px.merge(&py, &mut pxy);
        pxy.merge(&pz, &mut pxyz);
        for span in pxyz.iter() {
            let ((i, j), k) = span.label;
            out.push(span.hi, self.combine((i, j, k)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::new_partition;
    use crate::testing::*;
    use crate::walk::{Ended, Path};

    /// 3 × 2 × 4 cells of size (1, 0.5, 0.25) from (-1, 0, 2)
    fn grid() -> CartesianGrid {
        CartesianGrid::new(
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(1.0, 0.5, 0.25),
            [3, 2, 4],
        )
        .unwrap()
    }

    fn leg(origin: Vec3, dir: Vec3, length: f64) -> Leg {
        Leg {
            origin,
            dir: dir.normalize(),
            s_start: 0.0,
            s_end: length,
        }
    }

    fn segment(g: &CartesianGrid, l: &Leg) -> Partition<Option<usize>> {
        let mut out = new_partition();
        out.reset(l.s_start);
        g.segment_leg(l, &mut out);
        out
    }

    #[track_caller]
    fn check_consistent(g: &CartesianGrid, l: &Leg) {
        let part = segment(g, l);
        assert_close(part.start(), l.s_start);
        assert_close(part.end(), l.s_end);
        for span in part.iter() {
            let mid = l.point_at(span.midpoint());
            assert_eq!(
                Some(span.label),
                g.locate(mid, l.dir),
                "span {span:?} of {l:?}"
            );
        }
    }

    #[test]
    fn construction_validates() {
        let lo = Vec3::ZERO;
        assert!(matches!(
            CartesianGrid::new(lo, Vec3::new(1.0, 0.0, 1.0), [1, 1, 1]),
            Err(GridError::NonPositiveCell { axis: 1 })
        ));
        assert!(matches!(
            CartesianGrid::new(lo, Vec3::ONE, [1, 1, 0]),
            Err(GridError::NoCells { axis: 2 })
        ));
        assert!(matches!(
            CartesianGrid::new(Vec3::splat(f64::INFINITY), Vec3::ONE, [1, 1, 1]),
            Err(GridError::NonFinite)
        ));
        assert!(
            GridError::NoCells { axis: 2 }
                .to_string()
                .contains("axis 2")
        );
    }

    #[test]
    fn sizes_and_volumes() {
        let g = grid();
        assert_eq!(g.n_cells(), 24);
        assert_eq!(g.shape(), [3, 2, 4]);
        assert_close(g.cell_volume(), 0.125);
        assert_vec_close(g.upper(), Vec3::new(2.0, 1.0, 3.0));
        // Cell volumes add up to the grid volume
        let span = g.upper() - g.lower();
        assert_close(
            g.cell_volume() * g.n_cells() as f64,
            span.x * span.y * span.z,
        );
        let b = CartesianGrid::from_box(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0), [2, 3, 4]).unwrap();
        assert_vec_close(b.cell_size(), Vec3::ONE);
    }

    #[test]
    fn index_round_trips() {
        let g = grid();
        for m in 0..g.n_cells() {
            assert_eq!(g.index(g.ijk(m)), m);
        }
        assert_eq!(g.index([1, 0, 0]), 1);
        assert_eq!(g.index([0, 1, 0]), 3);
        assert_eq!(g.index([0, 0, 1]), 6);
        let (lo, hi) = g.cell_bounds(g.index([2, 1, 3]));
        assert_vec_close(lo, Vec3::new(1.0, 0.5, 2.75));
        assert_vec_close(hi, Vec3::new(2.0, 1.0, 3.0));
    }

    #[test]
    fn cell_of_inside_outside_and_on_planes() {
        let g = grid();
        let up = Vec3::ONE.normalize();
        assert_eq!(g.cell_of(Vec3::new(-0.5, 0.25, 2.1), up), Some(0));
        assert_eq!(g.cell_of(Vec3::new(1.5, 0.75, 2.9), up), Some(23));
        assert_eq!(g.cell_of(Vec3::new(3.0, 0.25, 2.1), up), None);
        // On the plane x = 0: moving +x -> the cell above, moving -x -> below
        let p = Vec3::new(0.0, 0.25, 2.1);
        assert_eq!(g.cell_of(p, Vec3::X), Some(g.index([1, 0, 0])));
        assert_eq!(g.cell_of(p, -Vec3::X), Some(g.index([0, 0, 0])));
        // On the grid's upper face moving outward: outside
        assert_eq!(g.cell_of(Vec3::new(2.0, 0.25, 2.1), Vec3::X), None);
        assert_eq!(g.cell_of(Vec3::new(2.0, 0.25, 2.1), -Vec3::X), Some(2));
    }

    #[test]
    fn axis_parallel_leg_crosses_one_row() {
        let g = grid();
        let part = segment(&g, &leg(Vec3::new(-1.5, 0.25, 2.1), Vec3::X, 4.0));
        assert_eq!(part.labels(), &[None, Some(0), Some(1), Some(2), None]);
        assert_all_close(part.edges(), &[0.0, 0.5, 1.5, 2.5, 3.5, 4.0]);
    }

    #[test]
    fn diagonal_leg_through_grid_corners_has_no_slivers() {
        // Unit cubes; the main diagonal passes exactly through cell corners
        let g = CartesianGrid::new(Vec3::ZERO, Vec3::ONE, [3, 3, 3]).unwrap();
        let d = Vec3::ONE.normalize();
        let part = segment(&g, &leg(Vec3::ZERO, d, 3f64.sqrt() * 3.0));
        assert_eq!(
            part.labels(),
            &[
                Some(g.index([0, 0, 0])),
                Some(g.index([1, 1, 1])),
                Some(g.index([2, 2, 2]))
            ]
        );
        assert!(part.iter().all(|s| s.length() > 1.0));
    }

    #[test]
    fn random_legs_are_consistent_with_locate() {
        let g = grid();
        let mut state = 11u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let planes = [
            [-1.0, 0.0, 1.0, 2.0],
            [0.0, 0.5, 1.0, 1.0],
            [2.0, 2.25, 2.5, 3.0],
        ];
        for i in 0..2000 {
            let mut o = Vec3::new(-2.0 + 5.0 * rnd(), -0.5 + 2.0 * rnd(), 1.5 + 2.0 * rnd());
            // Some legs start exactly on grid planes (one or several axes)
            if i % 3 == 0 {
                o.x = planes[0][i % 4];
            }
            if i % 5 == 0 {
                o.z = planes[2][i % 4];
            }
            let mut d = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
            // Some legs are axis-parallel or nearly so
            match i % 7 {
                0 => d.y = 0.0,
                1 => {
                    d.x = 0.0;
                    d.z = 0.0
                }
                2 => d.z = 1e-10,
                _ => {}
            }
            if d.length() == 0.0 {
                continue;
            }
            check_consistent(&g, &leg(o, d, 6.0 * rnd()));
        }
    }

    #[test]
    fn segment_path_follows_a_reflection() {
        // Along +x through the row, reflect at x = 2, come back to x = 0.5
        let g = grid();
        let mut path = Path::new(Vec3::new(-0.5, 0.25, 2.1), Vec3::X);
        path.push_leg(Leg {
            origin: Vec3::new(-0.5, 0.25, 2.1),
            dir: Vec3::X,
            s_start: 0.0,
            s_end: 2.5,
        });
        path.push_leg(Leg {
            origin: Vec3::new(2.0, 0.25, 2.1),
            dir: -Vec3::X,
            s_start: 2.5,
            s_end: 4.0,
        });
        path.finish(Ended::Reached);
        let mut out = new_partition();
        g.segment_path(&path, &mut out);
        assert_eq!(out.labels(), &[Some(0), Some(1), Some(2), Some(2), Some(1)]);
        assert_all_close(out.edges(), &[0.0, 0.5, 1.5, 2.5, 3.5, 4.0]);
    }

    #[test]
    fn grid_and_axis_bins_layers_merge() {
        // Overlay grid merged with another layer, as in composite segmentation
        let g = grid();
        let z = AxisBins::new(2, Bins::new(vec![2.0, 2.4, 3.0]).unwrap());
        let l = leg(Vec3::new(-0.9, 0.1, 2.05), Vec3::new(1.0, 0.3, 0.4), 3.0);
        let (pg, pz) = (segment(&g, &l), {
            let mut o = new_partition();
            z.segment_leg(&l, &mut o);
            o
        });
        let mut both = new_partition();
        pg.merge(&pz, &mut both);
        for span in both.iter() {
            let mid = l.point_at(span.midpoint());
            assert_eq!(Some(span.label.0), g.locate(mid, l.dir));
            assert_eq!(Some(span.label.1), z.locate(mid, l.dir));
        }
    }
}
