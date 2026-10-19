//! A CSG model: surfaces, cells defined by half-spaces, and the outer boundary.
//!
//! A [`Cell`] is the intersection of half-spaces (one side of each listed
//! surface) and holds one material. A [`Model`] owns the surfaces, the cells
//! and the [`BoundaryBox`] that closes the domain. Point location
//! ([`Model::find_cell`]) returns the first cell containing the point, using
//! the direction-aware tie rule on every surface.
//!
//! The model expects cells that are disjoint and cover the box; it does not
//! check this when built (that needs sampling), but
//! [`Model::check_partition`] tests it on a set of points.
//!
//! Unions, complements and lattices come later; intersections already cover
//! pin cells, slabs and nested spheres.

use std::fmt;

use mrr_core::MaterialId;
use mrr_geom3d::Vec3;
use mrr_geom3d::boundary::BoundaryBox;

use crate::surface::Surface;

/// Index of a surface in a model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceId(pub usize);

/// Index of a cell in a model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId(pub usize);

/// One side of a surface
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Halfspace {
    pub surface: SurfaceId,
    /// `true` for the positive side (`eval > 0`), `false` for the negative side
    pub positive: bool,
}

impl SurfaceId {
    /// Negative side: below a plane, inside a sphere or cylinder
    pub fn negative(self) -> Halfspace {
        Halfspace {
            surface: self,
            positive: false,
        }
    }

    /// Positive side: above a plane, outside a sphere or cylinder
    pub fn positive(self) -> Halfspace {
        Halfspace {
            surface: self,
            positive: true,
        }
    }
}

/// `-s` is the negative half-space of `s`, as in OpenMC's region syntax
impl std::ops::Neg for SurfaceId {
    type Output = Halfspace;
    fn neg(self) -> Halfspace {
        self.negative()
    }
}

/// A region (intersection of half-spaces) filled with one material
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    region: Vec<Halfspace>,
    material: MaterialId,
}

impl Cell {
    pub fn region(&self) -> &[Halfspace] {
        &self.region
    }

    pub fn material(&self) -> MaterialId {
        self.material
    }
}

/// Why a model could not be built
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// A half-space refers to a surface the model does not have
    UnknownSurface(SurfaceId),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelError::UnknownSurface(id) => write!(f, "cell refers to unknown surface {}", id.0),
        }
    }
}

impl std::error::Error for ModelError {}

/// A point that `check_partition` found in no cell or in several
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionDefect {
    /// Inside the box but in no cell
    Gap(Vec3),
    /// In more than one cell
    Overlap(Vec3, Vec<CellId>),
}

/// Surfaces, cells and the outer boundary of a problem
#[derive(Debug, Clone)]
pub struct Model {
    surfaces: Vec<Surface>,
    cells: Vec<Cell>,
    boundary: BoundaryBox,
}

impl Model {
    /// Empty model closed by `boundary`
    pub fn new(boundary: BoundaryBox) -> Self {
        Self {
            surfaces: Vec::new(),
            cells: Vec::new(),
            boundary,
        }
    }

    pub fn add_surface(&mut self, surface: Surface) -> SurfaceId {
        self.surfaces.push(surface);
        SurfaceId(self.surfaces.len() - 1)
    }

    /// Add a cell: the intersection of `region` (empty = the whole box),
    /// filled with `material`
    pub fn add_cell(
        &mut self,
        region: Vec<Halfspace>,
        material: MaterialId,
    ) -> Result<CellId, ModelError> {
        if let Some(h) = region.iter().find(|h| h.surface.0 >= self.surfaces.len()) {
            return Err(ModelError::UnknownSurface(h.surface));
        }
        self.cells.push(Cell { region, material });
        Ok(CellId(self.cells.len() - 1))
    }

    pub fn boundary(&self) -> &BoundaryBox {
        &self.boundary
    }

    pub fn surfaces(&self) -> &[Surface] {
        &self.surfaces
    }

    pub fn surface(&self, id: SurfaceId) -> &Surface {
        &self.surfaces[id.0]
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn cell(&self, id: CellId) -> &Cell {
        &self.cells[id.0]
    }

    /// `true` if `p` is in `cell`, ties on its surfaces broken by `dir`
    pub fn cell_contains(&self, cell: CellId, p: Vec3, dir: Vec3) -> bool {
        self.cells[cell.0]
            .region
            .iter()
            .all(|h| self.surface(h.surface).sense(p, dir) == h.positive)
    }

    /// The cell containing `p` (first match), ties broken by `dir`; `None`
    /// outside the boundary box or in no cell.
    ///
    /// Linear in the number of cells; acceleration (lattices, neighbour lists)
    /// comes later without changing this contract.
    pub fn find_cell(&self, p: Vec3, dir: Vec3) -> Option<CellId> {
        if !self.boundary.contains(p) {
            return None;
        }
        (0..self.cells.len())
            .map(CellId)
            .find(|&c| self.cell_contains(c, p, dir))
    }

    /// Material at `p` (see [`find_cell`](Self::find_cell))
    pub fn material_at(&self, p: Vec3, dir: Vec3) -> Option<MaterialId> {
        self.find_cell(p, dir).map(|c| self.cells[c.0].material)
    }

    /// Check that each point (inside the box) lies in exactly one cell. Returns
    /// every defect found; empty means the cells partition the sampled points.
    pub fn check_partition(
        &self,
        points: impl IntoIterator<Item = Vec3>,
        dir: Vec3,
    ) -> Vec<PartitionDefect> {
        let mut defects = Vec::new();
        for p in points {
            if !self.boundary.contains(p) {
                continue;
            }
            let hits: Vec<CellId> = (0..self.cells.len())
                .map(CellId)
                .filter(|&c| self.cell_contains(c, p, dir))
                .collect();
            match hits.len() {
                0 => defects.push(PartitionDefect::Gap(p)),
                1 => {}
                _ => defects.push(PartitionDefect::Overlap(p, hits)),
            }
        }
        defects
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::Axis;
    use mrr_geom3d::boundary::Bc;
    use mrr_geom3d::testing::*;

    const FUEL: MaterialId = MaterialId(0);
    const WATER: MaterialId = MaterialId(1);
    const PITCH: f64 = 1.26;
    const R_FUEL: f64 = 0.41;

    /// Infinite-lattice pin cell: fuel cylinder in water, reflective box
    fn pin_cell() -> (Model, CellId, CellId) {
        let h = PITCH / 2.0;
        let bbox =
            BoundaryBox::uniform(Vec3::new(-h, -h, 0.0), Vec3::new(h, h, 1.0), Bc::Reflective)
                .unwrap();
        let mut m = Model::new(bbox);
        let cyl = m.add_surface(Surface::cylinder(Axis::Z, Vec3::ZERO, R_FUEL));
        let fuel = m.add_cell(vec![-cyl], FUEL).unwrap();
        let water = m.add_cell(vec![cyl.positive()], WATER).unwrap();
        (m, fuel, water)
    }

    /// Two-region slab along x: fuel for x < 0, water for x > 0, inside [-1, 1]³
    fn slab() -> Model {
        let bbox = BoundaryBox::uniform(Vec3::splat(-1.0), Vec3::ONE, Bc::Vacuum).unwrap();
        let mut m = Model::new(bbox);
        let x0 = m.add_surface(Surface::axis_plane(Axis::X, 0.0));
        m.add_cell(vec![x0.negative()], FUEL).unwrap();
        m.add_cell(vec![x0.positive()], WATER).unwrap();
        m
    }

    fn grid_points(m: &Model, n: usize) -> Vec<Vec3> {
        let (lo, size) = (m.boundary().lower(), m.boundary().size());
        let mut pts = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let f = |q: usize| (q as f64 + 0.37) / n as f64;
                    pts.push(lo + Vec3::new(f(i), f(j), f(k)) * size);
                }
            }
        }
        pts
    }

    #[test]
    fn building_and_accessors() {
        let (m, fuel, water) = pin_cell();
        assert_eq!(m.surfaces().len(), 1);
        assert_eq!(m.cells().len(), 2);
        assert_eq!(m.cell(fuel).material(), FUEL);
        assert_eq!(m.cell(water).region(), &[SurfaceId(0).positive()]);
        assert_eq!(-SurfaceId(3), SurfaceId(3).negative());
    }

    #[test]
    fn unknown_surface_is_an_error() {
        let (mut m, _, _) = pin_cell();
        assert_eq!(
            m.add_cell(vec![-SurfaceId(5)], FUEL),
            Err(ModelError::UnknownSurface(SurfaceId(5)))
        );
        assert!(
            ModelError::UnknownSurface(SurfaceId(5))
                .to_string()
                .contains('5')
        );
    }

    #[test]
    fn pin_cell_point_location() {
        let (m, fuel, water) = pin_cell();
        let d = Vec3::X;
        assert_eq!(m.find_cell(Vec3::new(0.0, 0.0, 0.5), d), Some(fuel));
        assert_eq!(m.find_cell(Vec3::new(0.3, 0.2, 0.5), d), Some(fuel));
        assert_eq!(m.find_cell(Vec3::new(0.5, 0.5, 0.5), d), Some(water));
        assert_eq!(m.material_at(Vec3::new(0.6, 0.0, 0.1), d), Some(WATER));
        // Outside the box
        assert_eq!(m.find_cell(Vec3::new(1.0, 0.0, 0.5), d), None);
        assert_eq!(m.material_at(Vec3::new(0.0, 0.0, 1.5), d), None);
        // On the box face: inside (closed box)
        assert_eq!(
            m.find_cell(Vec3::new(PITCH / 2.0, 0.0, 0.5), d),
            Some(water)
        );
    }

    #[test]
    fn ties_on_a_surface_follow_the_direction() {
        let (m, fuel, water) = pin_cell();
        let on = Vec3::new(R_FUEL, 0.0, 0.5);
        assert_eq!(m.find_cell(on, Vec3::X), Some(water)); // heading out of the pin
        assert_eq!(m.find_cell(on, -Vec3::X), Some(fuel)); // heading in
        let s = slab();
        let on = Vec3::new(0.0, 0.3, 0.3);
        assert_eq!(s.material_at(on, Vec3::X), Some(WATER));
        assert_eq!(s.material_at(on, -Vec3::X), Some(FUEL));
    }

    #[test]
    fn cells_partition_the_box() {
        let (m, _, _) = pin_cell();
        assert!(m.check_partition(grid_points(&m, 12), Vec3::X).is_empty());
        assert!(
            slab()
                .check_partition(grid_points(&slab(), 9), Vec3::Y)
                .is_empty()
        );
    }

    #[test]
    fn check_partition_finds_gaps_and_overlaps() {
        let bbox = BoundaryBox::uniform(Vec3::splat(-1.0), Vec3::ONE, Bc::Vacuum).unwrap();
        let mut m = Model::new(bbox);
        let x0 = m.add_surface(Surface::axis_plane(Axis::X, 0.0));
        let x5 = m.add_surface(Surface::axis_plane(Axis::X, 0.5));
        m.add_cell(vec![-x0], FUEL).unwrap(); // x < 0
        m.add_cell(vec![-x5], WATER).unwrap(); // x < 0.5: overlaps the first
        // x > 0.5 is in no cell
        let defects = m.check_partition(
            [
                Vec3::new(-0.5, 0.0, 0.0),
                Vec3::new(0.25, 0.0, 0.0),
                Vec3::new(0.75, 0.0, 0.0),
            ],
            Vec3::X,
        );
        assert_eq!(
            defects,
            vec![
                PartitionDefect::Overlap(Vec3::new(-0.5, 0.0, 0.0), vec![CellId(0), CellId(1)]),
                PartitionDefect::Gap(Vec3::new(0.75, 0.0, 0.0)),
            ]
        );
    }

    #[test]
    fn empty_region_is_the_whole_box() {
        let bbox = BoundaryBox::uniform(Vec3::ZERO, Vec3::ONE, Bc::Vacuum).unwrap();
        let mut m = Model::new(bbox);
        let c = m.add_cell(vec![], WATER).unwrap();
        assert_eq!(m.find_cell(Vec3::splat(0.5), Vec3::Z), Some(c));
        assert_eq!(m.find_cell(Vec3::splat(2.0), Vec3::Z), None);
    }

    #[test]
    fn nested_spheres_by_intersection() {
        // Inner sphere fuel, shell water, rest of the box void-like "water"
        let bbox = BoundaryBox::uniform(Vec3::splat(-2.0), Vec3::splat(2.0), Bc::Vacuum).unwrap();
        let mut m = Model::new(bbox);
        let s1 = m.add_surface(Surface::sphere(Vec3::ZERO, 0.5));
        let s2 = m.add_surface(Surface::sphere(Vec3::ZERO, 1.0));
        let core = m.add_cell(vec![-s1], FUEL).unwrap();
        let shell = m.add_cell(vec![s1.positive(), -s2], WATER).unwrap();
        let outside = m.add_cell(vec![s2.positive()], MaterialId(2)).unwrap();
        assert_eq!(m.find_cell(Vec3::new(0.2, 0.0, 0.0), Vec3::X), Some(core));
        assert_eq!(m.find_cell(Vec3::new(0.0, 0.7, 0.0), Vec3::X), Some(shell));
        assert_eq!(
            m.find_cell(Vec3::new(1.5, 1.5, 0.0), Vec3::X),
            Some(outside)
        );
        assert!(m.check_partition(grid_points(&m, 10), Vec3::Z).is_empty());
        // Stochastic-style volume check of the shell on a fine grid
        let pts = grid_points(&m, 40);
        let frac = pts
            .iter()
            .filter(|&&p| m.find_cell(p, Vec3::Z) == Some(shell))
            .count() as f64
            / pts.len() as f64;
        let exact = 4.0 / 3.0 * std::f64::consts::PI * (1.0 - 0.125) / 64.0;
        // Grid quadrature of an indicator: error ~ surface cells / total, ~1%
        assert_close_tol(frac, exact, Tol::new(0.02, 0.0));
    }
}
