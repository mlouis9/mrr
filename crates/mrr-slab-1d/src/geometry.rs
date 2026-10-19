//! 1D slab geometry implementation
//!
//! The slab is the closed interval [lower, upper], split into regions that
//! each hold one material. All edge handling (which region owns an interface,
//! the right face, zero-length pieces) is delegated to `Bins`.
use crate::SlabError;
use mrr_core::traits::Geometry;
use mrr_core::{MaterialId, Segment};
use mrr_math::Bins;

#[derive(Debug, Clone)]
pub struct Slab1D {
    regions: Bins,
    materials: Vec<MaterialId>,
}

impl Slab1D {
    /// Regions [e_i, e_{i+1}) with `materials[i]` in region `i`
    pub fn new(region_edges: Vec<f64>, materials: Vec<MaterialId>) -> Result<Self, SlabError> {
        let regions = Bins::new(region_edges)?;
        if regions.n_bins() != materials.len() {
            return Err(SlabError::MaterialCount {
                expected: regions.n_bins(),
                got: materials.len(),
            });
        }
        Ok(Self { regions, materials })
    }

    /// One region [0, width] of a single material
    pub fn homogeneous(width: f64, material: MaterialId) -> Result<Self, SlabError> {
        Self::new(vec![0.0, width], vec![material])
    }

    pub fn span(&self) -> f64 {
        self.regions.span()
    }

    pub fn n_regions(&self) -> usize {
        self.regions.n_bins()
    }

    pub fn lower(&self) -> f64 {
        self.regions.lower()
    }

    pub fn upper(&self) -> f64 {
        self.regions.upper()
    }

    pub fn regions(&self) -> &Bins {
        &self.regions
    }

    pub fn materials(&self) -> &[MaterialId] {
        &self.materials
    }
}

impl Geometry for Slab1D {
    type Point = f64;
    const SAMPLE_DIM: usize = 1;

    fn volume(&self) -> f64 {
        self.span()
    }

    fn map_uniform(&self, u: &[f64]) -> Option<Self::Point> {
        Some(self.lower() + u[0] * self.span())
    }
}

/// Tracing. Inherent methods rather than a trait: only the slab physics
/// (gather, emission) uses them, and the 3D counterpart is `mrr_geom3d::Walker`.
impl Slab1D {
    /// `None` outside [lower, upper] or for NaN. Interfaces belong to the
    /// region on their right; the right face belongs to the last region.
    pub fn material_at(&self, p: f64) -> Option<MaterialId> {
        self.regions.locate_closed(p).map(|i| self.materials[i])
    }

    /// Segments from `from` to `to`, in travel order. Both points are expected
    /// to be inside the slab (true for anything produced by `map_uniform`);
    /// anything outside is clipped away.
    pub fn trace(&self, from: f64, to: f64, out: &mut Vec<Segment>) {
        debug_assert!(
            self.material_at(from).is_some(),
            "trace start {from} outside slab"
        );
        debug_assert!(
            self.material_at(to).is_some(),
            "trace end {to} outside slab"
        );

        out.clear();
        out.extend(self.regions.overlaps(from, to).map(|(i, length)| Segment {
            material: self.materials[i],
            length,
        }));
        // `overlaps` is ascending in x; travelling left means reverse order
        if to < from {
            out.reverse();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_math::bins::BinsError;
    use mrr_math::testing::*;

    const FUEL: MaterialId = MaterialId(0);
    const WATER: MaterialId = MaterialId(1);

    /// water [0,1) | fuel [1,3) | water [3,3.5]
    fn pin() -> Slab1D {
        Slab1D::new(vec![0.0, 1.0, 3.0, 3.5], vec![WATER, FUEL, WATER]).unwrap()
    }

    fn trace(slab: &Slab1D, a: f64, b: f64) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        slab.trace(a, b, &mut out);
        out.iter().map(|s| (s.material.0, s.length)).collect()
    }

    #[test]
    fn rejects_wrong_material_count() {
        assert!(matches!(
            Slab1D::new(vec![0.0, 1.0, 2.0], vec![FUEL]),
            Err(SlabError::MaterialCount {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn rejects_bad_edges() {
        assert!(matches!(
            Slab1D::new(vec![0.0, 0.0], vec![FUEL]),
            Err(SlabError::Bins(BinsError::NotIncreasing { index: 1 }))
        ));
        assert!(Slab1D::homogeneous(-1.0, FUEL).is_err());
    }

    #[test]
    fn sizes_and_volume() {
        let s = pin();
        assert_eq!(s.n_regions(), 3);
        assert_eq!(s.lower(), 0.0);
        assert_eq!(s.upper(), 3.5);
        assert_eq!(s.volume(), 3.5);
        assert_eq!(s.materials(), &[WATER, FUEL, WATER]);
    }

    #[test]
    fn map_uniform_covers_the_slab() {
        let s = pin();
        assert_eq!(s.map_uniform(&[0.0]), Some(0.0));
        assert_eq!(s.map_uniform(&[0.5]), Some(1.75));
    }

    #[test]
    fn material_at_inside_edges_and_faces() {
        let s = pin();
        assert_eq!(s.material_at(0.5), Some(WATER));
        assert_eq!(s.material_at(2.0), Some(FUEL));
        assert_eq!(s.material_at(0.0), Some(WATER)); // left face
        assert_eq!(s.material_at(1.0), Some(FUEL)); // interface -> right region
        assert_eq!(s.material_at(3.5), Some(WATER)); // right face is inside
    }

    #[test]
    fn material_at_outside_or_nan_is_none() {
        let s = pin();
        assert_eq!(s.material_at(-0.1), None);
        assert_eq!(s.material_at(3.6), None);
        assert_eq!(s.material_at(f64::NAN), None);
    }

    #[test]
    fn trace_in_homogeneous_slab() {
        let s = Slab1D::homogeneous(2.0, FUEL).unwrap();
        assert_eq!(trace(&s, 0.5, 1.75), vec![(0, 1.25)]);
    }

    #[test]
    fn trace_crossing_regions_left_to_right() {
        assert_eq!(
            trace(&pin(), 0.5, 3.25),
            vec![(1, 0.5), (0, 2.0), (1, 0.25)]
        );
    }

    #[test]
    fn trace_right_to_left_is_in_travel_order() {
        assert_eq!(
            trace(&pin(), 3.25, 0.5),
            vec![(1, 0.25), (0, 2.0), (1, 0.5)]
        );
    }

    #[test]
    fn trace_face_to_face_and_on_interfaces() {
        assert_eq!(trace(&pin(), 0.0, 3.5), vec![(1, 1.0), (0, 2.0), (1, 0.5)]);
        assert_eq!(trace(&pin(), 1.0, 3.0), vec![(0, 2.0)]); // no zero-length bits
    }

    #[test]
    fn trace_of_coincident_points_is_empty_and_clears_buffer() {
        let s = pin();
        let mut out = vec![Segment {
            material: FUEL,
            length: 9.0,
        }];
        s.trace(1.0, 1.0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn segment_lengths_sum_to_distance() {
        let s = pin();
        let mut out = Vec::new();
        for i in 0..=35 {
            for j in 0..=35 {
                let (a, b) = (0.1 * i as f64, 0.1 * j as f64);
                s.trace(a, b, &mut out);
                let total: f64 = out.iter().map(|seg| seg.length).sum();
                assert!(is_close(total, (b - a).abs(), Tol::EXACT), "{a} -> {b}");
            }
        }
    }

    #[test]
    fn trace_feeds_optical_depths() {
        use mrr_xs::{EnergyGroups, MaterialLibrary, MgMaterialData, optical_depths};
        let groups = EnergyGroups::new(vec![2e7, 1.0, 1e-5]).unwrap();
        let mut lib = MaterialLibrary::new(groups, 0);
        let mat = |name: &str, st: [f64; 2]| MgMaterialData {
            name: name.into(),
            sigma_t: st.to_vec(),
            scatter: vec![vec![0.0; 4]],
            nu_sigma_f: vec![0.0; 2],
            chi: vec![0.0; 2],
        };
        let fuel = lib.add(mat("fuel", [0.5, 1.2])).unwrap();
        let water = lib.add(mat("water", [0.3, 2.0])).unwrap();
        let s = Slab1D::new(vec![0.0, 1.0, 3.0], vec![water, fuel]).unwrap();

        let mut segs = Vec::new();
        s.trace(0.5, 2.0, &mut segs); // 0.5 water + 1.0 fuel
        let mut tau = [0.0; 2];
        optical_depths(&lib, &segs, &mut tau);
        assert_close(tau[0], 0.5 * 0.3 + 1.0 * 0.5);
        assert_close(tau[1], 0.5 * 2.0 + 1.0 * 1.2);
    }
}
