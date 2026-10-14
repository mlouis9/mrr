//! Assembling complete slab problems, and reference benchmarks.
//!
//! A [`SlabProblem`] owns the geometry and the cross sections. The gather and
//! emission borrow the library, so the runnable `Problem` is built on demand
//! with [`SlabProblem::problem`] and lives no longer than the `SlabProblem`.

use mrr_core::Problem;
use mrr_core::proposal::UniformProposal;
use mrr_xs::{EnergyGroups, MaterialLibrary, MgMaterialData};

use crate::SlabError;
use crate::emission::SlabEmission;
use crate::gather::ExactGather;
use crate::geometry::Slab1D;

/// The runnable problem type: exact gather, slab emission, uniform proposal
pub type ExactUniformProblem<'a> =
    Problem<Slab1D, ExactGather<'a>, SlabEmission<'a>, UniformProposal>;

#[derive(Debug, Clone)]
pub struct SlabProblem {
    slab: Slab1D,
    lib: MaterialLibrary,
}

impl SlabProblem {
    /// Regions [e_i, e_{i+1}) filled with the library materials named in
    /// `region_materials` (one name per region).
    pub fn new(
        lib: MaterialLibrary,
        region_edges: Vec<f64>,
        region_materials: &[&str],
    ) -> Result<Self, SlabError> {
        let ids = region_materials
            .iter()
            .enumerate()
            .map(|(region, &name)| {
                lib.find(name)
                    .ok_or_else(|| SlabError::UnknownMaterialName {
                        region,
                        name: name.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let slab = Slab1D::new(region_edges, ids)?;
        Ok(Self { slab, lib })
    }

    pub fn slab(&self) -> &Slab1D {
        &self.slab
    }

    pub fn library(&self) -> &MaterialLibrary {
        &self.lib
    }

    /// The runnable problem (exact gather, uniform proposal)
    pub fn problem(&self) -> ExactUniformProblem<'_> {
        Problem::new(
            self.slab.clone(),
            self.lib.layout(),
            ExactGather::new(&self.lib),
            SlabEmission::new(&self.lib),
            UniformProposal,
        )
    }
}

/// Analytical benchmarks from A. Sood, R. A. Forster, D. K. Parsons,
/// "Analytical Benchmark Test Set for Criticality Code Verification",
/// LA-UR-01-3082 (2001).
pub mod benchmarks {
    use super::*;

    /// One-group isotropic Pu-239 (problem set "PUa"), units 1/cm
    pub mod pua {
        pub const NU: f64 = 3.24;
        pub const SIGMA_F: f64 = 0.081600;
        pub const SIGMA_C: f64 = 0.019584;
        pub const SIGMA_S: f64 = 0.225216;
        pub const SIGMA_T: f64 = 0.32640;
        /// Critical half-thickness of the bare slab, PUa-1-0-SL (cm)
        pub const CRITICAL_HALF_WIDTH: f64 = 1.853722;
        /// k_inf of the infinite medium, PUa-1-0-IN
        pub const K_INF: f64 = 2.612903;
    }

    /// One-group isotropic material from Σ_t, Σ_s, νΣ_f (χ = 1)
    pub fn one_group_material(
        name: &str,
        sigma_t: f64,
        sigma_s: f64,
        nu_sigma_f: f64,
    ) -> MgMaterialData {
        MgMaterialData {
            name: name.into(),
            sigma_t: vec![sigma_t],
            scatter: vec![vec![sigma_s]],
            nu_sigma_f: vec![nu_sigma_f],
            chi: vec![1.0],
        }
    }

    /// One energy group spanning a conventional range (the value is unused)
    pub fn one_group() -> EnergyGroups {
        EnergyGroups::new(vec![2.0e7, 1.0e-5]).expect("valid one-group structure")
    }

    /// PUa-1-0-SL: bare Pu-239 slab of width 2 × 1.853722 cm, vacuum on both
    /// sides, isotropic scattering. Reference k = 1.
    pub fn pua_1_0_sl() -> SlabProblem {
        pua_slab(2.0 * pua::CRITICAL_HALF_WIDTH)
    }

    /// Bare PUa slab of any width (k < 1 below critical, k > 1 above)
    pub fn pua_slab(width: f64) -> SlabProblem {
        let mut lib = MaterialLibrary::new(one_group(), 0);
        lib.add(one_group_material(
            "pu239",
            pua::SIGMA_T,
            pua::SIGMA_S,
            pua::NU * pua::SIGMA_F,
        ))
        .expect("valid benchmark data");
        SlabProblem::new(lib, vec![0.0, width], &["pu239"]).expect("valid benchmark geometry")
    }

    /// Reference eigenvalue of PUa-1-0-SL
    pub const PUA_1_0_SL_K: f64 = 1.0;
}

#[cfg(test)]
mod tests {
    use super::benchmarks::*;
    use super::*;
    use mrr_core::Geometry;

    #[test]
    fn pua_data_is_consistent() {
        // Σ_t = Σ_s + Σ_c + Σ_f, and k_inf = νΣ_f / Σ_a
        let sum = pua::SIGMA_S + pua::SIGMA_C + pua::SIGMA_F;
        assert!((sum - pua::SIGMA_T).abs() < 1e-12);
        let k_inf = pua::NU * pua::SIGMA_F / (pua::SIGMA_C + pua::SIGMA_F);
        assert!((k_inf - pua::K_INF).abs() < 1e-6);
    }

    #[test]
    fn pua_slab_is_built() {
        let p = pua_1_0_sl();
        assert_eq!(p.slab().n_regions(), 1);
        assert!((p.slab().volume() - 3.707444).abs() < 1e-12);
        assert_eq!(p.library().n_groups(), 1);
        assert_eq!(p.library().max_order(), 0);
    }

    #[test]
    fn regions_are_resolved_by_name() {
        let mut lib = MaterialLibrary::new(one_group(), 0);
        lib.add(one_group_material("a", 1.0, 0.5, 0.0)).unwrap();
        lib.add(one_group_material("b", 2.0, 0.5, 0.0)).unwrap();
        let p = SlabProblem::new(lib, vec![0.0, 1.0, 2.0, 3.0], &["b", "a", "b"]).unwrap();
        let ids: Vec<usize> = p.slab().materials().iter().map(|m| m.0).collect();
        assert_eq!(ids, vec![1, 0, 1]);
    }

    #[test]
    fn unknown_material_name_is_an_error() {
        let mut lib = MaterialLibrary::new(one_group(), 0);
        lib.add(one_group_material("a", 1.0, 0.5, 0.0)).unwrap();
        let err = SlabProblem::new(lib, vec![0.0, 1.0, 2.0], &["a", "c"]).unwrap_err();
        assert!(
            matches!(err, SlabError::UnknownMaterialName { region: 1, ref name } if name == "c")
        );
        assert_eq!(err.to_string(), "region 1 uses unknown material 'c'");
    }

    #[test]
    fn region_count_mismatch_is_an_error() {
        let mut lib = MaterialLibrary::new(one_group(), 0);
        lib.add(one_group_material("a", 1.0, 0.5, 0.0)).unwrap();
        assert!(matches!(
            SlabProblem::new(lib, vec![0.0, 1.0, 2.0], &["a"]),
            Err(SlabError::MaterialCount {
                expected: 2,
                got: 1
            })
        ));
    }
}
