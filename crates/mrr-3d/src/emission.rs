//! 3D emission: flux moments → emission moments at a node.
//!
//! With the 4π-normalized harmonics, scattering is diagonal in (ℓ, m):
//!
//! ```text
//! Q_g,ℓm = Σ_g' Σ_s,ℓ,g'→g φ_g',ℓm  +  δ_ℓ0 δ_m0 χ_g Σ_g' νΣ_f,g' φ_g',00 / k
//! ```
//!
//! The fission part is returned without `1/k`; the driver chooses k. Both
//! parts are linear in φ.

use mrr_core::MomentLayout;
use mrr_core::traits::{Emission, Emitted};
use mrr_geom3d::{Vec3, Walker};
use mrr_math::harmonics::index;
use mrr_xs::MaterialLibrary;

/// Direction used to look up the material at a node. Nodes are sampled from a
/// density, so landing exactly on a surface has probability zero; the
/// direction only breaks that tie, and an irrational one avoids aligning with
/// axis-parallel surfaces.
const LOOKUP_DIR: Vec3 = Vec3::new(
    0.267_261_241_912_424_4,
    0.534_522_483_824_848_8,
    0.801_783_725_737_273_1,
);

/// Multigroup emission in real-harmonic moments
pub struct HarmonicEmission<'a> {
    lib: &'a MaterialLibrary,
    layout: MomentLayout,
}

impl<'a> HarmonicEmission<'a> {
    /// The layout (orders, groups) comes from the library, so emission and
    /// cross sections always agree
    pub fn new(lib: &'a MaterialLibrary) -> Self {
        Self {
            lib,
            layout: lib.harmonic_layout(),
        }
    }

    pub fn layout(&self) -> MomentLayout {
        self.layout
    }
}

impl<G: Walker> Emission<G> for HarmonicEmission<'_> {
    fn compute_emission(&self, geometry: &G, point: Vec3, phi: &[f64]) -> Emitted {
        let layout = self.layout;
        assert_eq!(
            phi.len(),
            layout.block_len(),
            "flux block has the wrong layout"
        );
        let id = geometry
            .material_at(point, LOOKUP_DIR)
            .unwrap_or_else(|| panic!("emission point {point:?} is outside the geometry"));
        let mat = self.lib.material(id);
        let n_groups = layout.n_groups();

        let mut scattering = layout.zero_block();
        let (mut x, mut y) = (vec![0.0; n_groups], vec![0.0; n_groups]);
        for l in 0..=self.lib.max_order() {
            let matrix = mat.scatter(l);
            for m in -(l as i64)..=(l as i64) {
                let k = index(l, m);
                for (g, xg) in x.iter_mut().enumerate() {
                    *xg = phi[layout.index(k, g)];
                }
                matrix.transfer(&x, &mut y); // y[to] = Σ_from S(from,to) x[from]
                for (g, &yg) in y.iter().enumerate() {
                    scattering[layout.index(k, g)] = yg;
                }
            }
        }

        let mut fission = layout.zero_block();
        if mat.is_fissile() {
            let production: f64 = (0..n_groups)
                .map(|g| mat.nu_sigma_f()[g] * phi[layout.index(0, g)])
                .sum();
            for (g, &chi) in mat.chi().iter().enumerate() {
                fission[layout.index(0, g)] = chi * production;
            }
        }
        Emitted {
            scattering,
            fission,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::{one_group_material, two_region_box};
    use mrr_geom3d::Bc;
    use mrr_geom3d::testing::*;
    use mrr_xs::{EnergyGroups, MgMaterialData};

    /// Two groups, L = 1. Material 0 fissile with downscatter, material 1 not.
    fn library() -> MaterialLibrary {
        let groups = EnergyGroups::new(vec![2e7, 1.0, 1e-5]).unwrap();
        let mut lib = MaterialLibrary::new(groups, 1);
        lib.add(MgMaterialData {
            name: "fuel".into(),
            sigma_t: vec![0.5, 1.2],
            scatter: vec![vec![0.3, 0.1, 0.0, 0.9], vec![0.05, -0.01, 0.0, 0.02]],
            nu_sigma_f: vec![0.01, 0.2],
            chi: vec![1.0, 0.0],
        })
        .unwrap();
        lib.add(MgMaterialData {
            name: "water".into(),
            sigma_t: vec![0.3, 2.0],
            scatter: vec![vec![0.2, 0.1, 0.0, 1.9], vec![0.1, 0.0, 0.0, 0.3]],
            nu_sigma_f: vec![0.0; 2],
            chi: vec![0.0; 2],
        })
        .unwrap();
        lib
    }

    /// φ for L = 1, two groups: [g0: 00 1-1 10 11 | g1: 00 1-1 10 11]
    const PHI: [f64; 8] = [2.0, 0.1, 0.4, -0.2, 3.0, 0.05, -0.6, 0.3];

    #[test]
    fn scattering_is_diagonal_in_lm() {
        let lib = library();
        let model = two_region_box(1.0, Bc::Vacuum);
        let e =
            HarmonicEmission::new(&lib).compute_emission(&model, Vec3::new(-0.5, 0.0, 0.0), &PHI);
        // ℓ = 0: g0 = 0.3·2 = 0.6; g1 = 0.1·2 + 0.9·3 = 2.9
        // ℓ = 1 (each m): g0 = 0.05 φ_g0; g1 = −0.01 φ_g0 + 0.02 φ_g1
        let expect = [
            0.6,
            0.05 * 0.1,
            0.05 * 0.4,
            0.05 * -0.2,
            2.9,
            -0.01 * 0.1 + 0.02 * 0.05,
            -0.01 * 0.4 + 0.02 * -0.6,
            -0.01 * -0.2 + 0.02 * 0.3,
        ];
        assert_all_close(&e.scattering, &expect);
    }

    #[test]
    fn fission_is_isotropic_and_follows_chi() {
        let lib = library();
        let model = two_region_box(1.0, Bc::Vacuum);
        let e =
            HarmonicEmission::new(&lib).compute_emission(&model, Vec3::new(-0.5, 0.0, 0.0), &PHI);
        let production = 0.01 * 2.0 + 0.2 * 3.0;
        let mut expect = [0.0; 8];
        expect[0] = production; // χ = [1, 0]
        assert_all_close(&e.fission, &expect);
    }

    #[test]
    fn material_is_looked_up_at_the_point() {
        let lib = library();
        let model = two_region_box(1.0, Bc::Vacuum); // fuel x < 0, water x > 0
        let e =
            HarmonicEmission::new(&lib).compute_emission(&model, Vec3::new(0.5, 0.0, 0.0), &PHI);
        assert!(e.fission.iter().all(|&f| f == 0.0));
        assert_close(e.scattering[0], 0.2 * 2.0);
    }

    #[test]
    fn emission_is_linear_and_layout_matches() {
        let lib = library();
        let em = HarmonicEmission::new(&lib);
        assert_eq!(em.layout().n_moments(), 4);
        let model = two_region_box(1.0, Bc::Vacuum);
        let p = Vec3::new(-0.5, 0.2, 0.1);
        let a = em.compute_emission(&model, p, &PHI);
        let doubled: Vec<f64> = PHI.iter().map(|v| 2.0 * v).collect();
        let b = em.compute_emission(&model, p, &doubled);
        for k in 0..8 {
            assert_close(b.scattering[k], 2.0 * a.scattering[k]);
            assert_close(b.fission[k], 2.0 * a.fission[k]);
        }
    }

    #[test]
    #[should_panic(expected = "outside the geometry")]
    fn point_outside_panics() {
        let mut lib = MaterialLibrary::new(EnergyGroups::new(vec![2e7, 1e-5]).unwrap(), 0);
        lib.add(one_group_material("a", 1.0, 0.5, 0.0)).unwrap();
        let model = two_region_box(1.0, Bc::Vacuum);
        HarmonicEmission::new(&lib).compute_emission(&model, Vec3::splat(5.0), &[1.0]);
    }
}
