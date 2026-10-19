//! Slab emission: flux moments → emission moments at a point.
//!
//! ```text
//! Q_{ℓ,g} = Σ_{g'} Σ_{s,ℓ,g'→g} φ_{ℓ,g'}  +  δ_{ℓ0} χ_g Σ_{g'} νΣ_{f,g'} φ_{0,g'} / k
//! ```
//!
//! The fission part is returned without 1/k; the driver chooses k. Both
//! parts are linear in φ, so they are unbiased for a fixed k.

use mrr_core::MomentLayout;
use mrr_core::traits::{Emission, Emitted, Tracer};
use mrr_xs::MaterialLibrary;

use crate::geometry::Slab1D;

pub struct SlabEmission<'a> {
    lib: &'a MaterialLibrary,
    layout: MomentLayout,
}

impl<'a> SlabEmission<'a> {
    pub fn new(lib: &'a MaterialLibrary) -> Self {
        Self {
            lib,
            layout: lib.layout(),
        }
    }
}

impl Emission<Slab1D> for SlabEmission<'_> {
    fn compute_emission(&self, geometry: &Slab1D, point: f64, phi: &[f64]) -> Emitted {
        let layout = self.layout;
        assert_eq!(
            phi.len(),
            layout.block_len(),
            "flux block has the wrong layout"
        );
        let id = geometry
            .material_at(&point)
            .unwrap_or_else(|| panic!("emission point {point} is outside the slab"));
        let mat = self.lib.material(id);
        let n_groups = self.lib.n_groups();

        let mut scattering = layout.zero_block();
        let (mut x, mut y) = (vec![0.0; n_groups], vec![0.0; n_groups]);
        for l in 0..=self.lib.max_order() {
            for (g, xg) in x.iter_mut().enumerate() {
                *xg = phi[layout.index(l, g)];
            }
            mat.scatter(l).transfer(&x, &mut y); // y[to] = Σ_from S(from,to) x[from]
            for (g, &yg) in y.iter().enumerate() {
                scattering[layout.index(l, g)] = yg;
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
    use mrr_xs::{EnergyGroups, MgMaterialData};

    use mrr_math::testing::*;

    /// 2 groups, L = 1. "fuel" in [0,1), "water" in [1,2].
    fn setup() -> (MaterialLibrary, Slab1D) {
        let mut lib = MaterialLibrary::new(EnergyGroups::new(vec![2e7, 1.0, 1e-5]).unwrap(), 1);
        let fuel = lib
            .add(MgMaterialData {
                name: "fuel".into(),
                sigma_t: vec![0.5, 1.2],
                // scatter[l] stored [from * G + to]
                scatter: vec![vec![0.3, 0.1, 0.0, 0.9], vec![0.05, -0.01, 0.0, 0.02]],
                nu_sigma_f: vec![0.01, 0.2],
                chi: vec![1.0, 0.0],
            })
            .unwrap();
        let water = lib
            .add(MgMaterialData {
                name: "water".into(),
                sigma_t: vec![0.3, 2.0],
                scatter: vec![vec![0.2, 0.1, 0.0, 1.9], vec![0.1, 0.0, 0.0, 0.3]],
                nu_sigma_f: vec![0.0, 0.0],
                chi: vec![0.0, 0.0],
            })
            .unwrap();
        let slab = Slab1D::new(vec![0.0, 1.0, 2.0], vec![fuel, water]).unwrap();
        (lib, slab)
    }

    // Layout index(l, g) = l + 2 g: block = [φ00, φ10, φ01, φ11]
    const PHI: [f64; 4] = [2.0, 0.4, 3.0, -0.6];

    #[test]
    fn scattering_moments_by_hand() {
        let (lib, slab) = setup();
        let e = SlabEmission::new(&lib).compute_emission(&slab, 0.5, &PHI);
        // ℓ=0: g0 = 0.3·2 + 0·3 = 0.6 ; g1 = 0.1·2 + 0.9·3 = 2.9
        // ℓ=1: g0 = 0.05·0.4 + 0·(-0.6) = 0.02 ; g1 = -0.01·0.4 + 0.02·(-0.6) = -0.016
        let want = [0.6, 0.02, 2.9, -0.016];
        for (got, want) in e.scattering.iter().zip(want) {
            assert!(is_close(*got, want, Tol::EXACT), "{got} vs {want}");
        }
    }

    #[test]
    fn fission_is_isotropic_and_follows_chi() {
        let (lib, slab) = setup();
        let e = SlabEmission::new(&lib).compute_emission(&slab, 0.5, &PHI);
        let production = 0.01 * 2.0 + 0.2 * 3.0;
        assert_close(e.fission[0], production); // ℓ=0, g=0: χ0 = 1
        assert_eq!(e.fission[1], 0.0); // ℓ=1
        assert_eq!(e.fission[2], 0.0); // χ1 = 0
        assert_eq!(e.fission[3], 0.0);
    }

    #[test]
    fn material_is_looked_up_at_the_point() {
        let (lib, slab) = setup();
        let em = SlabEmission::new(&lib);
        let water = em.compute_emission(&slab, 1.5, &PHI);
        assert!(water.fission.iter().all(|&f| f == 0.0));
        assert_close(water.scattering[0], 0.2 * 2.0);
        // Interface belongs to the region on its right, right face is inside
        assert_eq!(
            em.compute_emission(&slab, 1.0, &PHI).scattering,
            water.scattering
        );
        assert_eq!(
            em.compute_emission(&slab, 2.0, &PHI).scattering,
            water.scattering
        );
    }

    #[test]
    fn emission_is_linear_in_flux() {
        let (lib, slab) = setup();
        let em = SlabEmission::new(&lib);
        let a = em.compute_emission(&slab, 0.5, &PHI);
        let doubled: Vec<f64> = PHI.iter().map(|p| 2.0 * p).collect();
        let b = em.compute_emission(&slab, 0.5, &doubled);
        for k in 0..4 {
            assert_close(b.scattering[k], 2.0 * a.scattering[k]);
            assert_close(b.fission[k], 2.0 * a.fission[k]);
        }
    }

    #[test]
    #[should_panic(expected = "outside the slab")]
    fn point_outside_panics() {
        let (lib, slab) = setup();
        SlabEmission::new(&lib).compute_emission(&slab, 2.5, &PHI);
    }
}
