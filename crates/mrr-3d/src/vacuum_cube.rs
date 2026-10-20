//! A bare homogeneous cube with vacuum faces: the first 3D problem where
//! leakage matters, solved by the meshless method and by an independent
//! analog Monte Carlo reference ([`crate::analog`]).
//!
//! All faces are vacuum, so the gather needs no images and no cutoff
//! (`R = ∞`): the meshless k has no truncation bias, only the statistical
//! error and the `O(1/N)` bias of a ratio estimate in power iteration.

use mrr_core::convergence::MaxCycles;
use mrr_core::{KUpdate, Problem, RunConfig, power_iterate_with};
use mrr_geom3d::Bc;
use mrr_xs::{EnergyGroups, MaterialLibrary};

use crate::analog::{AnalogCube, AnalogResult};
use crate::problems::{homogeneous_box, one_group_material};
use crate::{ExactGather, HarmonicEmission, MixtureProposal};

/// The benchmark cube: half-width 2 cm (4 mfp across), c = Σ_s/Σ_t = 0.5,
/// νΣ_f = 0.8, so k_inf = 1.6 and leakage brings k near 1
pub const CUBE: AnalogCube = AnalogCube {
    half_width: 2.0,
    sigma_t: 1.0,
    sigma_s: 0.5,
    nu_sigma_f: 0.8,
};

/// Settings for one meshless run
#[derive(Debug, Clone, Copy)]
pub struct MeshlessSettings {
    pub n: usize,
    pub cycles: usize,
    pub inactive: usize,
    pub alpha: f64,
    /// Pole radius in units of the "k atoms per ball" radius
    pub atoms_per_ball: f64,
    pub seed: u64,
}

impl Default for MeshlessSettings {
    fn default() -> Self {
        Self {
            n: 200,
            cycles: 40,
            inactive: 12,
            alpha: 0.3,
            atoms_per_ball: 2.0,
            seed: 1,
        }
    }
}

/// k per cycle of the meshless method on `cube`
pub fn meshless_k(cube: &AnalogCube, s: &MeshlessSettings) -> Vec<f64> {
    let mut lib = MaterialLibrary::new(EnergyGroups::new(vec![2e7, 1e-5]).unwrap(), 0);
    lib.add(one_group_material(
        "fuel",
        cube.sigma_t,
        cube.sigma_s,
        cube.nu_sigma_f,
    ))
    .unwrap();
    let model = homogeneous_box(cube.half_width, Bc::Vacuum);
    let volume = (2.0 * cube.half_width).powi(3);
    let r_c = MixtureProposal::suggested_radius(volume, s.n, s.atoms_per_ball);
    let problem = Problem::new(
        model,
        lib.harmonic_layout(),
        ExactGather::new(&lib, f64::INFINITY),
        HarmonicEmission::new(&lib),
        MixtureProposal::new(s.alpha, r_c),
    );
    let cfg = RunConfig {
        rules: vec![Box::new(MaxCycles::new(s.cycles))],
        n_source_atoms: s.n,
        seed: s.seed,
        k_update: KUpdate::Split,
        sampler: Default::default(),
    };
    let mut ks = Vec::new();
    power_iterate_with(&problem, &cfg, |info| ks.extend(info.k));
    ks
}

/// Mean of the active cycles
pub fn active_mean(ks: &[f64], inactive: usize) -> f64 {
    let a = &ks[inactive..];
    a.iter().sum::<f64>() / a.len() as f64
}

/// The analog reference for [`CUBE`] (a few seconds in release)
pub fn reference() -> AnalogResult {
    CUBE.solve(50_000, 60, 10, 2026)
}
