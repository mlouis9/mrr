//! Exact angular gather (a).
//!
//! For an atom at z_i with weight w_i and moments q_{iℓ'g}, the flux moments
//! at z are
//!
//! ```text
//! φ_{ℓg}(z) = Σ_i w_i Σ_ℓ' (2ℓ'+1)/2 · s_i^{ℓ+ℓ'} · M_{ℓℓ'}(τ_{ig}) · q_{iℓ'g}
//! ```
//!
//! with s_i = sgn(z − z_i), τ_{ig} the group-g optical depth between z_i and
//! z, and M from [`TransferTensor`]. No angular error, no angular variance.
//!
//! **Coincident points.** If z == z_i the pair is skipped. Even moments then
//! carry an E_1(0) = ∞ log pole; the pole is integrable, so the set of
//! targets where it matters has measure zero and skipping it changes no
//! expectation. It only happens with deterministic (grid) samplers.

use mrr_core::traits::{Gather, Tracer};
use mrr_core::{Atoms, MomentLayout, Segment};
use mrr_math::TransferTensor;
use mrr_xs::{MaterialLibrary, optical_depths};

use crate::geometry::Slab1D;

pub struct ExactGather<'a> {
    lib: &'a MaterialLibrary,
    layout: MomentLayout,
    transfer: TransferTensor,
    /// (2ℓ'+1)/2 for each ℓ'
    half_norm: Vec<f64>,
}

impl<'a> ExactGather<'a> {
    /// The moment layout (orders, groups) is taken from the library, so the
    /// gather and the cross sections always agree.
    pub fn new(lib: &'a MaterialLibrary) -> Self {
        let max_order = lib.max_order();
        Self {
            lib,
            layout: lib.layout(),
            transfer: TransferTensor::new(max_order),
            half_norm: (0..=max_order).map(|l| (2 * l + 1) as f64 / 2.0).collect(),
        }
    }

    pub fn layout(&self) -> MomentLayout {
        self.layout
    }
}

/// Scratch buffers reused across the atoms of one `gather` call
struct Scratch {
    segments: Vec<Segment>,
    tau: Vec<f64>,
    expn: Vec<f64>,
    m: Vec<f64>,
}

impl Gather<Slab1D> for ExactGather<'_> {
    fn gather(&self, geometry: &Slab1D, src_atoms: &Atoms<f64>, target: f64) -> Vec<f64> {
        assert_eq!(
            src_atoms.layout().block_len(),
            self.layout.block_len(),
            "atoms and gather use different moment layouts"
        );
        let n_mom = self.transfer.n_moments();
        let n_groups = self.lib.n_groups();
        let mut s = Scratch {
            segments: Vec::new(),
            tau: vec![0.0; n_groups],
            expn: vec![0.0; self.transfer.n_expn()],
            m: vec![0.0; n_mom * n_mom],
        };
        let mut phi = self.layout.zero_block();

        for (i, (&z_i, &w_i)) in src_atoms
            .points()
            .iter()
            .zip(src_atoms.weights())
            .enumerate()
        {
            if target == z_i {
                continue; // see module docs
            }
            let right = target > z_i; // s_i = +1
            geometry.trace(&z_i, &target, &mut s.segments);
            optical_depths(self.lib, &s.segments, &mut s.tau);
            let q = src_atoms.atom_moments(i);

            for (g, &tau) in s.tau.iter().enumerate() {
                self.transfer.matrix(tau, &mut s.expn, &mut s.m);
                for l in 0..n_mom {
                    let mut acc = 0.0;
                    for lp in 0..n_mom {
                        let q_lp = q[self.layout.index(lp, g)];
                        if q_lp == 0.0 {
                            continue; // avoids 0·∞ at τ = 0 in a void
                        }
                        let sign = if right || (l + lp) % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        acc += self.half_norm[lp] * sign * s.m[l * n_mom + lp] * q_lp;
                    }
                    phi[self.layout.index(l, g)] += w_i * acc;
                }
            }
        }
        phi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_core::MaterialId;
    use mrr_math::expint::expn;
    use mrr_math::legendre::legendre;
    use mrr_math::quadrature::GaussLegendre;
    use mrr_xs::{EnergyGroups, MgMaterialData};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    /// Pure absorber(s) with the given Σ_t per group, scattering order L
    fn library(max_order: usize, sigma_t: &[&[f64]]) -> MaterialLibrary {
        let g = sigma_t[0].len();
        let edges: Vec<f64> = (0..=g).map(|k| 10f64.powi(-(k as i32))).collect();
        let mut lib = MaterialLibrary::new(EnergyGroups::new(edges).unwrap(), max_order);
        for (k, st) in sigma_t.iter().enumerate() {
            lib.add(MgMaterialData {
                name: format!("m{k}"),
                sigma_t: st.to_vec(),
                scatter: vec![vec![0.0; g * g]; max_order + 1],
                nu_sigma_f: vec![0.0; g],
                chi: vec![0.0; g],
            })
            .unwrap();
        }
        lib
    }

    fn one_atom(layout: MomentLayout, z: f64, w: f64, q: &[f64]) -> Atoms<f64> {
        let mut atoms = Atoms::new(1, layout);
        atoms.push(z, w, q);
        atoms
    }

    #[test]
    fn isotropic_atom_gives_half_e1_and_signed_e2() {
        let lib = library(1, &[&[0.8]]);
        let slab = Slab1D::homogeneous(4.0, MaterialId(0)).unwrap();
        let gather = ExactGather::new(&lib);
        let atoms = one_atom(gather.layout(), 1.0, 0.5, &[2.0, 0.0]);
        let tau = 0.8 * 1.5;
        let right = gather.gather(&slab, &atoms, 2.5);
        assert!(close(right[0], 0.5 * 0.5 * 2.0 * expn(1, tau)));
        assert!(close(right[1], 0.5 * 0.5 * 2.0 * expn(2, tau)));
        // Same distance on the left (atom at 1.0, target at 2.5 vs atom at
        // 2.5, target at 1.0): φ0 unchanged, φ1 flips sign
        let left = gather.gather(
            &slab,
            &one_atom(gather.layout(), 2.5, 0.5, &[2.0, 0.0]),
            1.0,
        );
        assert!(close(left[0], right[0]));
        assert!(close(left[1], -right[1]));
    }

    #[test]
    fn mirror_symmetry_flips_odd_moments() {
        let lib = library(3, &[&[0.6]]);
        let slab = Slab1D::homogeneous(4.0, MaterialId(0)).unwrap();
        let gather = ExactGather::new(&lib);
        let q = [1.0, 0.3, -0.2, 0.1];
        let q_mirror = [1.0, -0.3, -0.2, -0.1]; // q_ℓ → (-1)^ℓ q_ℓ
        let a = gather.gather(&slab, &one_atom(gather.layout(), 1.0, 1.0, &q), 2.2);
        let b = gather.gather(&slab, &one_atom(gather.layout(), 3.0, 1.0, &q_mirror), 1.8);
        for l in 0..4 {
            let sign = if l % 2 == 0 { 1.0 } else { -1.0 };
            assert!(close(b[l], sign * a[l]), "l={l}");
        }
    }

    #[test]
    fn order_one_matches_closed_form() {
        // I_0 = ½q0E1 + 3/2 s q1 E2,  I_1 = ½ s q0 E2 + 3/2 q1 E3
        let lib = library(1, &[&[1.3]]);
        let slab = Slab1D::homogeneous(2.0, MaterialId(0)).unwrap();
        let gather = ExactGather::new(&lib);
        let (q0, q1, w) = (0.7, -0.4, 0.25);
        let atoms = one_atom(gather.layout(), 1.5, w, &[q0, q1]);
        let tau = 1.3 * 1.1;
        let phi = gather.gather(&slab, &atoms, 0.4); // s = -1
        let (e1, e2, e3) = (expn(1, tau), expn(2, tau), expn(3, tau));
        assert!(close(phi[0], w * (0.5 * q0 * e1 - 1.5 * q1 * e2)));
        assert!(close(phi[1], w * (-0.5 * q0 * e2 + 1.5 * q1 * e3)));
    }

    #[test]
    fn matches_direct_angular_quadrature() {
        // φ_ℓ = w ∫_H P_ℓ(μ) q(μ) e^{-τ/|μ|}/|μ| dμ, q(μ) = Σ (2ℓ'+1)/2 q_ℓ' P_ℓ'(μ)
        let max_order = 4;
        let lib = library(max_order, &[&[0.9]]);
        let slab = Slab1D::homogeneous(5.0, MaterialId(0)).unwrap();
        let gather = ExactGather::new(&lib);
        let q = [1.0, 0.4, -0.3, 0.2, 0.05];
        let (z_i, w) = (2.0, 0.3);
        let atoms = one_atom(gather.layout(), z_i, w, &q);
        let gl = GaussLegendre::new(200);
        for &z in &[0.3, 4.5] {
            let phi = gather.gather(&slab, &atoms, z);
            let tau = 0.9 * (z - z_i).abs();
            let s = (z - z_i).signum();
            for (l, &got) in phi.iter().enumerate() {
                let f = |x: f64| {
                    let m = 0.5 * (x + 1.0);
                    let mu = s * m;
                    let qmu: f64 = (0..=max_order)
                        .map(|lp| (2 * lp + 1) as f64 / 2.0 * q[lp] * legendre(lp, mu))
                        .sum();
                    0.5 * legendre(l, mu) * qmu * (-tau / m).exp() / m
                };
                let want = w * gl.integrate(f);
                assert!((got - want).abs() < 1e-10, "z={z} l={l}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn heterogeneous_path_depends_only_on_optical_depth() {
        // 0.5 cm of Σ=2 plus 1 cm of Σ=0.5 has τ = 1.5, same as 1.5 cm of Σ=1
        let lib = library(2, &[&[2.0], &[0.5], &[1.0]]);
        let het = Slab1D::new(vec![0.0, 1.0, 3.0], vec![MaterialId(0), MaterialId(1)]).unwrap();
        let hom = Slab1D::homogeneous(3.0, MaterialId(2)).unwrap();
        let gather = ExactGather::new(&lib);
        let q = [1.0, 0.2, 0.1];
        let a = gather.gather(&het, &one_atom(gather.layout(), 0.5, 1.0, &q), 2.0);
        let b = gather.gather(&hom, &one_atom(gather.layout(), 0.5, 1.0, &q), 2.0);
        for l in 0..3 {
            assert!(close(a[l], b[l]));
        }
    }

    #[test]
    fn groups_are_independent() {
        let lib = library(1, &[&[0.5, 2.0]]);
        let slab = Slab1D::homogeneous(3.0, MaterialId(0)).unwrap();
        let gather = ExactGather::new(&lib);
        let layout = gather.layout();
        // Source only in group 1
        let mut q = layout.zero_block();
        q[layout.index(0, 1)] = 1.0;
        let phi = gather.gather(&slab, &one_atom(layout, 1.0, 1.0, &q), 2.0);
        assert_eq!(phi[layout.index(0, 0)], 0.0);
        assert_eq!(phi[layout.index(1, 0)], 0.0);
        assert!(close(phi[layout.index(0, 1)], 0.5 * expn(1, 2.0)));
        assert!(close(phi[layout.index(1, 1)], 0.5 * expn(2, 2.0)));
    }

    #[test]
    fn contributions_add_over_atoms() {
        let lib = library(1, &[&[1.0]]);
        let slab = Slab1D::homogeneous(4.0, MaterialId(0)).unwrap();
        let gather = ExactGather::new(&lib);
        let layout = gather.layout();
        let mut both = Atoms::new(2, layout);
        both.push(1.0, 0.5, &[1.0, 0.3]);
        both.push(3.0, 1.5, &[2.0, -0.1]);
        let a = gather.gather(&slab, &one_atom(layout, 1.0, 0.5, &[1.0, 0.3]), 2.2);
        let b = gather.gather(&slab, &one_atom(layout, 3.0, 1.5, &[2.0, -0.1]), 2.2);
        let ab = gather.gather(&slab, &both, 2.2);
        for l in 0..2 {
            assert!(close(ab[l], a[l] + b[l]));
        }
    }

    #[test]
    fn coincident_atom_is_skipped() {
        let lib = library(0, &[&[1.0]]);
        let slab = Slab1D::homogeneous(2.0, MaterialId(0)).unwrap();
        let gather = ExactGather::new(&lib);
        let phi = gather.gather(&slab, &one_atom(gather.layout(), 1.0, 1.0, &[1.0]), 1.0);
        assert_eq!(phi, vec![0.0]);
    }

    #[test]
    fn void_gives_finite_uncollided_flux() {
        // Σ_t = 0: φ_1 = ½ q0 E_2(0) = ½ (finite); φ_0 has the E_1(0) pole
        let lib = library(1, &[&[0.0]]);
        let slab = Slab1D::homogeneous(2.0, MaterialId(0)).unwrap();
        let gather = ExactGather::new(&lib);
        let phi = gather.gather(
            &slab,
            &one_atom(gather.layout(), 0.5, 1.0, &[1.0, 0.0]),
            1.5,
        );
        assert!(phi[0].is_infinite() && phi[0] > 0.0);
        assert!(close(phi[1], 0.5));
    }
}
