//! Angular basis of the 3D method: real spherical harmonics with the 4π
//! normalization of `mrr_math::harmonics`.
//!
//! Every angular quantity is stored as moments `f_ℓm = ∫ f Ŷ_ℓm dΩ`, so
//!
//! - `f_00 = ∫ f dΩ` is the total (what `MomentLayout::block_integral` sums),
//! - `f(Ω) = Σ_ℓm (2ℓ+1)/(4π) f_ℓm Ŷ_ℓm(Ω)` reconstructs it
//!   ([`AngularBasis::reconstruct`]),
//! - a pencil beam `a δ(Ω − Ω₀)` has moments `a Ŷ_ℓm(Ω₀)`
//!   ([`AngularBasis::add_pencil`]),
//! - scattering is diagonal: `Q_ℓm = Σ_s,ℓ φ_ℓm`.

use mrr_geom3d::Vec3;
use mrr_math::harmonics::{RealHarmonics, lm, reconstruction_weight};

/// Harmonics up to a fixed order, with the reconstruction weights `(2ℓ+1)/4π`
/// precomputed per flat index
#[derive(Debug, Clone)]
pub struct AngularBasis {
    harmonics: RealHarmonics,
    /// `weight[k] = (2ℓ+1)/(4π)` for `k = (ℓ, m)`
    weight: Vec<f64>,
}

impl AngularBasis {
    pub fn new(max_order: usize) -> Self {
        let harmonics = RealHarmonics::new(max_order);
        let weight = (0..harmonics.n_moments())
            .map(|k| reconstruction_weight(lm(k).0))
            .collect();
        Self { harmonics, weight }
    }

    pub fn max_order(&self) -> usize {
        self.harmonics.max_order()
    }

    /// `(L+1)²`
    pub fn n_moments(&self) -> usize {
        self.harmonics.n_moments()
    }

    /// `y[k] = Ŷ_k(dir)`; `dir` a unit vector, `y.len() == n_moments()`
    pub fn eval(&self, dir: Vec3, y: &mut [f64]) {
        self.harmonics.eval(dir.to_array(), y);
    }

    /// `f(Ω) = Σ_k (2ℓ+1)/(4π) f_k Ŷ_k(Ω)` from one group's moments `f` and
    /// precomputed `y = Ŷ(Ω)`
    pub fn reconstruct(&self, moments: &[f64], y: &[f64]) -> f64 {
        debug_assert_eq!(moments.len(), self.n_moments());
        debug_assert_eq!(y.len(), self.n_moments());
        moments
            .iter()
            .zip(y)
            .zip(&self.weight)
            .map(|((f, y), w)| w * f * y)
            .sum()
    }

    /// Add the moments of the pencil beam `value · δ(Ω − Ω₀)` to `out`, given
    /// `y = Ŷ(Ω₀)`: `out[k] += value · Ŷ_k(Ω₀)`
    pub fn add_pencil(&self, value: f64, y: &[f64], out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n_moments());
        for (o, y) in out.iter_mut().zip(y) {
            *o += value * y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_geom3d::testing::*;
    use mrr_math::SphereQuadrature;
    use mrr_math::harmonics::index;
    use std::f64::consts::PI;

    #[test]
    fn isotropic_source_reconstructs_to_total_over_four_pi() {
        let b = AngularBasis::new(3);
        let mut q = vec![0.0; b.n_moments()];
        q[0] = 2.0; // total emission 2
        let mut y = vec![0.0; b.n_moments()];
        for d in [Vec3::X, Vec3::new(0.6, 0.0, 0.8), -Vec3::Z] {
            b.eval(d, &mut y);
            assert_close(b.reconstruct(&q, &y), 2.0 / (4.0 * PI));
        }
    }

    #[test]
    fn linearly_anisotropic_source() {
        // q(Ω) = 3/(4π) a z for q_10 = a: forward-peaked along +z
        let b = AngularBasis::new(1);
        let mut q = vec![0.0; 4];
        q[index(1, 0)] = 0.5;
        let mut y = vec![0.0; 4];
        b.eval(Vec3::Z, &mut y);
        assert_close(b.reconstruct(&q, &y), 3.0 / (4.0 * PI) * 0.5);
        b.eval(Vec3::X, &mut y);
        assert_close(b.reconstruct(&q, &y), 0.0);
    }

    #[test]
    fn reconstruct_then_project_is_identity() {
        let b = AngularBasis::new(3);
        let q: Vec<f64> = (0..b.n_moments()).map(|k| 0.1 * k as f64 - 0.4).collect();
        let quad = SphereQuadrature::product(6, 12);
        let mut y = vec![0.0; b.n_moments()];
        let mut proj = vec![0.0; b.n_moments()];
        for (d, w) in quad.iter() {
            b.eval(Vec3::from_array(d), &mut y);
            let f = b.reconstruct(&q, &y);
            b.add_pencil(w * f, &y, &mut proj);
        }
        assert_all_close(&proj, &q);
    }

    #[test]
    fn pencil_moments() {
        let b = AngularBasis::new(2);
        let mut y = vec![0.0; b.n_moments()];
        let d = Vec3::new(0.0, 0.6, 0.8);
        b.eval(d, &mut y);
        let mut out = vec![0.0; b.n_moments()];
        b.add_pencil(3.0, &y, &mut out);
        assert_close(out[0], 3.0);
        assert_close(out[index(1, -1)], 3.0 * 0.6);
        assert_close(out[index(1, 0)], 3.0 * 0.8);
        assert_close(out[index(1, 1)], 0.0);
    }
}
