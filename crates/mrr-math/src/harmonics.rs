//! Real spherical harmonics, 4π-normalized ("Schmidt semi-normalized").
//!
//! # Convention
//!
//! For a unit vector Ω = (x, y, z) = (sinθ cosφ, sinθ sinφ, cosθ):
//!
//! ```text
//! Ŷ_ℓm(Ω) = N_ℓ|m| · P_ℓ^|m|(cosθ) · { cos(mφ)   m > 0
//!                                     { 1         m = 0
//!                                     { sin(|m|φ) m < 0
//! N_ℓm = sqrt((2 − δ_m0) (ℓ−m)! / (ℓ+m)!)        (no Condon–Shortley phase)
//! ```
//!
//! normalized so that
//!
//! ```text
//! ∫ Ŷ_ℓm Ŷ_ℓ'm' dΩ = 4π/(2ℓ+1) δ_ℓℓ' δ_mm'        Ŷ_00 = 1
//! P_ℓ(Ω·Ω') = Σ_m Ŷ_ℓm(Ω) Ŷ_ℓm(Ω')                 (addition theorem, no prefactor)
//! ```
//!
//! This is the 3D analogue of the 1D slab convention: moments are plain
//! projections `f_ℓm = ∫ f Ŷ_ℓm dΩ`, reconstruction is
//! `f(Ω) = Σ (2ℓ+1)/(4π) f_ℓm Ŷ_ℓm(Ω)`, and `f_00 = ∫ f dΩ` is the total. With
//! it the scattering source is simply `Q_ℓm = Σ_s,ℓ φ_ℓm` (Σ_s,ℓ the usual
//! Legendre moments of the scattering kernel), exactly as in 1D.
//!
//! # Evaluation
//!
//! Harmonics are computed in Cartesian form, without trigonometry or poles:
//! `sin^m θ cos(mφ) = Re((x + iy)^m)` and `sin^m θ sin(mφ) = Im((x + iy)^m)`,
//! times the polynomial `P̃_ℓ^m(z) = P_ℓ^m(z) / (1 − z²)^{m/2}` from its
//! three-term recurrence. Accurate for the orders used in transport
//! (ℓ ≲ 20).
//!
//! # Flat index
//!
//! `k = ℓ² + ℓ + m` for `m ∈ [−ℓ, ℓ]`, so order `L` has `(L+1)²` moments and
//! `k = 0` is `(0, 0)`.

/// Number of moments up to and including order `max_order`: `(L+1)²`
pub fn n_moments(max_order: usize) -> usize {
    (max_order + 1) * (max_order + 1)
}

/// Flat index of `(ℓ, m)`, `|m| <= ℓ`
pub fn index(l: usize, m: i64) -> usize {
    debug_assert!(m.unsigned_abs() as usize <= l, "|m| = {m} > l = {l}");
    ((l * l + l) as i64 + m) as usize
}

/// `(ℓ, m)` of flat index `k`
pub fn lm(k: usize) -> (usize, i64) {
    let l = (k as f64).sqrt() as usize;
    // Guard against rounding of the square root
    let l = if (l + 1) * (l + 1) <= k {
        l + 1
    } else if l * l > k {
        l - 1
    } else {
        l
    };
    (l, k as i64 - (l * l + l) as i64)
}

/// Evaluator for all real harmonics up to a fixed order, with the
/// normalizations precomputed
#[derive(Debug, Clone)]
pub struct RealHarmonics {
    max_order: usize,
    /// `norm[ℓ(ℓ+1)/2 + m]` = N_ℓm for 0 <= m <= ℓ
    norm: Vec<f64>,
}

impl RealHarmonics {
    pub fn new(max_order: usize) -> Self {
        let mut norm = Vec::with_capacity((max_order + 1) * (max_order + 2) / 2);
        for l in 0..=max_order {
            for m in 0..=l {
                // (ℓ−m)!/(ℓ+m)! = 1 / Π_{j=ℓ−m+1}^{ℓ+m} j, as a product to avoid overflow
                let ratio: f64 = ((l - m + 1)..=(l + m)).map(|j| 1.0 / j as f64).product();
                let factor = if m == 0 { 1.0 } else { 2.0 };
                norm.push((factor * ratio).sqrt());
            }
        }
        Self { max_order, norm }
    }

    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// `(L+1)²`
    pub fn n_moments(&self) -> usize {
        n_moments(self.max_order)
    }

    /// Fill `out[index(ℓ, m)] = Ŷ_ℓm(dir)` for every `ℓ <= L`. `dir` must be a
    /// unit vector; `out.len()` must be `n_moments()`.
    pub fn eval(&self, dir: [f64; 3], out: &mut [f64]) {
        let lmax = self.max_order;
        assert_eq!(
            out.len(),
            self.n_moments(),
            "output must have (L+1)² entries"
        );
        let [x, y, z] = dir;
        debug_assert!(
            ((x * x + y * y + z * z) - 1.0).abs() < 1e-10,
            "direction is not a unit vector: {dir:?}"
        );

        // c_m + i s_m = (x + i y)^m
        let (mut c, mut s) = (1.0, 0.0);
        // P̃_m^m = (2m − 1)!!
        let mut pmm = 1.0;
        for m in 0..=lmax {
            if m > 0 {
                (c, s) = (c * x - s * y, c * y + s * x);
                pmm *= (2 * m - 1) as f64;
            }
            // Recurrence in ℓ for fixed m: P̃_m^m, P̃_{m+1}^m, ...
            let (mut p_prev, mut p) = (0.0, pmm);
            for l in m..=lmax {
                if l > m {
                    let next = if l == m + 1 {
                        (2 * m + 1) as f64 * z * pmm
                    } else {
                        ((2 * l - 1) as f64 * z * p - (l + m - 1) as f64 * p_prev) / (l - m) as f64
                    };
                    (p_prev, p) = (p, next);
                }
                let n = self.norm[l * (l + 1) / 2 + m] * p;
                let base = l * l + l;
                if m == 0 {
                    out[base] = n;
                } else {
                    out[base + m] = n * c;
                    out[base - m] = n * s;
                }
            }
        }
    }

    /// Convenience: allocate and return all values
    pub fn values(&self, dir: [f64; 3]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_moments()];
        self.eval(dir, &mut out);
        out
    }
}

/// Reconstruction weight `(2ℓ+1)/(4π)` of order `ℓ`: `f(Ω) = Σ w_ℓ f_ℓm Ŷ_ℓm(Ω)`
pub fn reconstruction_weight(l: usize) -> f64 {
    (2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legendre::legendre;
    use crate::quadrature::SphereQuadrature;
    use crate::testing::*;
    use std::f64::consts::PI;

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    fn dirs() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            unit([0.3, -0.5, 0.8]),
            unit([-0.9, 0.2, -0.1]),
            unit([0.1, 0.1, -0.98]),
        ]
    }

    #[test]
    fn index_round_trips() {
        assert_eq!(n_moments(0), 1);
        assert_eq!(n_moments(3), 16);
        assert_eq!(index(0, 0), 0);
        assert_eq!(index(1, -1), 1);
        assert_eq!(index(1, 0), 2);
        assert_eq!(index(1, 1), 3);
        assert_eq!(index(2, -2), 4);
        for k in 0..n_moments(12) {
            let (l, m) = lm(k);
            assert!(m.unsigned_abs() as usize <= l);
            assert_eq!(index(l, m), k);
        }
    }

    #[test]
    fn low_orders_match_closed_forms() {
        let h = RealHarmonics::new(2);
        let r3 = 3f64.sqrt();
        for d in dirs() {
            let [x, y, z] = d;
            let v = h.values(d);
            let expect = [
                1.0,
                y,
                z,
                x,
                r3 * x * y,
                r3 * y * z,
                0.5 * (3.0 * z * z - 1.0),
                r3 * x * z,
                0.5 * r3 * (x * x - y * y),
            ];
            assert_all_close(&v, &expect);
        }
    }

    #[test]
    fn zonal_harmonics_are_legendre() {
        let h = RealHarmonics::new(10);
        for d in dirs() {
            let v = h.values(d);
            for l in 0..=10 {
                assert_close(v[index(l, 0)], legendre(l, d[2]));
            }
        }
    }

    #[test]
    fn orthogonality_with_four_pi_normalization() {
        let lmax = 6;
        let h = RealHarmonics::new(lmax);
        let q = SphereQuadrature::product(lmax + 2, 2 * lmax + 3);
        let n = h.n_moments();
        let mut gram = vec![0.0; n * n];
        let mut y = vec![0.0; n];
        for (d, w) in q.iter() {
            h.eval(d, &mut y);
            for a in 0..n {
                for b in 0..n {
                    gram[a * n + b] += w * y[a] * y[b];
                }
            }
        }
        for a in 0..n {
            for b in 0..n {
                let expected = if a == b {
                    4.0 * PI / (2 * lm(a).0 + 1) as f64
                } else {
                    0.0
                };
                assert_close(gram[a * n + b], expected);
            }
        }
    }

    #[test]
    fn addition_theorem() {
        let lmax = 8;
        let h = RealHarmonics::new(lmax);
        let ds = dirs();
        for a in &ds {
            for b in &ds {
                let (ya, yb) = (h.values(*a), h.values(*b));
                let cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                for l in 0..=lmax {
                    let sum: f64 = (-(l as i64)..=l as i64)
                        .map(|m| ya[index(l, m)] * yb[index(l, m)])
                        .sum();
                    assert_close(sum, legendre(l, cos));
                }
            }
        }
    }

    #[test]
    fn parity() {
        let h = RealHarmonics::new(7);
        for d in dirs() {
            let (v, w) = (h.values(d), h.values([-d[0], -d[1], -d[2]]));
            for k in 0..h.n_moments() {
                let sign = if lm(k).0.is_multiple_of(2) { 1.0 } else { -1.0 };
                assert_close(w[k], sign * v[k]);
            }
        }
    }

    #[test]
    fn reconstruction_inverts_projection() {
        // f(Ω) = Σ w_ℓ f_ℓm Ŷ_ℓm, with f_ℓm = ∫ f Ŷ_ℓm dΩ, for a band-limited f
        let lmax = 4;
        let h = RealHarmonics::new(lmax);
        let coeffs: Vec<f64> = (0..h.n_moments())
            .map(|k| ((k * 7 % 5) as f64 - 2.0) * 0.3)
            .collect();
        let f = |d: [f64; 3]| -> f64 {
            let y = h.values(d);
            (0..h.n_moments())
                .map(|k| reconstruction_weight(lm(k).0) * coeffs[k] * y[k])
                .sum()
        };
        let q = SphereQuadrature::product(lmax + 3, 2 * lmax + 5);
        let mut proj = vec![0.0; h.n_moments()];
        let mut y = vec![0.0; h.n_moments()];
        for (d, w) in q.iter() {
            h.eval(d, &mut y);
            let fd = f(d);
            for k in 0..h.n_moments() {
                proj[k] += w * fd * y[k];
            }
        }
        assert_all_close(&proj, &coeffs);
        // Total over the sphere is the (0,0) moment
        let total: f64 = q.iter().map(|(d, w)| w * f(d)).sum();
        assert_close(total, coeffs[0]);
    }

    #[test]
    fn high_order_values_stay_bounded() {
        // |Ŷ_ℓm| <= 1 for the 4π (Schmidt) normalization
        let h = RealHarmonics::new(20);
        for d in dirs() {
            assert!(h.values(d).iter().all(|v| v.abs() <= 1.0 + 1e-10));
        }
    }

    #[test]
    #[should_panic(expected = "(L+1)²")]
    fn wrong_buffer_panics() {
        RealHarmonics::new(2).eval([0.0, 0.0, 1.0], &mut [0.0; 4]);
    }
}
