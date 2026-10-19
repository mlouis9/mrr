//! Slab half-range transfer integrals
//!
//! M_{ll'}(τ) = ∫_0^1 P_l(m) P_l'(m) e^{-τ/m} / m dm
//!            = Σ_n D_{ll'n} E_{n+1}(τ),
//!
//! where D_{ll'n} = Σ_{p+r=n} a^(l)_p a^(l')_r is the Cauchy product of the
//! monomial coefficients of P_l and P_l'. D depends only on the maximum
//! Legendre order, so it is built once and shared by every atom, target and
//! group.

use crate::expint::expn_all;
use crate::legendre::legendre_monomials;

/// Precomputed D tensor for orders 0..=max_order.
#[derive(Debug, Clone)]
pub struct TransferTensor {
    n_moments: usize,
    n_powers: usize,
    /// D[(l * n_moments + l') * n_powers + n]
    d: Vec<f64>,
}

impl TransferTensor {
    pub fn new(max_order: usize) -> Self {
        let a = legendre_monomials(max_order);
        let n_moments = max_order + 1;
        let n_powers = 2 * max_order + 1;
        let mut d = vec![0.0; n_moments * n_moments * n_powers];
        for l in 0..n_moments {
            for lp in 0..n_moments {
                let base = (l * n_moments + lp) * n_powers;
                for (p, &ap) in a[l].iter().enumerate() {
                    for (r, &ar) in a[lp].iter().enumerate() {
                        d[base + p + r] += ap * ar;
                    }
                }
            }
        }
        Self {
            n_moments,
            n_powers,
            d,
        }
    }

    /// Number of Legendre moments, L + 1
    pub fn n_moments(&self) -> usize {
        self.n_moments
    }

    /// Number of exponential integrals needed, 2L + 1 (E_1 … E_{2L+1})
    pub fn n_expn(&self) -> usize {
        self.n_powers
    }

    pub fn d(&self, l: usize, lp: usize, n: usize) -> f64 {
        self.d[(l * self.n_moments + lp) * self.n_powers + n]
    }

    /// Fill `m[l * n_moments + l'] = M_{ll'}(τ)` given `e[k] = E_{k+1}(τ)`
    /// (from `expn_all`). Splitting the two steps lets callers reuse the
    /// E values, e.g. for a proposal density.
    pub fn matrix_from_expn(&self, e: &[f64], m: &mut [f64]) {
        assert_eq!(e.len(), self.n_powers, "need E_1..E_(2L+1)");
        assert_eq!(m.len(), self.n_moments * self.n_moments);
        for (row, mij) in self.d.chunks_exact(self.n_powers).zip(m.iter_mut()) {
            // Skip zero coefficients so E_1(0) = ∞ enters only where D_{ll'0}
            // ≠ 0 (both l, l' even), never as 0·∞ = NaN
            *mij = row
                .iter()
                .zip(e)
                .filter(|(dn, _)| **dn != 0.0)
                .map(|(dn, en)| dn * en)
                .sum();
        }
    }

    /// Convenience: M(τ) into `m`, using `e` as scratch of length `n_expn()`
    pub fn matrix(&self, tau: f64, e: &mut [f64], m: &mut [f64]) {
        expn_all(tau, e);
        self.matrix_from_expn(e, m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expint::expn;
    use crate::legendre::legendre;
    use crate::quadrature::GaussLegendre;
    use crate::testing::*;

    #[test]
    fn d_for_order_one() {
        let t = TransferTensor::new(1);
        // P0P0 = 1, P0P1 = m, P1P1 = m^2
        assert_eq!(t.d(0, 0, 0), 1.0);
        assert_eq!(t.d(0, 1, 1), 1.0);
        assert_eq!(t.d(1, 0, 1), 1.0);
        assert_eq!(t.d(1, 1, 2), 1.0);
        assert_eq!(t.d(1, 1, 0), 0.0);
    }

    #[test]
    fn order_one_matrix_is_e1_e2_e3() {
        let t = TransferTensor::new(1);
        let (mut e, mut m) = (vec![0.0; 3], vec![0.0; 4]);
        let tau = 0.7;
        t.matrix(tau, &mut e, &mut m);
        assert_eq!(
            m,
            vec![expn(1, tau), expn(2, tau), expn(2, tau), expn(3, tau)]
        );
    }

    #[test]
    fn matrix_is_symmetric() {
        let t = TransferTensor::new(4);
        let (mut e, mut m) = (vec![0.0; 9], vec![0.0; 25]);
        t.matrix(0.3, &mut e, &mut m);
        for l in 0..5 {
            for lp in 0..5 {
                assert_eq!(m[l * 5 + lp], m[lp * 5 + l]);
            }
        }
    }

    #[test]
    fn matches_direct_quadrature() {
        // For τ ≳ 0.5 the integrand vanishes smoothly at m = 0, so plain
        // Gauss-Legendre on (0, 1) converges fast.
        let max_order = 5;
        let t = TransferTensor::new(max_order);
        let n = max_order + 1;
        let (mut e, mut m) = (vec![0.0; t.n_expn()], vec![0.0; n * n]);
        let gl = GaussLegendre::new(200);
        for &tau in &[0.5, 1.0, 4.0] {
            t.matrix(tau, &mut e, &mut m);
            for l in 0..n {
                for lp in 0..n {
                    let f = |x: f64| {
                        let mu = 0.5 * (x + 1.0); // map [-1,1] -> [0,1]
                        0.5 * legendre(l, mu) * legendre(lp, mu) * (-tau / mu).exp() / mu
                    };
                    let q = gl.integrate(f);
                    let got = m[l * n + lp];
                    assert!(
                        is_close(got, q, Tol::NUMERICAL),
                        "l={l} l'={lp} τ={tau}: {got} vs {q}"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_depth_diverges_only_for_even_even() {
        // P_l(0) P_l'(0) ≠ 0 exactly when l and l' are both even; only then
        // does the E_1 log pole survive. Nothing may be NaN.
        let t = TransferTensor::new(3);
        let (mut e, mut m) = (vec![0.0; 7], vec![0.0; 16]);
        t.matrix(0.0, &mut e, &mut m);
        for l in 0..4 {
            for lp in 0..4 {
                let v = m[l * 4 + lp];
                assert!(!v.is_nan());
                assert_eq!(v.is_infinite(), l % 2 == 0 && lp % 2 == 0, "l={l} l'={lp}");
            }
        }
        assert_eq!(m[1], 1.0); // M_01(0) = E_2(0) = 1
    }
}
