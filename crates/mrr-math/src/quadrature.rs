//! Gauss-Legendre quadrature on [-1, 1]

use crate::legendre::legendre;

/// Nodes and weights of an n-point Gauss-Legendre rule on [-1, 1].
/// Integrates polynomials of degree <= 2n - 1 exactly.
#[derive(Debug, Clone)]
pub struct GaussLegendre {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl GaussLegendre {
    /// Build the rule by Newton iteration on the roots of P_n.
    /// Nodes are returned in increasing order.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "a quadrature rule needs at least one point");
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        let nf = n as f64;
        for i in 0..n.div_ceil(2) {
            // Standard initial guess for the i-th largest root
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
            let mut dp = 0.0;
            for _ in 0..100 {
                let p = legendre(n, x);
                // P_n'(x) = n (x P_n - P_{n-1}) / (x^2 - 1)
                dp = nf * (x * p - legendre(n - 1, x)) / (x * x - 1.0);
                let dx = p / dp;
                x -= dx;
                if dx.abs() < 1e-15 {
                    break;
                }
            }
            let w = 2.0 / ((1.0 - x * x) * dp * dp);
            // Symmetric pair: -x at the low end, +x at the high end
            nodes[i] = -x;
            nodes[n - 1 - i] = x;
            weights[i] = w;
            weights[n - 1 - i] = w;
        }
        Self { nodes, weights }
    }

    /// Integral of `f` over [-1, 1] with this rule
    pub fn integrate(&self, f: impl Fn(f64) -> f64) -> f64 {
        self.nodes
            .iter()
            .zip(&self.weights)
            .map(|(&x, &w)| w * f(x))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legendre::legendre;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-13 + 1e-12 * b.abs()
    }

    #[test]
    fn weights_sum_to_two_and_nodes_are_symmetric() {
        for n in 1..=16 {
            let q = GaussLegendre::new(n);
            assert!(close(q.weights.iter().sum(), 2.0));
            for i in 0..n {
                assert!(close(q.nodes[i], -q.nodes[n - 1 - i]));
            }
            assert!(q.nodes.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn integrates_polynomials_up_to_degree_2n_minus_1_exactly() {
        for n in 1..=10 {
            let q = GaussLegendre::new(n);
            for k in 0..2 * n {
                let exact = if k % 2 == 1 {
                    0.0
                } else {
                    2.0 / (k as f64 + 1.0)
                };
                assert!(
                    close(q.integrate(|x| x.powi(k as i32)), exact),
                    "n={n}, k={k}"
                );
            }
        }
    }

    #[test]
    fn known_two_point_rule() {
        let q = GaussLegendre::new(2);
        assert!(close(q.nodes[1], 1.0 / 3.0_f64.sqrt()));
        assert!(close(q.weights[0], 1.0));
    }

    #[test]
    fn legendre_polynomials_are_orthogonal() {
        let q = GaussLegendre::new(12);
        for l in 0..8 {
            for m in 0..8 {
                let integral = q.integrate(|x| legendre(l, x) * legendre(m, x));
                let expected = if l == m {
                    2.0 / (2.0 * l as f64 + 1.0)
                } else {
                    0.0
                };
                assert!(close(integral, expected), "l={l}, m={m}");
            }
        }
    }
}
