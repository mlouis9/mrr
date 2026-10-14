//! Legendre polynomials P_l(mu)

/// Fill `out[l] = P_l(mu)` for every l in `0..out.len()`, using the
/// three-term recurrence (l+1) P_{l+1} = (2l+1) mu P_l - l P_{l-1}.
///
/// Computing all orders at once costs the same as computing the highest one,
/// and callers almost always need every order up to L.
pub fn legendre_all(mu: f64, out: &mut [f64]) {
    if out.is_empty() {
        return;
    }
    out[0] = 1.0;
    if out.len() == 1 {
        return;
    }
    out[1] = mu;
    for l in 1..out.len() - 1 {
        let lf = l as f64;
        out[l + 1] = ((2.0 * lf + 1.0) * mu * out[l] - lf * out[l - 1]) / (lf + 1.0);
    }
}

/// P_l(mu) for a single order (allocation-free recurrence)
pub fn legendre(l: usize, mu: f64) -> f64 {
    let (mut p_prev, mut p) = (1.0, mu);
    if l == 0 {
        return p_prev;
    }
    for k in 1..l {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * mu * p - kf * p_prev) / (kf + 1.0);
        p_prev = p;
        p = p_next;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-13 + 1e-12 * b.abs()
    }

    #[test]
    fn low_orders_match_closed_forms() {
        for &mu in &[-1.0, -0.3, 0.0, 0.42, 1.0] {
            assert!(close(legendre(0, mu), 1.0));
            assert!(close(legendre(1, mu), mu));
            assert!(close(legendre(2, mu), 0.5 * (3.0 * mu * mu - 1.0)));
            assert!(close(legendre(3, mu), 0.5 * (5.0 * mu.powi(3) - 3.0 * mu)));
        }
    }

    #[test]
    fn value_at_one_and_parity() {
        for l in 0..20 {
            assert!(close(legendre(l, 1.0), 1.0));
            let sign = if l % 2 == 0 { 1.0 } else { -1.0 };
            assert!(close(legendre(l, -0.37), sign * legendre(l, 0.37)));
        }
    }

    #[test]
    fn all_orders_agree_with_single_order() {
        let mut out = [0.0; 12];
        legendre_all(0.61, &mut out);
        for (l, &p) in out.iter().enumerate() {
            assert!(close(p, legendre(l, 0.61)));
        }
    }

    #[test]
    fn short_output_buffers_are_handled() {
        let mut none: [f64; 0] = [];
        legendre_all(0.5, &mut none);
        let mut one = [9.0];
        legendre_all(0.5, &mut one);
        assert_eq!(one, [1.0]);
    }
}
