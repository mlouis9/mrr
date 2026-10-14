//! Exponential integrals E_n(x) = ∫_1^∞ e^{-x t} / t^n dt
//!
//! In slab geometry the angle-integrated free-flight kernel reduces to E_n
//! functions of the optical depth (E_1(τ)/2 for an isotropic plane source).

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const EPS: f64 = 1e-16;
const MAX_ITER: usize = 1000;

/// E_n(x) for integer n >= 0 and x >= 0.
///
/// Uses a power series for x <= 1 and a continued fraction (modified Lentz)
/// for x > 1, the standard split for double precision.
///
/// Special cases: E_0(x) = e^{-x}/x; E_n(0) = 1/(n-1) for n >= 2;
/// E_0(0) and E_1(0) are infinite.
pub fn expn(n: u32, x: f64) -> f64 {
    assert!(x >= 0.0, "E_n(x) requires x >= 0, got {x}");
    if n == 0 {
        return (-x).exp() / x;
    }
    if x == 0.0 {
        return if n == 1 {
            f64::INFINITY
        } else {
            1.0 / (n as f64 - 1.0)
        };
    }
    if x > 1.0 {
        continued_fraction(n, x)
    } else {
        series(n, x)
    }
}

/// Continued fraction for x > 1, evaluated with the modified Lentz method
fn continued_fraction(n: u32, x: f64) -> f64 {
    let nm1 = n as f64 - 1.0;
    let mut b = x + n as f64;
    let mut c = 1.0 / f64::MIN_POSITIVE;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITER {
        let fi = i as f64;
        let a = -fi * (nm1 + fi);
        b += 2.0;
        d = 1.0 / (a * d + b);
        c = b + a / c;
        let del = c * d;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return h * (-x).exp();
        }
    }
    panic!("E_{n}({x}): continued fraction did not converge");
}

/// Power series for 0 < x <= 1
fn series(n: u32, x: f64) -> f64 {
    let nm1 = n as i64 - 1;
    // Leading term: 1/(n-1) for n >= 2, or -ln(x) - gamma for n = 1
    let mut ans = if nm1 != 0 {
        1.0 / nm1 as f64
    } else {
        -x.ln() - EULER_GAMMA
    };
    let mut fact = 1.0;
    for i in 1..=MAX_ITER as i64 {
        fact *= -x / i as f64;
        let del = if i != nm1 {
            -fact / (i - nm1) as f64
        } else {
            // psi(n) = -gamma + sum_{k=1}^{n-1} 1/k
            let psi = -EULER_GAMMA + (1..=nm1).map(|k| 1.0 / k as f64).sum::<f64>();
            fact * (-x.ln() + psi)
        };
        ans += del;
        if del.abs() < ans.abs() * EPS {
            return ans;
        }
    }
    panic!("E_{n}({x}): series did not converge");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rtol: f64) -> bool {
        (a - b).abs() <= rtol * b.abs()
    }

    #[test]
    fn matches_tabulated_values() {
        // Reference values (Abramowitz & Stegun Table 5.1 / high-precision evaluation)
        let cases = [
            (1, 0.1, 1.822_923_958_419_39),
            (1, 0.5, 0.559_773_594_776_160_8),
            (1, 1.0, 0.219_383_934_395_520_3),
            (1, 2.0, 0.048_900_510_708_061_12),
            (1, 5.0, 0.001_148_295_591_275_326),
            (2, 1.0, 0.148_495_506_775_922_1),
            (3, 1.0, 0.109_691_967_197_760_1),
            (2, 0.5, 0.326_643_862_324_553),
        ];
        for (n, x, expected) in cases {
            assert!(
                close(expn(n, x), expected, 1e-12),
                "E_{n}({x}) = {}",
                expn(n, x)
            );
        }
    }

    #[test]
    fn recurrence_holds_on_both_sides_of_the_switch() {
        // n E_{n+1}(x) = e^{-x} - x E_n(x)
        for &x in &[0.01, 0.3, 0.99, 1.0, 1.01, 3.0, 20.0] {
            for n in 1..8 {
                let lhs = n as f64 * expn(n + 1, x);
                let rhs = (-x).exp() - x * expn(n, x);
                assert!(
                    (lhs - rhs).abs() <= 1e-13 + 1e-11 * rhs.abs(),
                    "n={n}, x={x}"
                );
            }
        }
    }

    #[test]
    fn special_cases() {
        assert_eq!(expn(1, 0.0), f64::INFINITY);
        assert!(close(expn(2, 0.0), 1.0, 1e-15));
        assert!(close(expn(3, 0.0), 0.5, 1e-15));
        assert!(close(expn(0, 2.0), (-2.0f64).exp() / 2.0, 1e-15));
    }

    #[test]
    fn continuous_across_the_series_fraction_switch() {
        for n in 1..6 {
            let below = expn(n, 1.0 - 1e-9);
            let above = expn(n, 1.0 + 1e-9);
            assert!(close(below, above, 1e-7), "n={n}");
        }
    }

    #[test]
    fn decreasing_in_x() {
        for n in 1..5 {
            let xs = [0.05, 0.5, 1.0, 2.0, 10.0];
            assert!(xs.windows(2).all(|w| expn(n, w[0]) > expn(n, w[1])));
        }
    }

    #[test]
    #[should_panic(expected = "requires x >= 0")]
    fn negative_argument_panics() {
        expn(1, -1.0);
    }
}
