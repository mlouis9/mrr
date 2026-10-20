//! The bare cube: the meshless k agrees with the analog Monte Carlo
//! reference. Vacuum faces exercise zero samples (pole flights that leak)
//! and the infinite-cutoff gather.
use mrr_3d::vacuum_cube::{CUBE, MeshlessSettings, active_mean, meshless_k, reference};

fn seed_means(n: usize, seeds: u64) -> (f64, f64) {
    let ks: Vec<f64> = (0..seeds)
        .map(|seed| {
            let s = MeshlessSettings {
                n,
                seed,
                ..Default::default()
            };
            let ks = meshless_k(&CUBE, &s);
            assert!(ks.iter().all(|k| k.is_finite() && *k > 0.0));
            active_mean(&ks, s.inactive)
        })
        .collect();
    let m = ks.len() as f64;
    let mean = ks.iter().sum::<f64>() / m;
    let var = ks.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / (m - 1.0);
    (mean, (var / m).sqrt())
}

#[test]
fn meshless_k_matches_the_analog_reference() {
    let r = reference();
    let (k, se) = seed_means(120, 4);
    let tol = 4.0 * (se * se + r.std_error * r.std_error).sqrt() + 0.01 * r.k;
    assert!(
        (k - r.k).abs() < tol,
        "meshless {k:.4} ± {se:.4} vs analog {:.4} ± {:.4}",
        r.k,
        r.std_error
    );
}

/// Slow: `cargo test --release -p mrr-3d --test vacuum_cube -- --ignored`
#[test]
#[ignore]
fn meshless_k_converges_with_n() {
    let r = reference();
    let (k, se) = seed_means(400, 8);
    println!(
        "N = 400: {k:.5} ± {se:.5}, reference {:.5} ± {:.5}",
        r.k, r.std_error
    );
    assert!((k - r.k).abs() < 4.0 * (se * se + r.std_error * r.std_error).sqrt() + 0.003 * r.k);
}
