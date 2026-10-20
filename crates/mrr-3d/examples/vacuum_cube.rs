//! `cargo run --release -p mrr-3d --example vacuum_cube`
//!
//! Meshless k for the bare cube at several N (several seeds each) against
//! the analog Monte Carlo reference.
use mrr_3d::vacuum_cube::{CUBE, MeshlessSettings, active_mean, meshless_k, reference};

fn main() {
    let r = reference();
    println!("analog reference: k = {:.5} ± {:.5}", r.k, r.std_error);
    println!(
        "{:>6} {:>10} {:>10} {:>10}",
        "N", "mean k", "std err", "rel diff"
    );
    for n in [50, 100, 200, 400] {
        let seeds = 8;
        let ks: Vec<f64> = (0..seeds)
            .map(|seed| {
                let s = MeshlessSettings {
                    n,
                    seed,
                    ..Default::default()
                };
                active_mean(&meshless_k(&CUBE, &s), s.inactive)
            })
            .collect();
        let mean = ks.iter().sum::<f64>() / seeds as f64;
        let var = ks.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / (seeds as f64 - 1.0);
        let se = (var / seeds as f64).sqrt();
        println!(
            "{n:>6} {mean:>10.5} {se:>10.5} {:>+10.4}",
            (mean - r.k) / r.k
        );
    }
}
