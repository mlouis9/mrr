//! Acceptance tests against the Sood analytical benchmarks.
//!
//! These are statistical: k from one cycle is a ratio estimator with
//! O(1/N) bias and cycle-to-cycle correlation. With N = 200 atoms and the
//! mean over 30 active cycles, the spread of the mean across seeds was
//! measured at σ ≈ 0.005 (release build, seeds 0–4), so a tolerance of 0.02
//! is ≈ 4σ. Seeds are fixed, so the tests are deterministic.

use mrr_core::convergence::MaxCycles;
use mrr_core::{RunConfig, power_iterate_with};
use mrr_slab_1d::problems::SlabProblem;
use mrr_slab_1d::problems::benchmarks::{PUA_1_0_SL_K, pua, pua_1_0_sl, pua_slab};

const N_ATOMS: usize = 200;
const INACTIVE: usize = 10;
const ACTIVE: usize = 30;

/// Mean k over the active cycles
fn mean_k(problem: &SlabProblem, seed: u64) -> f64 {
    let cfg = RunConfig {
        rules: vec![Box::new(MaxCycles::new(INACTIVE + ACTIVE))],
        n_source_atoms: N_ATOMS,
        seed,
    };
    let mut ks = Vec::new();
    power_iterate_with(&problem.problem(), &cfg, |snap| {
        if snap.cycle > INACTIVE {
            ks.push(snap.k.expect("k exists after cycle 0"));
        }
    });
    assert_eq!(ks.len(), ACTIVE);
    ks.iter().sum::<f64>() / ks.len() as f64
}

#[test]
fn pua_1_0_sl_is_critical() {
    let k = mean_k(&pua_1_0_sl(), 1);
    assert!(
        (k - PUA_1_0_SL_K).abs() < 0.02,
        "PUa-1-0-SL: k = {k}, reference {PUA_1_0_SL_K}"
    );
}

#[test]
fn thinner_and_thicker_slabs_bracket_criticality() {
    let critical = 2.0 * pua::CRITICAL_HALF_WIDTH;
    let thin = mean_k(&pua_slab(0.5 * critical), 2);
    let thick = mean_k(&pua_slab(3.0 * critical), 3);
    assert!(
        thin < 0.9,
        "half-critical slab should be clearly subcritical: {thin}"
    );
    assert!(
        thick > 1.2,
        "3x critical slab should be clearly supercritical: {thick}"
    );
    assert!(
        thick < pua::K_INF,
        "a finite slab leaks, so k < k_inf: {thick}"
    );
}
