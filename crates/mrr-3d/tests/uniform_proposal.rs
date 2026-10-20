//! The exact gather under the uniform proposal: unbiased, but with an
//! infinite-variance (heavy-tailed) node estimator in 3D.
//!
//! # What is tested and why
//!
//! A node `x` placed uniformly scores `w φ_00(x)`. Near an atom `φ ~ 1/r²`, so
//! `P(score > v) ~ v^{−3/2}`: the mean is finite (unbiased) but the variance
//! is infinite (§8.1). Tests therefore never use sample-variance error bars on
//! raw scores. Instead:
//!
//! 1. **Unbiasedness, exactly.** In a reflective box the image sum is the
//!    infinite-medium kernel truncated at path length `R`, so for *any* atoms
//!    the flux integral is `∫ φ_00 = Φ(Q) (1 − e^{−Σ_t R})/Σ_t`. Subtracting the
//!    gather at a small cutoff `ε` removes the pole exactly: the score
//!    `w (φ_R − φ_ε)(x)` has expectation `Φ (e^{−Σε} − e^{−ΣR})/Σ` and bounded
//!    variance, so a CLT test is valid. This checks gather + images + walk +
//!    proposal weights together.
//! 2. **The tail itself.** Raw scores have tail index ≈ 3/2 (Hill estimator),
//!    regularized ones are bounded. This is the before/after evidence that the
//!    pole-matched proposal must change.
//! 3. **Power iteration** in an infinite medium (reflective homogeneous box),
//!    where `k_R = νΣ_f P / (Σ_t − Σ_s P)` exactly. The multiplicative k update
//!    survives the heavy tail; typical runs sit a few percent below `k_R`
//!    (the mean is carried by rare spikes), which the slow test documents.

use mrr_3d::problems::{homogeneous_box, one_group_material};
use mrr_3d::{ExactGather, HarmonicEmission};
use mrr_core::convergence::MaxCycles;
use mrr_core::proposal::UniformProposal;
use mrr_core::sampler::{RandomStreams, purpose};
use mrr_core::traits::{Gather, Proposal, Source, Targets};
use mrr_core::{Atoms, Geometry, KUpdate, Problem, RunConfig, power_iterate_with};
use mrr_csg::Model;
use mrr_geom3d::Bc;
use mrr_xs::{EnergyGroups, MaterialLibrary};

/// One-group library with a single material
fn library(sigma_t: f64, sigma_s: f64, nu_sigma_f: f64) -> MaterialLibrary {
    let mut lib = MaterialLibrary::new(EnergyGroups::new(vec![2e7, 1e-5]).unwrap(), 0);
    lib.add(one_group_material("m", sigma_t, sigma_s, nu_sigma_f))
        .unwrap();
    lib
}

/// `n` uniformly placed isotropic atoms with unit moment and weight V/n (so Φ = V)
fn uniform_atoms(
    model: &Model,
    lib: &MaterialLibrary,
    n: usize,
    seed: u64,
) -> Atoms<mrr_geom3d::Vec3> {
    let streams = RandomStreams {
        base_seed: seed,
        purpose: purpose::INIT,
        cycle: 0,
    };
    let Targets { points, weights } =
        UniformProposal.sample_target_atoms(model, &Source::Initialize(n), &streams);
    Atoms::from_targets(points, weights, lib.harmonic_layout(), &[1.0])
}

/// For `reps` replicas of `m` uniform nodes, `score(node) · weight` summed per replica
fn replicate(
    model: &Model,
    m: usize,
    reps: usize,
    mut score: impl FnMut(mrr_geom3d::Vec3) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut replica_sums = Vec::with_capacity(reps);
    let mut node_scores = Vec::with_capacity(reps * m);
    for r in 0..reps {
        let streams = RandomStreams {
            base_seed: 7,
            purpose: purpose::PROPOSAL,
            cycle: r,
        };
        let Targets { points, weights } =
            UniformProposal.sample_target_atoms(model, &Source::Initialize(m), &streams);
        let mut sum = 0.0;
        for (p, w) in points.iter().zip(&weights) {
            let s = w * score(*p);
            node_scores.push(s);
            sum += s;
        }
        replica_sums.push(sum);
    }
    (replica_sums, node_scores)
}

fn mean_and_std_error(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// Hill estimator of the tail index from the `k` largest of `xs`
fn hill_tail_index(xs: &[f64], k: usize) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let xk = sorted[k];
    let h = sorted[..k].iter().map(|x| (x / xk).ln()).sum::<f64>() / k as f64;
    1.0 / h
}

// Shared setup: reflective unit-half-width box (V = 8), pure absorber Σ_t = 1
const R: f64 = 2.0;
const EPS: f64 = 0.25;

#[test]
fn uniform_nodes_estimate_the_flux_integral_without_bias() {
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let atoms = uniform_atoms(&model, &lib, 16, 99);
    let (far, near) = (ExactGather::new(&lib, R), ExactGather::new(&lib, EPS));
    let (sums, _) = replicate(&model, 32, 150, |x| {
        far.gather(&model, &atoms, x)[0] - near.gather(&model, &atoms, x)[0]
    });
    let exact = model.volume() * ((-EPS).exp() - (-R).exp());
    let (mean, se) = mean_and_std_error(&sums);
    let z = (mean - exact) / se;
    assert!(
        z.abs() < 4.0,
        "shell-regularized estimate {mean} vs exact {exact} (z = {z:.2})"
    );
}

#[test]
fn node_scores_have_the_predicted_heavy_tail() {
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let atoms = uniform_atoms(&model, &lib, 16, 99);
    let (far, near) = (ExactGather::new(&lib, R), ExactGather::new(&lib, EPS));
    let (_, raw) = replicate(&model, 32, 150, |x| far.gather(&model, &atoms, x)[0]);
    let (_, regular) = replicate(&model, 32, 150, |x| {
        far.gather(&model, &atoms, x)[0] - near.gather(&model, &atoms, x)[0]
    });

    // Raw scores: power-law tail with index ≈ 3/2 (finite mean, infinite variance)
    let alpha = hill_tail_index(&raw, 100);
    assert!(
        (1.2..2.0).contains(&alpha),
        "tail index {alpha:.2}, expected ≈ 1.5"
    );

    // Regularized scores are bounded: each atom contributes at most
    // w_atom /(4π ε²) to φ, times the node weight V/m
    let node_weight = model.volume() / 32.0;
    let bound =
        node_weight * 16.0 * (model.volume() / 16.0) / (4.0 * std::f64::consts::PI * EPS * EPS);
    let max = regular.iter().cloned().fold(0.0, f64::max);
    assert!(
        max <= bound,
        "regularized score {max} above its bound {bound}"
    );
    // ...while raw scores exceed that bound (the pole)
    assert!(raw.iter().cloned().fold(0.0, f64::max) > max);
}

/// k_R for the homogeneous reflective box with path cutoff R
fn k_truncated(sigma_t: f64, sigma_s: f64, nu_sigma_f: f64, r: f64) -> f64 {
    let p = 1.0 - (-sigma_t * r).exp();
    nu_sigma_f * p / (sigma_t - sigma_s * p)
}

fn run_reflective_box(n: usize, cycles: usize, seed: u64, r: f64) -> Vec<f64> {
    let lib = library(1.0, 0.5, 0.6);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let problem = Problem::new(
        model,
        lib.harmonic_layout(),
        ExactGather::new(&lib, r),
        HarmonicEmission::new(&lib),
        UniformProposal,
    );
    let cfg = RunConfig {
        rules: vec![Box::new(MaxCycles::new(cycles))],
        n_source_atoms: n,
        seed,
        k_update: KUpdate::Multiplicative { initial: 1.0 },
    };
    let mut ks = Vec::new();
    power_iterate_with(&problem, &cfg, |s| ks.extend(s.k));
    ks
}

#[test]
fn power_iteration_survives_the_heavy_tail() {
    // Smoke test: the multiplicative update keeps k positive and finite and
    // in the right range despite node spikes
    let ks = run_reflective_box(40, 8, 1, R);
    let k_r = k_truncated(1.0, 0.5, 0.6, R);
    assert_eq!(ks.len(), 8);
    assert!(ks.iter().all(|k| k.is_finite() && *k > 0.0));
    let mut sorted = ks.clone();
    sorted.sort_by(f64::total_cmp);
    let median = 0.5 * (sorted[3] + sorted[4]);
    assert!(
        (0.75 * k_r..1.25 * k_r).contains(&median),
        "median k {median:.3} far from k_R = {k_r:.3}"
    );
}

/// Slow (run with `cargo test --release -p mrr-3d -- --ignored`): typical runs
/// sit a few percent *below* k_R. Not a bias of the gather (test 1 shows it
/// is unbiased) but of the nonlinear eigenvalue estimate under heavy-tailed
/// node values: the mean is carried by rare spikes that most runs miss.
#[test]
#[ignore]
fn power_iteration_k_in_an_infinite_medium() {
    let r = 4.0;
    let k_r = k_truncated(1.0, 0.5, 0.6, r);
    let mut means: Vec<f64> = (0..8)
        .map(|seed| {
            let ks = run_reflective_box(100, 25, seed, r);
            ks[5..].iter().sum::<f64>() / 20.0
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let median = 0.5 * (means[3] + means[4]);
    println!("k_R = {k_r:.4}, median over seeds = {median:.4}, all = {means:?}");
    assert!(
        (0.9 * k_r..1.02 * k_r).contains(&median),
        "median {median:.4} outside the expected heavy-tail band below k_R = {k_r:.4}"
    );
}

#[test]
fn uniform_proposal_covers_the_box() {
    // Basic contract in 3D: n points inside the box, weights V/n
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.5, Bc::Vacuum);
    let atoms = uniform_atoms(&model, &lib, 500, 3);
    assert_eq!(atoms.len(), 500);
    assert!(atoms.points().iter().all(|p| model.boundary().contains(*p)));
    let total: f64 = atoms.weights().iter().sum();
    assert!((total - model.volume()).abs() < 1e-9);
    // The mean position is near the centre (std of the mean ≈ 1.5/√(3·500))
    let mean = atoms.points().iter().copied().sum::<mrr_geom3d::Vec3>() / 500.0;
    assert!(mean.length() < 0.25, "{mean:?}");
    let _ = model.volume();
}
