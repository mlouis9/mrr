//! The global mixture proposal end to end.
//!
//! The uniform-proposal tests needed the shell trick `φ_R − φ_ε` because raw
//! scores had infinite variance. With the pole-matched mixture they do not:
//! every test here uses **raw** scores and ordinary CLT error bars.
//!
//! 1. `Σ w` estimates `V` (the weights are a correct `1/(N p)` for the
//!    sampler's actual density, including zero samples in vacuum).
//! 2. `Σ w 1_B` estimates `|B|` for a small ball `B` around an atom, where
//!    almost all nodes come from the pole: checks `π` itself.
//! 3. `Σ w φ_00` estimates `Φ P/Σ_t` exactly in a reflective box.
//! 4. Scores are bounded (tail no longer ≈ 3/2) compared with uniform nodes.
//! 5. Power iteration converges to `k_R` within a CLT band.

use std::f64::consts::PI;

use mrr_3d::problems::{homogeneous_box, one_group_material};
use mrr_3d::{ExactGather, HarmonicEmission, MixtureProposal};
use mrr_core::convergence::MaxCycles;
use mrr_core::proposal::UniformProposal;
use mrr_core::sampler::{RandomStreams, purpose};
use mrr_core::traits::{Gather, Proposal, Source, Targets};
use mrr_core::{Atoms, Geometry, KUpdate, Problem, RunConfig, power_iterate_with};
use mrr_csg::Model;
use mrr_geom3d::{Bc, Vec3};
use mrr_xs::{EnergyGroups, MaterialLibrary};

fn library(sigma_t: f64, sigma_s: f64, nu_sigma_f: f64) -> MaterialLibrary {
    let mut lib = MaterialLibrary::new(EnergyGroups::new(vec![2e7, 1e-5]).unwrap(), 0);
    lib.add(one_group_material("m", sigma_t, sigma_s, nu_sigma_f))
        .unwrap();
    lib
}

/// `n` uniform isotropic atoms, unit moment, weight V/n (Φ = V)
fn uniform_atoms(model: &Model, lib: &MaterialLibrary, n: usize, seed: u64) -> Atoms<Vec3> {
    let streams = RandomStreams {
        base_seed: seed,
        purpose: purpose::INIT,
        cycle: 0,
    };
    let Targets { points, weights } =
        UniformProposal.sample_target_atoms(model, &Source::Initialize(n), &streams);
    Atoms::from_targets(points, weights, lib.harmonic_layout(), &[1.0])
}

/// Per-replica sums of `w · score(x)` and all per-node terms
fn replicate<P: Proposal<Model>>(
    proposal: &P,
    model: &Model,
    atoms: &Atoms<Vec3>,
    reps: usize,
    mut score: impl FnMut(Vec3) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut sums = Vec::with_capacity(reps);
    let mut terms = Vec::new();
    for r in 0..reps {
        let streams = RandomStreams {
            base_seed: 5,
            purpose: purpose::PROPOSAL,
            cycle: r,
        };
        let Targets { points, weights } =
            proposal.sample_target_atoms(model, &Source::Existing(atoms, atoms.len()), &streams);
        let mut sum = 0.0;
        for (x, w) in points.iter().zip(&weights) {
            let t = w * score(*x);
            terms.push(t);
            sum += t;
        }
        sums.push(sum);
    }
    (sums, terms)
}

fn mean_and_std_error(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

fn assert_z(name: &str, xs: &[f64], exact: f64) {
    let (mean, se) = mean_and_std_error(xs);
    let z = (mean - exact) / se;
    assert!(
        z.abs() < 4.0,
        "{name}: {mean} ± {se} vs {exact} (z = {z:.2})"
    );
}

#[test]
fn weights_estimate_the_volume() {
    let lib = library(1.0, 0.0, 0.0);
    for bc in [Bc::Reflective, Bc::Vacuum] {
        let model = homogeneous_box(1.0, bc);
        let atoms = uniform_atoms(&model, &lib, 16, 1);
        let prop = MixtureProposal::new(0.3, 0.6);
        let (sums, _) = replicate(&prop, &model, &atoms, 200, |_| 1.0);
        assert_z(&format!("Σw ({bc:?})"), &sums, model.volume());
    }
}

#[test]
fn weights_estimate_a_small_ball_around_an_atom() {
    // Ball of radius 0.15 around an interior atom: nearly all nodes there
    // come from the pole, so this tests π(x) where it dominates
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let atoms = uniform_atoms(&model, &lib, 16, 2);
    let centre = *atoms
        .points()
        .iter()
        .min_by(|a, b| a.length().total_cmp(&b.length()))
        .unwrap();
    let rho = 0.15;
    assert!(
        centre.abs().max_element() < 1.0 - rho,
        "ball must be inside the box"
    );
    let prop = MixtureProposal::new(0.3, 0.6);
    let (sums, _) = replicate(&prop, &model, &atoms, 200, |x| {
        f64::from((x - centre).length() < rho)
    });
    assert_z("ball", &sums, 4.0 / 3.0 * PI * rho.powi(3));
}

const R: f64 = 2.0;

#[test]
fn raw_scores_estimate_the_flux_integral() {
    // Reflective box, pure absorber: ∫ φ_00 = Φ (1 − e^{−R}) with Φ = V.
    // Uniform nodes would need the ε-shell here; the mixture does not.
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let atoms = uniform_atoms(&model, &lib, 16, 3);
    let gather = ExactGather::new(&lib, R);
    let prop = MixtureProposal::new(0.3, 0.6);
    let (sums, _) = replicate(&prop, &model, &atoms, 100, |x| {
        gather.gather(&model, &atoms, x)[0]
    });
    assert_z("∫φ", &sums, model.volume() * (1.0 - (-R).exp()));
}

#[test]
fn scores_are_bounded_unlike_uniform_nodes() {
    // Same number of nodes per replica for both proposals
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let atoms = uniform_atoms(&model, &lib, 16, 4);
    let gather = ExactGather::new(&lib, R);
    let score = |x| gather.gather(&model, &atoms, x)[0];
    let (_, mix) = replicate(&MixtureProposal::new(0.3, 0.6), &model, &atoms, 60, score);
    let (_, uni) = replicate(&UniformProposal, &model, &atoms, 60, score);

    // Bound: near atom a, φ ≲ w_a/(4π r²) + smooth, w ≤ 1/(N_p a_a/(4π r_c r²)),
    // so w φ ≲ w_a r_c /(N_p a_a) + V/N_s · φ_smooth. Check max/mean instead
    // of the constant: the mixture's ratio is O(1–10), uniform's is huge.
    let ratio = |xs: &[f64]| {
        let (m, _) = mean_and_std_error(xs);
        xs.iter().cloned().fold(0.0, f64::max) / m
    };
    let (rm, ru) = (ratio(&mix), ratio(&uni));
    assert!(rm < 30.0, "mixture max/mean = {rm}");
    assert!(ru > 3.0 * rm, "uniform max/mean = {ru} vs mixture {rm}");
}

fn k_truncated(sigma_t: f64, sigma_s: f64, nu_sigma_f: f64, r: f64) -> f64 {
    let p = 1.0 - (-sigma_t * r).exp();
    nu_sigma_f * p / (sigma_t - sigma_s * p)
}

#[test]
fn power_iteration_converges_to_k_r() {
    let lib = library(1.0, 0.5, 0.6);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let n = 60;
    let r_c = MixtureProposal::suggested_radius(model.volume(), n, 2.0);
    let problem = Problem::new(
        model,
        lib.harmonic_layout(),
        ExactGather::new(&lib, R),
        HarmonicEmission::new(&lib),
        MixtureProposal::new(0.3, r_c),
    );
    let cfg = RunConfig {
        rules: vec![Box::new(MaxCycles::new(30))],
        n_source_atoms: n,
        seed: 9,
        k_update: KUpdate::Split,
    };
    let mut ks = Vec::new();
    power_iterate_with(&problem, &cfg, |s| ks.extend(s.k));
    let active = &ks[10..];
    let (mean, se) = mean_and_std_error(active);
    let k_r = k_truncated(1.0, 0.5, 0.6, R);
    // Cycles are correlated: allow a generous 6 naive standard errors + 2%
    assert!(
        (mean - k_r).abs() < 6.0 * se + 0.02 * k_r,
        "k = {mean:.4} ± {se:.4} vs k_R = {k_r:.4}"
    );
}

// ---------------------------------------------------------------------------
// Finite variance
// ---------------------------------------------------------------------------

/// Pointwise bound on one node's term `w(x) φ_00(x)` (isotropic unit atoms,
/// pure absorber, path cutoff `R`). For each (atom, image) pair at path
/// length `r`, using `e^{−τ} ≤ 1` and each piece of the mixture density on
/// its own (`N p ≥ N_s/V` and, inside the pole, `N p ≥ N_p a_a/(4π r_c r²)`):
///
/// ```text
/// r <  r_c:  w_a/(4π r²) / (N p)  ≤  r_c Φ / N_p          (the pole cancels)
/// r >= r_c:  w_a/(4π r²) / (N p)  ≤  w_a V / (4π r² N_s)  ≤ w_a V / (4π r_c² N_s)
/// ```
///
/// Both are finite, and the number of pairs within `R` is finite, so the
/// node term is bounded: the estimator has finite variance (all moments).
fn node_term_bound(
    model: &Model,
    atoms: &Atoms<Vec3>,
    r_c: f64,
    n_s: usize,
    n_p: usize,
    x: Vec3,
) -> f64 {
    let bbox = model.boundary();
    let phi: f64 = atoms.weights().iter().map(|w| w.abs()).sum();
    let mut images = Vec::new();
    bbox.images_within(x, R, &mut images);
    let mut bound = 0.0;
    for &t in &images {
        let y = bbox.image(t, x);
        for (x_a, w_a) in atoms.points().iter().zip(atoms.weights()) {
            let r = (y - *x_a).length();
            if r >= R {
                continue;
            }
            bound += if r < r_c {
                r_c * phi / n_p as f64
            } else {
                w_a * model.volume() / (4.0 * PI * r * r * n_s as f64)
            };
        }
    }
    bound
}

#[test]
fn every_node_term_is_below_its_analytic_bound() {
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let atoms = uniform_atoms(&model, &lib, 16, 6);
    let gather = ExactGather::new(&lib, R);
    let (alpha, r_c) = (0.3, 0.6);
    let prop = MixtureProposal::new(alpha, r_c);
    let (n_s, n_p) = prop.split_counts(atoms.len());
    for cycle in 0..40 {
        let streams = RandomStreams {
            base_seed: 8,
            purpose: purpose::PROPOSAL,
            cycle,
        };
        let Targets { points, weights } =
            prop.sample_target_atoms(&model, &Source::Existing(&atoms, atoms.len()), &streams);
        for (x, w) in points.iter().zip(&weights) {
            let term = w * gather.gather(&model, &atoms, *x)[0];
            let bound = node_term_bound(&model, &atoms, r_c, n_s, n_p, *x);
            assert!(
                term <= bound * (1.0 + 1e-12),
                "term {term} above bound {bound} at {x:?}"
            );
        }
    }
}

/// Hill estimator of the tail index from the `k` largest of `xs`
fn hill_tail_index(xs: &[f64], k: usize) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let xk = sorted[k];
    let h = sorted[..k].iter().map(|x| (x / xk).ln()).sum::<f64>() / k as f64;
    1.0 / h
}

#[test]
fn the_tail_index_moves_past_two() {
    // Uniform nodes: ≈ 3/2 (infinite variance). Mixture: bounded, so the
    // estimated index is well above 2 (finite variance)
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let atoms = uniform_atoms(&model, &lib, 16, 7);
    let gather = ExactGather::new(&lib, R);
    let score = |x| gather.gather(&model, &atoms, x)[0];
    let (_, mix) = replicate(&MixtureProposal::new(0.3, 0.6), &model, &atoms, 150, score);
    let (_, uni) = replicate(&UniformProposal, &model, &atoms, 150, score);
    let (a_mix, a_uni) = (hill_tail_index(&mix, 100), hill_tail_index(&uni, 100));
    assert!(a_uni < 2.0, "uniform tail index {a_uni:.2}");
    assert!(a_mix > 3.0, "mixture tail index {a_mix:.2}");
}

#[test]
fn replica_variance_is_stable_as_replicas_grow() {
    // With finite variance the sample variance of replica sums converges;
    // compare the first half of the replicas with all of them
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    let atoms = uniform_atoms(&model, &lib, 16, 10);
    let gather = ExactGather::new(&lib, R);
    let (sums, _) = replicate(&MixtureProposal::new(0.3, 0.6), &model, &atoms, 400, |x| {
        gather.gather(&model, &atoms, x)[0]
    });
    let var = |xs: &[f64]| {
        let (_, se) = mean_and_std_error(xs);
        se * se * xs.len() as f64
    };
    let (v_half, v_all) = (var(&sums[..200]), var(&sums));
    let ratio = v_all / v_half;
    // Sampling error of a variance from 200 values is ≈ √(2/200) ≈ 10% for a
    // light tail; allow 35%
    assert!((ratio - 1.0).abs() < 0.35, "variance {v_half} → {v_all}");
}
