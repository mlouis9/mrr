//! `cargo run --release -p mrr-3d --example qmc_convergence [quick]`
//!
//! Error vs N for pseudo-random (MC) and Owen-scrambled Sobol (RQMC) node
//! sets, over independent seeds / scrambles. Three experiments, each with an
//! exact answer:
//!
//! 1. **Smooth baseline**: `∫ Π_d (1 + ½ sin 2πu_d + u_d²)` over `[0,1)³`.
//!    Confirms the sampler: MC → N^{-1/2}, RQMC → about N^{-3/2}.
//! 2. **Frozen source**: fixed atoms in a reflective box (pure absorber,
//!    path cutoff R), one node set from the mixture proposal, estimate
//!    `∫ φ_00 = Φ (1 − e^{−Σ_t R})`. The integration rate of the method,
//!    pole matching included.
//!    (2a) with 64 fixed atoms, (2b) with as many atoms as nodes, which is
//!    the regime of power iteration.
//! 3. **k_R**: power iteration in the same box (Σ_s, νΣ_f > 0), where
//!    `k_R = νΣ_f P/(Σ_t − Σ_s P)` exactly. Reports RMSE, |bias| and the
//!    spread across seeds of the active-cycle mean.
//!
//! Slopes are least-squares fits of log RMSE on log N.

use std::f64::consts::PI;

use mrr_3d::problems::{homogeneous_box, one_group_material};
use mrr_3d::{ExactGather, HarmonicEmission, MixtureProposal, PoleProfile};
use mrr_core::convergence::MaxCycles;
use mrr_core::proposal::UniformProposal;
use mrr_core::sampler::{AnyStreams, SamplerStreams, UnitSampler, purpose};
use mrr_core::traits::{Gather, Proposal, Source, Targets};
use mrr_core::{Atoms, Geometry, KUpdate, Problem, RunConfig, SamplerKind, power_iterate_with};
use mrr_geom3d::Bc;
use mrr_xs::{EnergyGroups, MaterialLibrary};

const KINDS: [SamplerKind; 2] = [SamplerKind::Random, SamplerKind::Sobol];

fn library(sigma_t: f64, sigma_s: f64, nu_sigma_f: f64) -> MaterialLibrary {
    let mut lib = MaterialLibrary::new(EnergyGroups::new(vec![2e7, 1e-5]).unwrap(), 0);
    lib.add(one_group_material("m", sigma_t, sigma_s, nu_sigma_f))
        .unwrap();
    lib
}

/// Run `f(seed)` for `seeds` seeds on all cores
fn par_map(seeds: u64, f: impl Fn(u64) -> f64 + Sync) -> Vec<f64> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let mut out = vec![0.0; seeds as usize];
    std::thread::scope(|scope| {
        let f = &f;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    (t..seeds)
                        .step_by(threads as usize)
                        .map(|s| (s, f(s)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (s, v) in h.join().unwrap() {
                out[s as usize] = v;
            }
        }
    });
    out
}

/// (rmse, |bias|, std) of estimates against `exact`
fn errors(est: &[f64], exact: f64) -> (f64, f64, f64) {
    let n = est.len() as f64;
    let mean = est.iter().sum::<f64>() / n;
    let rmse = (est.iter().map(|e| (e - exact).powi(2)).sum::<f64>() / n).sqrt();
    let std = (est.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    (rmse, (mean - exact).abs(), std)
}

/// Least-squares slope of log y on log x
fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let lx: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|y| y.ln()).collect();
    let n = lx.len() as f64;
    let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
    let sxy: f64 = lx.iter().zip(&ly).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = lx.iter().map(|x| (x - mx).powi(2)).sum();
    sxy / sxx
}

/// One configuration of an experiment: label and (rmse, |bias|, std) per N
type Row = (String, Vec<(f64, f64, f64)>);

fn report(title: &str, ns: &[usize], rows: &[Row]) {
    println!("\n## {title}\n");
    print!("{:>6}", "N");
    for (label, _) in rows {
        print!(" | {:>27}", format!("{label}: rmse / std"));
    }
    println!();
    for (i, n) in ns.iter().enumerate() {
        print!("{n:>6}");
        for (_, e) in rows {
            print!(" | {:>13.3e} {:>13.3e}", e[i].0, e[i].2);
        }
        println!();
    }
    let x: Vec<f64> = ns.iter().map(|n| *n as f64).collect();
    print!("{:>6}", "slope");
    for (_, errs) in rows {
        let rmse: Vec<f64> = errs.iter().map(|e| e.0).collect();
        let std: Vec<f64> = errs.iter().map(|e| e.2).collect();
        print!(" | {:>+13.2} {:>+13.2}", slope(&x, &rmse), slope(&x, &std));
    }
    println!();
    let bias: Vec<String> = rows
        .iter()
        .map(|(l, e)| format!("{l} {:.1e}", e.last().unwrap().1))
        .collect();
    println!("|bias| at largest N: {}", bias.join(", "));
}

/// Every (sampler, profile) configuration
fn configs(profiles: &[PoleProfile]) -> Vec<(SamplerKind, PoleProfile, String)> {
    let mut out = Vec::new();
    for &p in profiles {
        for k in KINDS {
            let kind = if k == SamplerKind::Random {
                "MC"
            } else {
                "QMC"
            };
            let label = if profiles.len() > 1 {
                format!("{kind} {p:?}")
            } else {
                kind.to_string()
            };
            out.push((k, p, label));
        }
    }
    out
}

fn smooth_baseline(ns: &[usize], seeds: u64) {
    let f = |u: &[f64]| {
        u.iter()
            .map(|x| 1.0 + 0.5 * (2.0 * PI * x).sin() + x * x)
            .product::<f64>()
    };
    let exact = (4.0f64 / 3.0).powi(3);
    let rows = configs(&[PoleProfile::Uniform])
        .into_iter()
        .map(|(kind, _, label)| {
            let errs = ns
                .iter()
                .map(|&n| {
                    let est = par_map(seeds, |seed| {
                        let mut s = AnyStreams::new(kind, seed, purpose::PROPOSAL, 0).stream(0);
                        let mut u = [0.0; 3];
                        (0..n)
                            .map(|_| {
                                s.next_point(&mut u);
                                f(&u)
                            })
                            .sum::<f64>()
                            / n as f64
                    });
                    errors(&est, exact)
                })
                .collect();
            (label, errs)
        })
        .collect::<Vec<_>>();
    report("1. Smooth 3D integrand (sampler check)", ns, &rows);
}

const R_FROZEN: f64 = 1.0;

/// `n_atoms = None`: as many atoms as nodes (as in power iteration)
fn frozen_source(ns: &[usize], seeds: u64, n_atoms: Option<usize>, profiles: &[PoleProfile]) {
    let lib = library(1.0, 0.0, 0.0);
    let model = homogeneous_box(1.0, Bc::Reflective);
    // Fixed uniform atoms, unit isotropic moment, Φ = V
    let make_atoms = |m: usize| {
        let streams = AnyStreams::new(SamplerKind::Random, 77, purpose::INIT, 0);
        let Targets { points, weights } =
            UniformProposal.sample_target_atoms(&model, &Source::Initialize(m), &streams);
        Atoms::from_targets(points, weights, lib.harmonic_layout(), &[1.0])
    };
    let exact = model.volume() * (1.0 - (-R_FROZEN).exp());
    let gather = ExactGather::new(&lib, R_FROZEN);
    let rows = configs(profiles)
        .into_iter()
        .map(|(kind, profile, label)| {
            let errs = ns
                .iter()
                .map(|&n| {
                    let atoms = make_atoms(n_atoms.unwrap_or(n));
                    // Pole radius tied to the atom spacing, as in a run
                    let r_c = MixtureProposal::suggested_radius(model.volume(), atoms.len(), 2.0);
                    let prop = MixtureProposal::new(0.3, r_c).with_profile(profile);
                    let est = par_map(seeds, |seed| {
                        let st = AnyStreams::new(kind, seed, purpose::PROPOSAL, 0);
                        let t = prop.sample_target_atoms(&model, &Source::Existing(&atoms, n), &st);
                        t.points
                            .iter()
                            .zip(&t.weights)
                            .map(|(x, w)| w * gather.gather(&model, &atoms, *x)[0])
                            .sum::<f64>()
                    });
                    errors(&est, exact)
                })
                .collect();
            (label, errs)
        })
        .collect::<Vec<_>>();
    let title = match n_atoms {
        Some(m) => format!("2a. Frozen source, {m} fixed atoms: ∫φ over a reflective box"),
        None => "2b. Frozen source, as many atoms as nodes: ∫φ over a reflective box".into(),
    };
    report(&title, ns, &rows);
}

const R_K: f64 = 0.5;

/// `fixed_r_c = None`: the pole radius shrinks with N (2 atoms per ball)
fn k_problem(
    ns: &[usize],
    seeds: u64,
    cycles: usize,
    inactive: usize,
    profiles: &[PoleProfile],
    fixed_r_c: Option<f64>,
) {
    let (st, ss, nf) = (1.0, 0.5, 0.6);
    let lib = library(st, ss, nf);
    let p = 1.0 - (-st * R_K).exp();
    let k_r = nf * p / (st - ss * p);
    println!("\nk_R = {k_r:.6} (R = {R_K})");
    let rows = configs(profiles)
        .into_iter()
        .map(|(kind, profile, label)| {
            let errs = ns
                .iter()
                .map(|&n| {
                    let est = par_map(seeds, |seed| {
                        let model = homogeneous_box(1.0, Bc::Reflective);
                        let r_c = fixed_r_c.unwrap_or_else(|| {
                            MixtureProposal::suggested_radius(model.volume(), n, 2.0)
                        });
                        let problem = Problem::new(
                            model,
                            lib.harmonic_layout(),
                            ExactGather::new(&lib, R_K),
                            HarmonicEmission::new(&lib),
                            MixtureProposal::new(0.3, r_c).with_profile(profile),
                        );
                        let cfg = RunConfig {
                            rules: vec![Box::new(MaxCycles::new(cycles))],
                            n_source_atoms: n,
                            seed,
                            k_update: KUpdate::Split,
                            sampler: kind,
                        };
                        let mut ks = Vec::new();
                        power_iterate_with(&problem, &cfg, |s| ks.extend(s.k));
                        ks[inactive..].iter().sum::<f64>() / (cycles - inactive) as f64
                    });
                    errors(&est, k_r)
                })
                .collect();
            (label, errs)
        })
        .collect::<Vec<_>>();
    report(
        &format!(
            "3{}. k_R in a reflective box ({cycles} cycles, {inactive} inactive), r_c {}",
            if fixed_r_c.is_some() { "b" } else { "a" },
            fixed_r_c.map_or("= suggested(N)".to_string(), |r| format!("= {r} fixed")),
        ),
        ns,
        &rows,
    );
}

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let pow2 = |lo: u32, hi: u32| (lo..=hi).map(|m| 1usize << m).collect::<Vec<_>>();
    let seeds = if quick { 8 } else { 32 };
    smooth_baseline(&pow2(6, 14), seeds);
    let both = [PoleProfile::Uniform, PoleProfile::Tapered];
    frozen_source(
        &pow2(6, if quick { 10 } else { 13 }),
        seeds,
        Some(64),
        &both,
    );
    frozen_source(&pow2(6, if quick { 9 } else { 11 }), seeds, None, &both);
    let k_ns = pow2(6, if quick { 8 } else { 10 });
    let k_seeds = if quick { 8 } else { 16 };
    k_problem(&k_ns, k_seeds, 40, 10, &both, None);
    k_problem(&k_ns, k_seeds, 40, 10, &[PoleProfile::Uniform], Some(R_K));
}
