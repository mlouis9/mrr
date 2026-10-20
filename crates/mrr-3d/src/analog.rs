//! Independent reference: analog Monte Carlo k-eigenvalue for a homogeneous
//! one-group isotropic cube with vacuum faces.
//!
//! Shares nothing with the meshless method except the random number
//! streams: no gather, no images, no walker. Each generation tracks
//! neutrons from fission sites: exponential flights `−ln(u)/Σ_t`, leakage
//! when the flight leaves the cube, and at each collision
//!
//! - bank `⌊νΣ_f/Σ_t + u⌋` fission sites at the collision point (expected
//!   value `νΣ_f/Σ_t` per collision, the collision estimator of fission),
//! - scatter isotropically with probability `Σ_s/Σ_t`, else stop.
//!
//! `k_gen = sites banked / sites started`; the bank is resampled to the
//! population size for the next generation. The estimate is the mean over
//! active generations; its standard error ignores inter-generation
//! correlation (small for these populations).

use mrr_core::sampler::{RandomSampler, UnitSampler, stream_seed};
use mrr_geom3d::Vec3;

use crate::proposal::isotropic_direction;

/// Analog k estimate for the cube `[−h, h]³`
#[derive(Debug, Clone, Copy)]
pub struct AnalogCube {
    pub half_width: f64,
    pub sigma_t: f64,
    pub sigma_s: f64,
    pub nu_sigma_f: f64,
}

/// Mean k over active generations and its (naive) standard error
#[derive(Debug, Clone, Copy)]
pub struct AnalogResult {
    pub k: f64,
    pub std_error: f64,
}

impl AnalogCube {
    pub fn solve(
        &self,
        population: usize,
        generations: usize,
        inactive: usize,
        seed: u64,
    ) -> AnalogResult {
        assert!(generations > inactive + 1);
        let h = self.half_width;
        let inside = |p: Vec3| p.abs().max_element() < h;
        let mut rng = RandomSampler::new(stream_seed(seed, 99, 0, 0));
        let mut u = [0.0; 2];
        let next = |rng: &mut RandomSampler| {
            let mut v = [0.0];
            rng.next_point(&mut v);
            v[0]
        };

        // Uniform initial sites
        let mut sites: Vec<Vec3> = (0..population)
            .map(|_| {
                let mut p = [0.0; 3];
                rng.next_point(&mut p);
                Vec3::new(p[0], p[1], p[2]) * (2.0 * h) - Vec3::splat(h)
            })
            .collect();

        let fission_per_collision = self.nu_sigma_f / self.sigma_t;
        let scatter_probability = self.sigma_s / self.sigma_t;
        let mut ks = Vec::with_capacity(generations - inactive);
        for generation in 0..generations {
            let mut bank = Vec::with_capacity(2 * population);
            for &site in &sites {
                let mut x = site;
                loop {
                    rng.next_point(&mut u);
                    let dir = isotropic_direction(u[0], u[1]);
                    let d = -(1.0 - next(&mut rng)).ln() / self.sigma_t;
                    x += d * dir;
                    if !inside(x) {
                        break; // leaked
                    }
                    let n_new = (fission_per_collision + next(&mut rng)).floor() as usize;
                    bank.extend(std::iter::repeat_n(x, n_new));
                    if next(&mut rng) >= scatter_probability {
                        break; // absorbed
                    }
                }
            }
            let k = bank.len() as f64 / sites.len() as f64;
            if generation >= inactive {
                ks.push(k);
            }
            assert!(!bank.is_empty(), "population died out");
            // Resample the bank to the population size
            sites = (0..population)
                .map(|_| {
                    let i = ((next(&mut rng) * bank.len() as f64) as usize).min(bank.len() - 1);
                    bank[i]
                })
                .collect();
        }
        let n = ks.len() as f64;
        let k = ks.iter().sum::<f64>() / n;
        let var = ks.iter().map(|x| (x - k).powi(2)).sum::<f64>() / (n - 1.0);
        AnalogResult {
            k,
            std_error: (var / n).sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huge_cube_approaches_k_infinity() {
        // 40 mfp half-width: leakage is a few percent at most
        let cube = AnalogCube {
            half_width: 40.0,
            sigma_t: 1.0,
            sigma_s: 0.5,
            nu_sigma_f: 0.6,
        };
        let r = cube.solve(2000, 30, 10, 1);
        let k_inf = 0.6 / 0.5;
        assert!(r.k < k_inf && r.k > 0.97 * k_inf, "{r:?} vs k_inf {k_inf}");
    }

    #[test]
    fn pure_absorber_k_is_fission_over_absorption_times_non_leakage() {
        // Σ_s = 0: one flight per neutron, so k = (νΣ_f/Σ_t) · P(collide inside)
        let cube = AnalogCube {
            half_width: 1.0,
            sigma_t: 1.0,
            sigma_s: 0.0,
            nu_sigma_f: 1.0,
        };
        let r = cube.solve(5000, 20, 5, 2);
        assert!(r.k > 0.3 && r.k < 0.9, "{r:?}");
    }
}
