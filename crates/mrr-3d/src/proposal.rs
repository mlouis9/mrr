//! The global mixture proposal (balance heuristic, §8.2–8.3).
//!
//! # The density
//!
//! Node `x` is drawn from
//!
//! ```text
//! p(x) = α p_spread(x) + (1 − α) π(x)
//!
//! π(x) = Σ_a a_a Σ_{t : r_{a,t} < R_c}  1 / (4π R_c r_{a,t}²),   r_{a,t} = |image(t, x) − x_a|
//! ```
//!
//! - `p_spread = 1/V` (uniform; a smoother overlay density can replace it later).
//! - `π` is the **pole** component: pick atom `a` with probability
//!   `a_a ∝ Σ_g |w_a q_a,g,00|`, an isotropic direction Ω and a path length
//!   `r ~ U(0, R_c)`, then walk `r` from `x_a` along Ω (reflecting). In the
//!   unfolded space the endpoint density is `1/(4π R_c r²)` on the ball of
//!   radius `R_c`; folding back into the box sums it over the images `t`.
//!   This `1/r²` exactly matches the kernel pole, so `φ(x)/p(x)` is bounded
//!   near every atom and the node estimator has finite variance.
//!
//! # The weight
//!
//! Every node, however it was drawn, gets the same weight `1 / (N p(x))`
//! with the **full** mixture density, and is gathered the same way. Draws
//! are stratified (`N_s = round(αN)` spread, `N_p = N − N_s` pole), so the
//! density actually used is the one with `α = N_s/N`:
//!
//! ```text
//! w(x) = 1 / (N_s / V + N_p π(x))
//! ```
//!
//! # Zero samples
//!
//! A pole flight that leaves through a vacuum face lands nowhere: it is a
//! sample of `f = 0`. It still counts in `N` (the weights above already
//! assume it) but produces no node. So `Targets` may hold fewer than `N`
//! points; this is correct, not a loss.
//!
//! # Determinism
//!
//! Spread and pole draws use separate strata of the given streams
//! ([`SamplerStreams::stratum`]), each chunked in fixed `CHUNK_SIZE` pieces.
//! With scrambled Sobol streams each stratum is one RQMC point set: 3D for
//! spread nodes, 4D `(atom, μ, φ, r)` for pole flights, with atoms ordered
//! along a Z-curve so that the stratified atom coordinate is also stratified
//! in space.

use std::f64::consts::PI;

use mrr_core::sampler::{CHUNK_SIZE, SamplerStreams, UnitSampler};
use mrr_core::traits::{Proposal, Source, Targets};
use mrr_core::{Atoms, MaterialId};
use mrr_geom3d::{BoundaryBox, Ended, Partition, Path, Vec3, Walker, new_partition, tol};

/// Global mixture of a uniform (spread) and an atom-centred `1/r²` (pole)
/// proposal
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixtureProposal {
    alpha: f64,
    r_c: f64,
    profile: PoleProfile,
}

/// Radial distribution of pole flights, `r ∈ [0, r_c)` with density `g(r)`.
/// The pole density is `π_a(x) = g(r)/(4π r²)` around each atom.
///
/// - `Uniform`: `g = 1/r_c`. Simplest, but `π` jumps to zero at `r = r_c`,
///   so `φ/p` is discontinuous on every atom's ball surface: `N` surfaces
///   at the node spacing, which removes most of the QMC gain.
/// - `Tapered`: `g = 3(1 − r/r_c)²/r_c`, sampled by
///   `r = r_c (1 − (1 − u)^{1/3})`. Same `1/r²` pole (so the same bounded
///   scores), but `π` and its first derivative vanish at `r_c`: `p` is
///   C¹ away from atoms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoleProfile {
    #[default]
    Uniform,
    Tapered,
}

impl PoleProfile {
    /// Flight length from `u ∈ [0, 1)` (inverse CDF)
    pub fn radius(self, r_c: f64, u: f64) -> f64 {
        match self {
            PoleProfile::Uniform => r_c * u,
            PoleProfile::Tapered => r_c * (1.0 - (1.0 - u).cbrt()),
        }
    }

    /// Radial density `g(r)` on `[0, r_c)`, zero outside
    pub fn pdf(self, r_c: f64, r: f64) -> f64 {
        if !(0.0..r_c).contains(&r) {
            return 0.0;
        }
        match self {
            PoleProfile::Uniform => 1.0 / r_c,
            PoleProfile::Tapered => 3.0 * (1.0 - r / r_c).powi(2) / r_c,
        }
    }
}

impl MixtureProposal {
    /// `alpha ∈ (0, 1]`: spread fraction (must be > 0 so that every point
    /// has positive density); `r_c > 0`: pole radius
    pub fn new(alpha: f64, r_c: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "spread fraction must be in (0, 1], got {alpha}"
        );
        assert!(
            r_c > 0.0 && r_c.is_finite(),
            "pole radius must be positive and finite, got {r_c}"
        );
        Self {
            alpha,
            r_c,
            profile: PoleProfile::Uniform,
        }
    }

    /// The same proposal with another radial profile
    pub fn with_profile(self, profile: PoleProfile) -> Self {
        Self { profile, ..self }
    }

    pub fn profile(&self) -> PoleProfile {
        self.profile
    }

    /// Radius at which an atom's ball holds about `k` atoms of a uniform
    /// population of `n` in volume `v`: `(3 k v / (4π n))^{1/3}`
    pub fn suggested_radius(volume: f64, n: usize, k: f64) -> f64 {
        (3.0 * k * volume / (4.0 * PI * n as f64)).cbrt()
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn r_c(&self) -> f64 {
        self.r_c
    }

    /// Stratified counts `(N_s, N_p)` for `n` nodes: at least one spread node
    pub fn split_counts(&self, n: usize) -> (usize, usize) {
        let n_s = ((self.alpha * n as f64).round() as usize).clamp(1.min(n), n);
        (n_s, n - n_s)
    }
}

// ---------------------------------------------------------------------------
// Atom selection
// ---------------------------------------------------------------------------

/// Pole masses `a_a ∝ Σ_g |w_a q_a,g,00|` and their cumulative table
#[derive(Debug, Clone)]
pub struct PoleTable {
    /// Normalized masses `a_a` (sum to 1)
    mass: Vec<f64>,
    /// Atom indices in selection order (Morton order of their positions for
    /// [`new`](Self::new), identity for [`from_masses`](Self::from_masses))
    order: Vec<usize>,
    /// `cdf[k] = a_{order[0]} + … + a_{order[k]}`, with `cdf.last() == 1`
    cdf: Vec<f64>,
}

/// Interleave the low 21 bits of three integers (Morton / Z-order key)
fn morton_key(x: u64, y: u64, z: u64) -> u64 {
    fn spread(mut v: u64) -> u64 {
        v &= 0x1f_ffff;
        v = (v | v << 32) & 0x1f_0000_0000_ffff;
        v = (v | v << 16) & 0x1f_0000_ff00_00ff;
        v = (v | v << 8) & 0x100f_00f0_0f00_f00f;
        v = (v | v << 4) & 0x10c3_0c30_c30c_30c3;
        v = (v | v << 2) & 0x1249_2492_4924_9249;
        v
    }
    spread(x) | spread(y) << 1 | spread(z) << 2
}

/// Atom indices sorted along a Z-order curve over the atoms' bounding box.
///
/// Why: with QMC points the selection coordinate `u_0` is stratified, so
/// consecutive slices of `[0, 1)` should map to *nearby* atoms; the flights
/// (and the nodes) then inherit the stratification in space. With random
/// points the order is irrelevant.
pub fn spatial_order(points: &[Vec3]) -> Vec<usize> {
    if points.is_empty() {
        return Vec::new();
    }
    let lo = points.iter().copied().reduce(Vec3::min).unwrap();
    let hi = points.iter().copied().reduce(Vec3::max).unwrap();
    let scale = (hi - lo).max(Vec3::splat(1e-300));
    let max = ((1u64 << 21) - 1) as f64;
    let key = |p: Vec3| {
        let q = (p - lo) / scale * max;
        morton_key(q.x as u64, q.y as u64, q.z as u64)
    };
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by_key(|&i| key(points[i]));
    order
}

impl PoleTable {
    /// `None` when every atom has zero isotropic emission (nothing to aim at)
    pub fn new(atoms: &Atoms<Vec3>) -> Option<Self> {
        let layout = *atoms.layout();
        let raw: Vec<f64> = (0..atoms.len())
            .map(|i| {
                let q = atoms.atom_moments(i);
                let w = atoms.weights()[i];
                (0..layout.n_groups())
                    .map(|g| (w * q[layout.group_range(g).start]).abs())
                    .sum()
            })
            .collect();
        Self::with_order(&raw, spatial_order(atoms.points()))
    }

    /// From unnormalized non-negative masses
    pub fn from_masses(raw: &[f64]) -> Option<Self> {
        Self::with_order(raw, (0..raw.len()).collect())
    }

    /// From unnormalized masses, selected in the given order of atom indices
    pub fn with_order(raw: &[f64], order: Vec<usize>) -> Option<Self> {
        assert_eq!(order.len(), raw.len());
        assert!(raw.iter().all(|m| *m >= 0.0 && m.is_finite()));
        let total: f64 = raw.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let mass: Vec<f64> = raw.iter().map(|m| m / total).collect();
        let mut acc = 0.0;
        let mut cdf: Vec<f64> = order
            .iter()
            .map(|&i| {
                acc += mass[i];
                acc
            })
            .collect();
        *cdf.last_mut().unwrap() = 1.0; // remove round-off
        Some(Self { mass, order, cdf })
    }

    pub fn mass(&self) -> &[f64] {
        &self.mass
    }

    pub fn cdf(&self) -> &[f64] {
        &self.cdf
    }

    /// The atom `order[k]` with `cdf[k−1] <= u < cdf[k]`, for `u ∈ [0, 1)`.
    /// Atoms of zero mass are never selected.
    pub fn select(&self, u: f64) -> usize {
        // Clamp: u can round to 1.0 (e.g. from a QMC point), and then no
        // entry of the cdf exceeds it
        let k = self
            .cdf
            .partition_point(|c| *c <= u)
            .min(self.cdf.len() - 1);
        self.order[k]
    }
}

// ---------------------------------------------------------------------------
// The pole component: sampling and density
// ---------------------------------------------------------------------------

/// Isotropic unit vector from `(u1, u2) ∈ [0,1)²`: `μ = 2u1 − 1`, `φ = 2π u2`
pub fn isotropic_direction(u1: f64, u2: f64) -> Vec3 {
    let mu = 2.0 * u1 - 1.0;
    let s = (1.0 - mu * mu).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    Vec3::new(s * phi.cos(), s * phi.sin(), mu)
}

/// Scratch for pole flights
pub struct FlightScratch {
    path: Path,
    mats: Partition<MaterialId>,
}

impl Default for FlightScratch {
    fn default() -> Self {
        Self {
            path: Path::new(Vec3::ZERO, Vec3::Z),
            mats: new_partition(),
        }
    }
}

/// One pole flight from atom position `x_a`: direction from `u[0], u[1]`
/// (isotropic), path length `r = r_c u[2]`, walked with reflections.
/// `None` if the walk leaves through a vacuum face (a zero sample).
pub fn pole_flight<G: Walker>(
    geometry: &G,
    x_a: Vec3,
    r_c: f64,
    u: [f64; 3],
    scratch: &mut FlightScratch,
) -> Option<Vec3> {
    pole_flight_with(geometry, x_a, r_c, PoleProfile::Uniform, u, scratch)
}

/// [`pole_flight`] with a radial profile: `r = profile.radius(r_c, u[2])`
pub fn pole_flight_with<G: Walker>(
    geometry: &G,
    x_a: Vec3,
    r_c: f64,
    profile: PoleProfile,
    u: [f64; 3],
    scratch: &mut FlightScratch,
) -> Option<Vec3> {
    let omega = isotropic_direction(u[0], u[1]);
    let r = profile.radius(r_c, u[2]);
    let ended = geometry.walk(x_a, omega, r, &mut scratch.path, &mut scratch.mats);
    if let Ended::Reached = ended {
        Some(scratch.path.end_point())
    } else {
        None
    }
}

/// Pole density `π(x) = Σ_a a_a Σ_{t: r < r_c} 1/(4π r_c r²)`, by brute force
/// over atoms and the node's images. Pairs with `r <= tol::LEN` are skipped,
/// as in the gather (the density is infinite there, so the weight is 0, and
/// it happens with probability zero).
///
/// `images` is scratch.
pub fn pole_density(
    bbox: &BoundaryBox,
    points: &[Vec3],
    table: &PoleTable,
    r_c: f64,
    x: Vec3,
    images: &mut Vec<[i32; 3]>,
) -> f64 {
    pole_density_with(bbox, points, table, r_c, PoleProfile::Uniform, x, images)
}

/// [`pole_density`] with a radial profile: `Σ_a a_a Σ_t g(r)/(4π r²)`
pub fn pole_density_with(
    bbox: &BoundaryBox,
    points: &[Vec3],
    table: &PoleTable,
    r_c: f64,
    profile: PoleProfile,
    x: Vec3,
    images: &mut Vec<[i32; 3]>,
) -> f64 {
    images.clear();
    bbox.images_within(x, r_c, images);
    let mut density = 0.0;
    for &t in images.iter() {
        let y = bbox.image(t, x);
        for (&x_a, &a) in points.iter().zip(table.mass()) {
            let r2 = (y - x_a).length_squared();
            if r2 < r_c * r_c && r2 > tol::LEN * tol::LEN {
                density += a * profile.pdf(r_c, r2.sqrt()) / (4.0 * PI * r2);
            }
        }
    }
    density
}

/// The importance weight `1 / (N_s/V + N_p π(x))` of a node, given its pole
/// density. With stratified counts this is `1/(N p(x))` for `α = N_s/N`.
pub fn mixture_weight(n_spread: usize, n_pole: usize, volume: f64, pole: f64) -> f64 {
    1.0 / (n_spread as f64 / volume + n_pole as f64 * pole)
}

// ---------------------------------------------------------------------------
// The proposal
// ---------------------------------------------------------------------------

impl<G: Walker> Proposal<G> for MixtureProposal {
    fn sample_target_atoms(
        &self,
        geometry: &G,
        source: &Source<Vec3>,
        streams: &impl SamplerStreams,
    ) -> Targets<Vec3> {
        let n = source.n_targets();
        assert!(n > 0, "no nodes requested");
        let volume = geometry.volume();

        // Without atoms (initialization) or emission, the pole has nothing to
        // aim at: all-spread, i.e. the uniform proposal
        let pole = match source {
            Source::Existing(atoms, _) => PoleTable::new(atoms).map(|t| (*atoms, t)),
            Source::Initialize(_) => None,
        };
        let (n_s, n_p) = match pole {
            Some(_) => self.split_counts(n),
            None => (n, 0),
        };

        let mut targets = Targets {
            points: Vec::with_capacity(n),
            weights: Vec::with_capacity(n),
        };

        // Spread and pole draws are separate strata, each its own point set
        // (for QMC: its own net, starting at index 0)
        let (spread_streams, pole_streams) = (streams.stratum(0), streams.stratum(1));
        let c_s = n_s.div_ceil(CHUNK_SIZE);
        let mut u = vec![0.0; G::SAMPLE_DIM];
        for chunk in 0..c_s {
            let n_here = CHUNK_SIZE.min(n_s - chunk * CHUNK_SIZE);
            let mut sampler = spread_streams.stream(chunk);
            let mut count = 0;
            while count < n_here {
                sampler.next_point(&mut u);
                if let Some(x) = geometry.map_uniform(&u) {
                    targets.points.push(x);
                    count += 1;
                }
            }
        }

        // Pole draws
        if let Some((atoms, table)) = &pole {
            let mut scratch = FlightScratch::default();
            let mut u = [0.0; 4];
            for chunk in 0..n_p.div_ceil(CHUNK_SIZE) {
                let n_here = CHUNK_SIZE.min(n_p - chunk * CHUNK_SIZE);
                let mut sampler = pole_streams.stream(chunk);
                for _ in 0..n_here {
                    sampler.next_point(&mut u);
                    let a = table.select(u[0]);
                    let x_a = atoms.points()[a];
                    if let Some(x) = pole_flight_with(
                        geometry,
                        x_a,
                        self.r_c,
                        self.profile,
                        [u[1], u[2], u[3]],
                        &mut scratch,
                    ) {
                        targets.points.push(x);
                    }
                }
            }
        }

        // One weight per node from the full mixture density
        let mut images = Vec::new();
        targets.weights = targets
            .points
            .iter()
            .map(|&x| {
                let pi = match &pole {
                    Some((atoms, table)) => pole_density_with(
                        geometry.boundary(),
                        atoms.points(),
                        table,
                        self.r_c,
                        self.profile,
                        x,
                        &mut images,
                    ),
                    None => 0.0,
                };
                mixture_weight(n_s, n_p, volume, pi)
            })
            .collect();
        targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::homogeneous_box;
    use mrr_core::MomentLayout;
    use mrr_core::sampler::{RandomStreams, purpose};
    use mrr_geom3d::Bc;
    use mrr_geom3d::testing::*;

    fn streams(cycle: usize) -> RandomStreams {
        RandomStreams {
            base_seed: 11,
            purpose: purpose::PROPOSAL,
            cycle,
        }
    }

    /// Isotropic one-group atoms with q_00 = 1 at the given points and weights
    fn atoms(points: &[Vec3], weights: &[f64]) -> Atoms<Vec3> {
        let layout = MomentLayout::from_harmonic_order(0, 1);
        let mut a = Atoms::new(points.len(), layout);
        for (p, w) in points.iter().zip(weights) {
            a.push(*p, *w, &[1.0]);
        }
        a
    }

    // ---- select -----------------------------------------------------------

    #[test]
    fn select_follows_the_cdf() {
        let t = PoleTable::from_masses(&[1.0, 0.0, 3.0]).unwrap();
        assert_all_close(t.mass(), &[0.25, 0.0, 0.75]);
        assert_eq!(t.select(0.0), 0);
        assert_eq!(t.select(0.2499), 0);
        assert_eq!(t.select(0.25), 2); // zero-mass atom 1 is skipped
        assert_eq!(t.select(0.9999999), 2);
        assert_eq!(t.select(1.0), 2); // guarded
    }

    #[test]
    fn select_frequencies_match_masses() {
        let t = PoleTable::from_masses(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let n = 40_000;
        let mut counts = [0usize; 4];
        let mut s = streams(0).stream(0);
        let mut u = [0.0];
        for _ in 0..n {
            s.next_point(&mut u);
            counts[t.select(u[0])] += 1;
        }
        for (c, m) in counts.iter().zip(t.mass()) {
            let p = *c as f64 / n as f64;
            let se = (m * (1.0 - m) / n as f64).sqrt();
            assert!((p - m).abs() < 4.0 * se, "{p} vs {m}");
        }
    }

    #[test]
    fn pole_table_uses_weighted_isotropic_emission() {
        let a = atoms(&[Vec3::ZERO, Vec3::X], &[1.0, -3.0]);
        let t = PoleTable::new(&a).unwrap();
        assert_all_close(t.mass(), &[0.25, 0.75]);
        assert!(PoleTable::from_masses(&[0.0, 0.0]).is_none());
    }

    // ---- flight -----------------------------------------------------------

    #[test]
    fn isotropic_directions_are_unit_with_zero_mean() {
        let mut s = streams(1).stream(0);
        let (mut u, mut mean, n) = ([0.0; 2], Vec3::ZERO, 20_000);
        for _ in 0..n {
            s.next_point(&mut u);
            let d = isotropic_direction(u[0], u[1]);
            assert_close_tol(d.length(), 1.0, Tol::NUMERICAL);
            mean += d;
        }
        // Each component has variance 1/3
        let se = (1.0 / (3.0 * n as f64)).sqrt();
        assert!((mean / n as f64).abs().max_element() < 4.0 * se);
    }

    #[test]
    fn flight_without_boundaries_lands_at_uniform_radius() {
        // Big vacuum box: no face within r_c, so |x − x_a| = r_c u exactly
        let model = homogeneous_box(10.0, Bc::Vacuum);
        let mut sc = FlightScratch::default();
        for (k, &u2) in [0.0, 0.3, 0.999].iter().enumerate() {
            let u = [0.1 * k as f64, 0.7, u2];
            let x = pole_flight(&model, Vec3::ONE, 2.0, u, &mut sc).unwrap();
            assert_close_tol((x - Vec3::ONE).length(), 2.0 * u2, Tol::NUMERICAL);
            let expected = Vec3::ONE + 2.0 * u2 * isotropic_direction(u[0], u[1]);
            assert_vec_close_tol(x, expected, Tol::NUMERICAL);
        }
    }

    #[test]
    fn flight_reflects_into_the_box() {
        // Straight up from 0.2 below the top of a reflective box, length 0.5:
        // folds back to 0.3 below the top
        let model = homogeneous_box(1.0, Bc::Reflective);
        let mut sc = FlightScratch::default();
        let x_a = Vec3::new(0.0, 0.0, 0.8);
        // u1 → 1 gives μ → 1 (straight up); use the exact pole via u1 = 1
        let x = pole_flight(&model, x_a, 1.0, [1.0, 0.0, 0.5], &mut sc).unwrap();
        assert_vec_close_tol(x, Vec3::new(0.0, 0.0, 0.7), Tol::NUMERICAL);
    }

    #[test]
    fn vacuum_escape_fraction_matches_the_formula() {
        // Atom at distance d below a vacuum face (the top of a big box; the
        // other faces are out of reach). A flight escapes iff r μ > d:
        // P = [(R − d) − d ln(R/d)] / (2R)
        let model = homogeneous_box(10.0, Bc::Vacuum);
        let (d, r_c) = (0.4, 1.5);
        let x_a = Vec3::new(0.0, 0.0, 10.0 - d);
        let exact = ((r_c - d) - d * (r_c / d).ln()) / (2.0 * r_c);
        let mut sc = FlightScratch::default();
        let mut s = streams(2).stream(0);
        let (n, mut escaped) = (20_000, 0);
        let mut u = [0.0; 3];
        for _ in 0..n {
            s.next_point(&mut u);
            if pole_flight(&model, x_a, r_c, u, &mut sc).is_none() {
                escaped += 1;
            }
        }
        let p = escaped as f64 / n as f64;
        let se = (exact * (1.0 - exact) / n as f64).sqrt();
        assert!(
            (p - exact).abs() < 4.0 * se,
            "escaped {p}, expected {exact}"
        );
    }

    // ---- density ----------------------------------------------------------

    #[test]
    fn pole_density_by_hand() {
        let r_c = 1.0;
        let norm = 1.0 / (4.0 * PI * r_c);
        let t = PoleTable::from_masses(&[1.0]).unwrap();
        let mut images = Vec::new();

        // Vacuum: direct term only, zero beyond r_c
        let vac = homogeneous_box(1.0, Bc::Vacuum);
        let b = vac.boundary();
        let x_a = [Vec3::ZERO];
        let d = pole_density(b, &x_a, &t, r_c, Vec3::new(0.5, 0.0, 0.0), &mut images);
        assert_close(d, norm / 0.25);
        let far = pole_density(b, &x_a, &t, r_c, Vec3::new(0.9, 0.9, 0.0), &mut images);
        assert_eq!(far, 0.0);

        // Reflective top face only: atom 0.2 below it, node 0.1 below it.
        // Direct r = 0.1, image r = 0.2 + 0.1 = 0.3
        let mut faces = [[Bc::Vacuum; 2]; 3];
        faces[2][1] = Bc::Reflective;
        let one = BoundaryBox::new(Vec3::splat(-1.0), Vec3::splat(1.0), faces).unwrap();
        let x_a = [Vec3::new(0.0, 0.0, 0.8)];
        let d = pole_density(&one, &x_a, &t, r_c, Vec3::new(0.0, 0.0, 0.9), &mut images);
        assert_close(d, norm * (1.0 / 0.01 + 1.0 / 0.09));
    }

    #[test]
    fn tapered_radii_follow_their_density() {
        // E[r] = r_c/4 and P(r < r_c/2) = 1 − 1/8 for g = 3(1 − r/r_c)²/r_c
        let (r_c, n) = (2.0, 40_000);
        let mut s = streams(9).stream(0);
        let (mut u, mut sum, mut inner) = ([0.0], 0.0, 0usize);
        for _ in 0..n {
            s.next_point(&mut u);
            let r = PoleProfile::Tapered.radius(r_c, u[0]);
            assert!((0.0..r_c).contains(&r));
            sum += r;
            inner += usize::from(r < 0.5 * r_c);
        }
        // std of r is r_c √(3/80)
        let se = r_c * (3.0f64 / 80.0).sqrt() / (n as f64).sqrt();
        assert!((sum / n as f64 - 0.5).abs() < 4.0 * se);
        let p = inner as f64 / n as f64;
        assert!((p - 0.875).abs() < 4.0 * (0.875f64 * 0.125 / n as f64).sqrt());
        // pdf integrates to 1 and vanishes at r_c
        let gl = mrr_math::GaussLegendre::new(20);
        let total =
            gl.integrate(|x| r_c / 2.0 * PoleProfile::Tapered.pdf(r_c, r_c / 2.0 * (x + 1.0)));
        assert_close(total, 1.0);
        assert_eq!(PoleProfile::Tapered.pdf(r_c, r_c), 0.0);
    }

    #[test]
    fn mixture_weight_limits() {
        // No pole: the uniform weight V/N
        assert_close(mixture_weight(10, 0, 8.0, 123.0), 0.8);
        // Far from atoms (π = 0): V/N_s, larger than V/N
        assert_close(mixture_weight(5, 5, 8.0, 0.0), 1.6);
        // At a pole the weight vanishes like r²
        assert!(mixture_weight(5, 5, 8.0, 1e12) < 1e-12);
    }

    // ---- the proposal ----------------------------------------------------

    #[test]
    fn initialize_is_the_uniform_proposal() {
        let model = homogeneous_box(1.0, Bc::Reflective);
        let prop = MixtureProposal::new(0.5, 0.3);
        let t = prop.sample_target_atoms(&model, &Source::Initialize(100), &streams(0));
        assert_eq!(t.points.len(), 100);
        assert!(t.weights.iter().all(|w| (w - 8.0 / 100.0).abs() < 1e-14));
    }

    #[test]
    fn existing_source_gives_n_nodes_in_a_reflective_box_deterministically() {
        let model = homogeneous_box(1.0, Bc::Reflective);
        // N = 2000 nodes aimed at two atoms
        let src = atoms(&[Vec3::ZERO, Vec3::splat(0.9)], &[1.0, 1.0]);
        let prop = MixtureProposal::new(0.3, 0.5);
        let a = prop.sample_target_atoms(&model, &Source::Existing(&src, 2000), &streams(4));
        let b = prop.sample_target_atoms(&model, &Source::Existing(&src, 2000), &streams(4));
        assert_eq!(a.points.len(), 2000);
        assert_eq!(a.points, b.points);
        assert!(a.points.iter().all(|p| model.boundary().contains(*p)));
        // Pole nodes concentrate: more than 3x the uniform share within r_c/2
        let near = a.points.iter().filter(|p| p.length() < 0.25).count() as f64;
        let uniform_share = 2000.0 * (4.0 / 3.0 * PI * 0.25f64.powi(3)) / 8.0;
        assert!(near > 3.0 * uniform_share, "{near} vs {uniform_share}");
    }

    #[test]
    fn split_counts_round_and_keep_a_spread_node() {
        let p = MixtureProposal::new(0.25, 1.0);
        assert_eq!(p.split_counts(100), (25, 75));
        assert_eq!(MixtureProposal::new(0.001, 1.0).split_counts(10), (1, 9));
        assert_eq!(MixtureProposal::new(1.0, 1.0).split_counts(7), (7, 0));
    }

    #[test]
    #[should_panic(expected = "spread fraction")]
    fn zero_alpha_is_rejected() {
        MixtureProposal::new(0.0, 1.0);
    }
}
