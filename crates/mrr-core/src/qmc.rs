//! Randomized quasi-Monte Carlo: Owen-scrambled Sobol points.
//!
//! # The point set
//!
//! The Sobol sequence in base 2 is a (t, s)-sequence: every block of `2^m`
//! consecutive points starting at a multiple of `2^m` is a (t, m, s)-net,
//! i.e. stratified in every dyadic box of volume `2^{t−m}`. Dimension 0 is
//! the van der Corput sequence; dimensions 1.. use the Joe–Kuo direction
//! numbers (`new-joe-kuo-6.21201`). [`MAX_DIM`] dimensions are tabulated,
//! enough for every sampler in the method (3D positions, 4D pole draws).
//!
//! # Randomization
//!
//! Raw Sobol points are deterministic: no error bars and a biased average
//! for a fixed N. Owen's nested uniform scramble randomly permutes the
//! dyadic intervals at every level, which
//!
//! - keeps the net structure (so the QMC convergence rate),
//! - makes each point uniform on `[0,1)^s` (so the estimator is unbiased),
//! - makes independent scrambles independent replicas (so error bars come
//!   from the spread across scrambles, never from within one).
//!
//! The scramble is the hash-based form of Burley (JCGT 2020): bit-reverse,
//! apply a Laine–Karras style hash whose low bits only depend on lower
//! bits, bit-reverse back. Each dimension gets its own seed.
//!
//! # Chunks
//!
//! As for [`RandomStreams`](crate::sampler::RandomStreams), chunk `c` of a
//! stream covers global indices `c·CHUNK_SIZE ..`. All chunks of one stream
//! therefore form *one* point set, whatever the thread count.

use crate::sampler::{CHUNK_SIZE, SamplerStreams, UnitSampler, stream_seed};

/// Number of tabulated Sobol dimensions
pub const MAX_DIM: usize = 8;

/// Joe–Kuo primitive polynomials for dimensions 1..: (degree s, coefficients a, initial m_k)
const JOE_KUO: [(u32, u32, &[u32]); MAX_DIM - 1] = [
    (1, 0, &[1]),
    (2, 1, &[1, 3]),
    (3, 1, &[1, 3, 1]),
    (3, 2, &[1, 1, 1]),
    (4, 1, &[1, 1, 3, 3]),
    (4, 4, &[1, 3, 5, 13]),
    (5, 2, &[1, 1, 5, 5, 17]),
];

/// The 32 direction numbers of every dimension (bit `31 − k` is level k)
fn direction_numbers() -> [[u32; 32]; MAX_DIM] {
    let mut v = [[0u32; 32]; MAX_DIM];
    for (k, x) in v[0].iter_mut().enumerate() {
        *x = 1 << (31 - k);
    }
    for (d, &(s, a, m)) in JOE_KUO.iter().enumerate() {
        let s = s as usize;
        let row = &mut v[d + 1];
        for k in 0..s.min(32) {
            row[k] = m[k] << (31 - k);
        }
        for k in s..32 {
            let mut x = row[k - s] ^ (row[k - s] >> s);
            for j in 1..s {
                if (a >> (s - 1 - j)) & 1 == 1 {
                    x ^= row[k - j];
                }
            }
            row[k] = x;
        }
    }
    v
}

/// The direction numbers, computed once
fn directions() -> &'static [[u32; 32]; MAX_DIM] {
    static V: std::sync::OnceLock<[[u32; 32]; MAX_DIM]> = std::sync::OnceLock::new();
    V.get_or_init(direction_numbers)
}

/// Unscrambled Sobol coordinate `dim` of point `index`, as a 32-bit fraction
fn sobol_bits(v: &[[u32; 32]; MAX_DIM], index: u32, dim: usize) -> u32 {
    let mut x = 0;
    let mut i = index;
    let mut k = 0;
    while i != 0 {
        if i & 1 == 1 {
            x ^= v[dim][k];
        }
        i >>= 1;
        k += 1;
    }
    x
}

/// Hash whose bit `k` depends only on bits `<= k` of `x` (Laine–Karras form):
/// applied to bit-reversed values it permutes dyadic intervals level by
/// level, i.e. an Owen scramble
fn laine_karras(mut x: u32, seed: u32) -> u32 {
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x
}

/// Owen (nested uniform) scramble of a 32-bit fraction
pub fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
    laine_karras(x.reverse_bits(), seed).reverse_bits()
}

const TO_UNIT: f64 = 1.0 / 4_294_967_296.0; // 2^-32

/// One chunk of an Owen-scrambled Sobol point set
pub struct SobolSampler {
    v: &'static [[u32; 32]; MAX_DIM],
    seeds: [u32; MAX_DIM],
    /// `false` gives the raw (unscrambled) sequence
    scrambled: bool,
    index: u32,
}

impl SobolSampler {
    /// Points `start, start + 1, ...` of the sequence scrambled with `seed`
    pub fn new(seed: u64, start: usize, scrambled: bool) -> Self {
        let mut seeds = [0u32; MAX_DIM];
        for (d, s) in seeds.iter_mut().enumerate() {
            *s = stream_seed(seed, 0x5eed, d as u64, 0) as u32;
        }
        Self {
            v: directions(),
            seeds,
            scrambled,
            index: u32::try_from(start).expect("Sobol index beyond 2^32"),
        }
    }
}

impl UnitSampler for SobolSampler {
    fn next_point(&mut self, u: &mut [f64]) {
        assert!(
            u.len() <= MAX_DIM,
            "Sobol sampler has {MAX_DIM} dimensions, {} requested",
            u.len()
        );
        for (d, x) in u.iter_mut().enumerate() {
            let mut bits = sobol_bits(self.v, self.index, d);
            if self.scrambled {
                bits = nested_uniform_scramble(bits, self.seeds[d]);
            }
            // Midpoint of the 2^-32 cell: never exactly 0 (or 1)
            *x = (bits as f64 + 0.5) * TO_UNIT;
        }
        self.index = self.index.checked_add(1).expect("Sobol index overflow");
    }
}

/// Scrambled Sobol streams for one (base_seed, purpose, cycle, stratum):
/// the scramble is seeded from these, chunk `c` starts at `c·CHUNK_SIZE`
#[derive(Debug, Clone, Copy)]
pub struct SobolStreams {
    pub base_seed: u64,
    pub purpose: u64,
    pub cycle: usize,
    pub stratum: u64,
    /// `false` gives the raw sequence (for tests of the net property)
    pub scrambled: bool,
}

impl SobolStreams {
    pub fn new(base_seed: u64, purpose: u64, cycle: usize) -> Self {
        Self {
            base_seed,
            purpose,
            cycle,
            stratum: 0,
            scrambled: true,
        }
    }

    fn seed(&self) -> u64 {
        stream_seed(
            self.base_seed,
            self.purpose,
            self.cycle as u64,
            0xa5a5_0000 ^ self.stratum,
        )
    }
}

impl SamplerStreams for SobolStreams {
    type Sampler = SobolSampler;
    fn stream(&self, chunk: usize) -> SobolSampler {
        SobolSampler::new(self.seed(), chunk * CHUNK_SIZE, self.scrambled)
    }

    fn stratum(&self, s: u64) -> Self {
        Self {
            stratum: self.stratum.wrapping_mul(31).wrapping_add(s + 1),
            ..*self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(n: usize, dims: usize) -> Vec<Vec<f64>> {
        let mut s = SobolSampler::new(0, 0, false);
        (0..n)
            .map(|_| {
                let mut u = vec![0.0; dims];
                s.next_point(&mut u);
                u
            })
            .collect()
    }

    #[test]
    fn first_points_match_the_known_sequence() {
        // Leading bits (before the half-cell offset): dimension 0 is van der
        // Corput; dimensions 1, 2 from m = (1), (1, 3) by the Sobol recurrence
        let pts = raw(8, 3);
        let d0: Vec<f64> = pts.iter().map(|p| (p[0] * 8.0).floor() / 8.0).collect();
        let d1: Vec<f64> = pts.iter().map(|p| (p[1] * 8.0).floor() / 8.0).collect();
        let d2: Vec<f64> = pts.iter().map(|p| (p[2] * 8.0).floor() / 8.0).collect();
        assert_eq!(d0, [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]);
        // Direct (not Gray-code) order: point i XORs v_k for the set bits of i
        assert_eq!(d1, [0.0, 0.5, 0.75, 0.25, 0.625, 0.125, 0.375, 0.875]);
        assert_eq!(d2, [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125]);
    }

    /// Every pair of dims: the first 2^m points put exactly one point in each
    /// of the 2^m boxes of shape 2^-a × 2^-(m−a), for small t (t ≤ 2 here)
    fn is_net(pts: &[Vec<f64>], m: u32, t: u32) -> bool {
        let dims = pts[0].len();
        for i in 0..dims {
            for j in (i + 1)..dims {
                for a in 0..=(m - t) {
                    let b = m - t - a;
                    let mut counts = vec![0usize; 1 << (a + b)];
                    for p in pts {
                        let x = (p[i] * (1u64 << a) as f64) as usize;
                        let y = (p[j] * (1u64 << b) as f64) as usize;
                        counts[x * (1 << b) + y] += 1;
                    }
                    let expect = 1 << t;
                    if counts.iter().any(|&c| c != expect) {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn raw_and_scrambled_points_are_nets() {
        // 2D projections of the first 4 dims are (t ≤ 2, 10, 2)-nets
        assert!(is_net(&raw(1 << 10, 4), 10, 2));
        let mut s = SobolSampler::new(123, 0, true);
        let pts: Vec<Vec<f64>> = (0..1 << 10)
            .map(|_| {
                let mut u = vec![0.0; 4];
                s.next_point(&mut u);
                u
            })
            .collect();
        assert!(is_net(&pts, 10, 2), "scrambling must keep the net");
        // The first two dimensions are a (0, m, 2)-net
        let pair: Vec<Vec<f64>> = pts.iter().map(|p| p[..2].to_vec()).collect();
        assert!(is_net(&pair, 10, 0));
    }

    #[test]
    fn chunks_continue_one_point_set() {
        let st = SobolStreams::new(5, 1, 0);
        let mut whole = st.stream(0);
        let mut u = [0.0; 3];
        for _ in 0..CHUNK_SIZE {
            whole.next_point(&mut u);
        }
        whole.next_point(&mut u);
        let mut second = [0.0; 3];
        st.stream(1).next_point(&mut second);
        assert_eq!(u, second);
    }

    #[test]
    fn scrambles_differ_by_seed_cycle_and_stratum_and_stay_in_unit_cube() {
        let a = SobolStreams::new(1, 1, 0);
        let first = |s: &SobolStreams| {
            let mut u = [0.0; 4];
            s.stream(0).next_point(&mut u);
            u
        };
        let fa = first(&a);
        assert_ne!(fa, first(&SobolStreams::new(2, 1, 0)));
        assert_ne!(fa, first(&SobolStreams::new(1, 1, 1)));
        assert_ne!(fa, first(&a.stratum(1)));
        assert!(fa.iter().all(|x| *x > 0.0 && *x < 1.0));
    }

    #[test]
    fn smooth_integral_converges_faster_than_monte_carlo() {
        // ∫ Π cos-like smooth f over [0,1)^3, exact value known; RMSE over
        // 16 scrambles at N = 2^12 must be far below the MC error σ/√N
        let f = |u: &[f64]| {
            u.iter()
                .map(|x| 1.0 + 0.5 * (2.0 * std::f64::consts::PI * x).sin() + x * x)
                .product::<f64>()
        };
        let exact = (1.0f64 + 1.0 / 3.0).powi(3);
        let n = 1 << 12;
        let mut sq = 0.0;
        for seed in 0..16 {
            let mut s = SobolSampler::new(seed, 0, true);
            let mut u = [0.0; 3];
            let mut sum = 0.0;
            for _ in 0..n {
                s.next_point(&mut u);
                sum += f(&u);
            }
            sq += (sum / n as f64 - exact).powi(2);
        }
        let rmse = (sq / 16.0).sqrt();
        // σ(f) ≈ 0.9 here, so MC error ≈ 0.014; RQMC should be < 1e-3
        assert!(rmse < 1e-3, "RQMC rmse {rmse}");
    }
}
