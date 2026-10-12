//! Sources of points in the unit cube [0,1)^d
//!
//! # Reproducibility independent of thread count
//!
//! Work (e.g. sampling atoms) is split into fixed-size chunks of `CHUNK_SIZE`
//! items. Each chunk gets its *own* sampler, whose seed is derived only from
//! *what* the chunk is:
//!
//! ```text
//! seed = stream_seed(base_seed, purpose, cycle, chunk_index)
//! ```
//!
//! never from *who* computes it (thread id) or *when* (scheduling order).
//! Chunks can therefore be processed in any order, on any number of threads,
//! and every chunk still sees exactly the same numbers. Within a chunk, work
//! is sequential, so a rejected sample only shifts that chunk's own stream.

/// Number of items per chunk of work. Must be a fixed constant: if it were
/// derived from the number of threads, results would change with thread count.
pub const CHUNK_SIZE: usize = 1024;

/// Tags that separate random streams used for different parts of the
/// algorithm, so that e.g. initialization and the proposal never receive
/// identical numbers for the same (cycle, chunk).
pub mod purpose {
    pub const INIT: u64 = 0;
    pub const PROPOSAL: u64 = 1;
}

/// Anything that can produce the next point in [0,1)^d
pub trait UnitSampler {
    /// Fill `u` with the next point (u.len() is the dimension)
    fn next_point(&mut self, u: &mut [f64]);
}

/// Anything that can hand out an independent `UnitSampler` for each chunk.
///
/// Code that processes work in chunks takes one of these instead of a single
/// sampler, and asks for `stream(chunk_index)` at the start of each chunk.
pub trait SamplerStreams {
    type Sampler: UnitSampler;
    fn stream(&self, chunk: usize) -> Self::Sampler;
}

// ---------------------------------------------------------------------------
// Seed derivation
// ---------------------------------------------------------------------------

/// Scramble a u64 so that nearby inputs give unrelated outputs
/// (the SplitMix64 finalizer). `wrapping_*` because overflow is intended here
/// and would otherwise panic in debug builds.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed for one chunk's stream. Each field is folded in through `mix`, so
/// changing any one of them gives an unrelated seed. (Plain addition would
/// collide, e.g. (seed 1, cycle 1) == (seed 2, cycle 0).)
pub fn stream_seed(base_seed: u64, purpose: u64, cycle: u64, chunk: u64) -> u64 {
    mix(mix(mix(mix(base_seed) ^ purpose) ^ cycle) ^ chunk)
}

// ---------------------------------------------------------------------------
// Pseudo-random samplers (the only place that uses the `rand` crate)
// ---------------------------------------------------------------------------

/// Pseudo-random sampler. Xoshiro256++ is a named, fixed algorithm (unlike
/// `StdRng`, which may change between `rand` versions), fast, and cheap to seed.
pub struct RandomSampler {
    rng: rand::rngs::Xoshiro256PlusPlus,
}

impl RandomSampler {
    pub fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
}

impl UnitSampler for RandomSampler {
    fn next_point(&mut self, u: &mut [f64]) {
        use rand::RngExt;
        for x in u.iter_mut() {
            *x = self.rng.random();
        }
    }
}

/// Random streams for one (base_seed, purpose, cycle): chunk `c` gets a
/// `RandomSampler` seeded with `stream_seed(base_seed, purpose, cycle, c)`.
pub struct RandomStreams {
    pub base_seed: u64,
    pub purpose: u64,
    pub cycle: usize,
}

impl SamplerStreams for RandomStreams {
    type Sampler = RandomSampler;
    fn stream(&self, chunk: usize) -> RandomSampler {
        RandomSampler::new(stream_seed(
            self.base_seed,
            self.purpose,
            self.cycle as u64,
            chunk as u64,
        ))
    }
}

// ---------------------------------------------------------------------------
// Deterministic samplers (for tests)
// ---------------------------------------------------------------------------

/// Deterministic 1D midpoints (i + 0.5)/n, starting from index `i`.
pub struct GridSampler1D {
    n: usize,
    i: usize,
}

impl GridSampler1D {
    pub fn new(n: usize) -> Self {
        Self { n, i: 0 }
    }
}

impl UnitSampler for GridSampler1D {
    fn next_point(&mut self, u: &mut [f64]) {
        u[0] = (self.i as f64 + 0.5) / self.n as f64;
        self.i += 1;
    }
}

/// Grid streams: chunk `c` continues the global grid at index `c * CHUNK_SIZE`,
/// so the chunked result equals the unchunked grid (assuming no rejections).
pub struct GridStreams1D {
    pub n: usize,
}

impl SamplerStreams for GridStreams1D {
    type Sampler = GridSampler1D;
    fn stream(&self, chunk: usize) -> GridSampler1D {
        GridSampler1D {
            n: self.n,
            i: chunk * CHUNK_SIZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_seeds_do_not_collide_like_addition() {
        assert_ne!(stream_seed(1, 0, 1, 0), stream_seed(2, 0, 0, 0));
        assert_ne!(
            stream_seed(0, purpose::INIT, 0, 0),
            stream_seed(0, purpose::PROPOSAL, 0, 0)
        );
    }

    #[test]
    fn same_stream_gives_same_numbers() {
        let streams = RandomStreams {
            base_seed: 42,
            purpose: purpose::INIT,
            cycle: 3,
        };
        let (mut a, mut b) = (streams.stream(7), streams.stream(7));
        let (mut ua, mut ub) = ([0.0; 3], [0.0; 3]);
        for _ in 0..10 {
            a.next_point(&mut ua);
            b.next_point(&mut ub);
            assert_eq!(ua, ub);
        }
    }
}
