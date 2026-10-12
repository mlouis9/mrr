//! Sources of points in the unit cube [0,1)^d

/// Anything that can produce the next point in [0,1)^d
pub trait UnitSampler {
    /// Fill `u` with the next point (u.len() is the dimension)
    fn next_point(&mut self, u: &mut [f64]);
}

/// Pseudo-random, seeded: the only place that uses the `rand` crate
pub struct RandomSampler {
    rng: rand::rngs::StdRng,
}

impl RandomSampler {
    pub fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
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

/// Deterministic 1D midpoints: (0.5/n, 1.5/n, ...). For tests.
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
