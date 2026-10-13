use crate::traits::Medium;
use std::ops::Range;

/// Minimal 1D medium on [0, width) that accepts every sample
pub(crate) struct TestSlab {
    pub width: f64,
    pub n_groups: usize,
    /// Test acceptance range for u\in[0,1) for rejection sampling
    accept: Range<f64>,
}

impl TestSlab {
    pub(crate) fn new(width: f64, n_groups: usize, accept: Range<f64>) -> Self {
        Self {
            width,
            n_groups,
            accept,
        }
    }
}

impl Medium for TestSlab {
    type Point = f64;
    const SAMPLE_DIM: usize = 1;
    fn n_groups(&self) -> usize {
        self.n_groups
    }
    fn optical_depth(&self, _: &f64, _: &f64) -> f64 {
        0.0
    }
    fn volume(&self) -> f64 {
        self.width
    }
    fn map_uniform(&self, u: &[f64]) -> Option<f64> {
        if self.accept.contains(&u[0]) {
            Some(u[0] * self.width)
        } else {
            None
        }
    }
}
