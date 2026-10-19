use crate::traits::{Emission, Emitted, Gather, Geometry};
use std::ops::Range;

// Closeness assertions and tolerances are shared across the workspace
// (see `mrr_math::testing` for the rule and the presets).
pub(crate) use mrr_math::testing::*;

/// Minimal 1D geometry on [0, width) that accepts every sample
pub(crate) struct TestSlab {
    pub width: f64,
    /// Test acceptance range for u\in[0,1) for rejection sampling
    accept: Range<f64>,
}

impl TestSlab {
    pub(crate) fn new(width: f64, accept: Range<f64>) -> Self {
        Self { width, accept }
    }
}

impl Geometry for TestSlab {
    type Point = f64;
    const SAMPLE_DIM: usize = 1;
    fn volume(&self) -> f64 {
        self.width * (self.accept.end - self.accept.start)
    }
    fn map_uniform(&self, u: &[f64]) -> Option<f64> {
        if self.accept.contains(&u[0]) {
            Some(u[0] * self.width)
        } else {
            None
        }
    }
}

/// Gather that just returns a constant block for ach atom
pub(crate) struct ConstGather {
    block: Vec<f64>,
}

impl ConstGather {
    pub(crate) fn new(block: Vec<f64>) -> Self {
        Self { block }
    }
}

impl<G: Geometry> Gather<G> for ConstGather {
    fn gather(
        &self,
        _geometry: &G,
        _src_atoms: &crate::atoms::Atoms<<G as Geometry>::Point>,
        _target_point: <G as Geometry>::Point,
    ) -> Vec<f64> {
        self.block.clone()
    }
}

/// Emission that returns fixed multiples of the flux as its scattering and
/// fission parts. If the gathered flux integrates to V, then
/// k = fission * V / (V - scatter * V) = fission / (1 - scatter).
pub(crate) struct ScaledEmission {
    scatter: f64,
    fission: f64,
}

impl ScaledEmission {
    pub(crate) fn new(scatter: f64, fission: f64) -> Self {
        Self { scatter, fission }
    }
}

impl<G: Geometry> Emission<G> for ScaledEmission {
    fn compute_emission(&self, _geometry: &G, _point: G::Point, flux_moments: &[f64]) -> Emitted {
        Emitted {
            scattering: flux_moments.iter().map(|m| self.scatter * m).collect(),
            fission: flux_moments.iter().map(|m| self.fission * m).collect(),
        }
    }
}
