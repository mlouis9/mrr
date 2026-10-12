//! Defines interface traits that are used to express the power iteration updates generically
use crate::atoms::Atoms;
use crate::sampler::SamplerStreams;

/// A medium for the purposes of transport, required for computing optical depths
pub trait Medium {
    type Point: Copy + Clone;

    const SAMPLE_DIM: usize;

    fn n_groups(&self) -> usize;

    /// Optical depth τ(a, b) = ∫ Σ_t,g along the straight line from a to b.
    fn optical_depth(&self, x: &Self::Point, x_prime: &Self::Point) -> f64;

    /// Volume of the domain (length in a slab).
    fn volume(&self) -> f64;

    /// Map u ∈ [0,1)^SAMPLE_DIM to a point, uniformly over the domain.
    /// Returns None if the sample is rejected (falls outside the geometry).
    fn map_uniform(&self, u: &[f64]) -> Option<Self::Point>;
}

/// An operation that 'gathers' the angular flux contributions of source atoms into a new source point flux
pub trait Gather {}

/// Computes the emission from a gathered source point flux
pub trait Emission {}

/// Whether an empirical source is already existing or is being initialized, consequences for reuse of Proposal during initialization
pub enum Source<'a, P> {
    Existing(&'a Atoms<P>),
    Initialize(usize),
}

/// Proposal over medium for new source points for importance sampling
pub trait Proposal<M: Medium> {
    /// Sample target atoms according to a proposal, updating weights according to importance sampling
    fn sample_target_atoms(
        &self,
        medium: &M,
        source: &Source<M::Point>,
        streams: &impl SamplerStreams,
    ) -> (Vec<M::Point>, Vec<f64>);
}
