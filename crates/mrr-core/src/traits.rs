//! Defines interface traits that are used to express the power iteration updates generically.
//!
//! `mrr-core` knows about geometry (where points are, how to sample them) and
//! the algorithm. It deliberately knows nothing about cross sections, energy
//! groups vs. continuous energy, or optical depth: those live entirely inside
//! the `Gather` and `Emission` implementations of a physics crate.
use crate::atoms::Atoms;
use crate::geometry::{MaterialId, Segment};
use crate::sampler::SamplerStreams;

/// The spatial domain: what a point is, its volume, and how to sample it uniformly.
/// Energy-independent by design.
pub trait Geometry {
    type Point: Copy + Clone;

    const SAMPLE_DIM: usize;

    /// Volume of the domain (length in a slab).
    fn volume(&self) -> f64;

    /// Map u ∈ [0,1)^SAMPLE_DIM to a point, uniformly over the domain.
    /// Returns None if the sample is rejected (falls outside the geometry).
    fn map_uniform(&self, u: &[f64]) -> Option<Self::Point>;
}

/// An operation that 'gathers' the angular flux contributions of source atoms into a new source point flux
pub trait Gather<G: Geometry> {
    /// Take the source atoms and gather flux contributions to the target atom (given by its point)
    /// to compute the flux moments
    fn gather(&self, geometry: &G, src_atoms: &Atoms<G::Point>, target_point: G::Point)
    -> Vec<f64>;
}

/// Emission moments at one point, split into scattering and fission parts.
///
/// `fission` is returned *without* the 1/k factor: the driver determines k
/// from the integrals of the two parts and combines them as
/// `scattering + fission / k`. Both blocks use the run's `MomentLayout`.
pub struct Emitted {
    pub scattering: Vec<f64>,
    pub fission: Vec<f64>,
}

/// Computes the emission from a gathered source point flux
pub trait Emission<G: Geometry> {
    /// Compute emission at `point` (used to find the local material) from the
    /// flux moments gathered there
    fn compute_emission(&self, geometry: &G, point: G::Point, flux_moments: &[f64]) -> Emitted;
}

/// Whether an empirical source is already existing or is being initialized, consequences for
/// reuse of Proposal during initialization
pub enum Source<'a, P> {
    Existing(&'a Atoms<P>),
    Initialize(usize),
}

impl<P: Copy + Clone> Source<'_, P> {
    pub fn n_atoms(&self) -> usize {
        match self {
            Source::Existing(atoms) => atoms.len(),
            Source::Initialize(n) => *n,
        }
    }
}

/// Target points sampled by a `Proposal`, with their importance-sampling weights.
/// `points[i]` and `weights[i]` belong to the same target.
pub struct Targets<P> {
    pub points: Vec<P>,
    pub weights: Vec<f64>,
}

/// Proposal over geometry for new source points for importance sampling
pub trait Proposal<G: Geometry> {
    /// Sample target atoms according to a proposal, updating weights according to importance sampling
    fn sample_target_atoms(
        &self,
        geometry: &G,
        source: &Source<G::Point>,
        streams: &impl SamplerStreams,
    ) -> Targets<G::Point>;
}

/// A geometry that knows its materials and can trace straight paths.
pub trait Tracer: Geometry {
    /// Material at `p`, or None outside the domain.
    fn material_at(&self, p: &Self::Point) -> Option<MaterialId>;

    /// Segments of the straight path from `from` to `to`, in order.
    /// Clears `out` first; reuses its allocation across calls.
    fn trace(&self, from: &Self::Point, to: &Self::Point, out: &mut Vec<Segment>);
}
