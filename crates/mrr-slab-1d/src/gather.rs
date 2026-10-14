//! Gather estimators for the slab: flux moments at a target from all atoms.
//!
//! Only the exact angular gather (a) exists so far. The stochastic angular
//! (b) and shared-direction ray (c) variants will live next to it.
pub mod exact;

pub use exact::ExactGather;
