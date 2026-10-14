//! Generic, implementation agnostic driver code

pub mod atoms;
pub mod convergence;
pub mod driver;
pub mod geometry;
pub mod init;
pub mod moments;
pub mod proposal;
pub mod sampler;
pub mod traits;

pub use atoms::Atoms;
pub use driver::{Problem, RunConfig, RunResult, power_iterate};
pub use moments::MomentLayout;
pub use traits::{Emission, Emitted, Gather, Geometry, Proposal};

#[cfg(test)]
mod test_utils;
