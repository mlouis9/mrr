//! The 3D meshless random ray method: angular basis, point kernel, emission,
//! gather and test problems, on top of any `mrr_geom3d::Walker` geometry.
//!
//! Conventions (shared with `mrr_math::harmonics`): angular moments are
//! projections on 4π-normalized real harmonics, `f_ℓm = ∫ f Ŷ_ℓm dΩ`, so the
//! (0,0) moment is the total and scattering is diagonal in (ℓ, m). The layout
//! of every atom's block is `MaterialLibrary::harmonic_layout()`:
//! `(L+1)²` moments × G groups, flat index `ℓ² + ℓ + m`.

pub mod analog;
pub mod angular;
pub mod emission;
pub mod gather;
pub mod kernel;
pub mod problems;
pub mod proposal;
pub mod vacuum_cube;

pub use angular::AngularBasis;
pub use emission::HarmonicEmission;
pub use gather::ExactGather;
pub use kernel::{PointKernel, point_kernel};
pub use proposal::MixtureProposal;
