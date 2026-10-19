//! Geometry concepts shared by every 3D backend: paths and legs, layers,
//! the outer boundary and its images, the overlay grid, shared tolerances, and
//! a conformance suite that any `Walker` implementation must pass.
//!
//! The commonly used types are re-exported at the crate root, so downstream
//! code can write `use mrr_geom3d::{Path, Walker, BoundaryBox, Bc};`.

pub type Vec3 = glam::DVec3;
pub type DMat3 = glam::DMat3;

pub mod affine;
pub mod boundary;
pub mod conformance;
pub mod grid;
pub mod layer;
pub mod testing;
pub mod tol;
pub mod traits;
pub mod walk;

pub use affine::Affine;
pub use boundary::{Bc, BoundaryBox, BoundaryError, Crossed, Crossing, FaceHit, Side};
pub use grid::{CartesianGrid, GridError};
pub use layer::{AxisBins, new_partition};
pub use traits::{Layer, Walker};
pub use walk::{Ended, Leg, Path, material_segments};

// Types that appear in this crate's public signatures, re-exported so users
// do not need direct dependencies on `mrr-core` / `mrr-math` for them.
pub use mrr_core::{Geometry, MaterialId, Segment};
pub use mrr_math::{Partition, Span};
