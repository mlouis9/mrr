//! Native constructive-solid-geometry backend.
//!
//! - [`surface`]: planes, spheres and axis-aligned cylinders with signed
//!   distance, direction-aware sense and ray crossing distance.
//! - [`model`]: cells as intersections of half-spaces, and the [`Model`] that
//!   holds them together with the outer `BoundaryBox`.
//!
//! The walker (`impl Walker for Model`) is built on these pieces.

pub mod model;
pub mod surface;

pub use model::{Cell, CellId, Halfspace, Model, ModelError, SurfaceId};
pub use surface::{Axis, Surface};
