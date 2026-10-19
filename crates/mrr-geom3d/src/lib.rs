//! Abstract geometry concepts and conformance tests

pub type Vec3 = glam::DVec3;
pub type DMat3 = glam::DMat3;

pub mod affine;
pub mod boundary;
pub mod grid;
pub mod layer;
pub mod testing;
pub mod tol;
pub mod traits;
pub mod walk;
