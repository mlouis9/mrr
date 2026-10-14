//! Shared mathematical, implementation agnostic code.
//!
//! Plain numerics only: nothing here knows about atoms, geometry or cross
//! sections, so every function can be tested against known mathematical facts.

pub mod bins;
pub mod expint;
pub mod group_matrix;
pub mod legendre;
pub mod quadrature;
