//! Cross-section data: energy structures, materials and the material library.
//! Dimension-independent: used by any geometry (1D slab now, 3D later).

pub mod energy;
pub mod error;
pub mod library;
pub mod material;
pub mod optical;

pub use energy::EnergyGroups;
pub use error::XsError;
pub use library::MaterialLibrary;
pub use material::{MgMaterial, MgMaterialData};
