//! Native constructive-solid-geometry backend.
//!
//! - [`surface`]: planes, spheres and axis-aligned cylinders with signed
//!   distance, direction-aware sense and ray crossing distance.
//! - [`model`]: cells as intersections of half-spaces, and the [`Model`] that
//!   holds them together with the outer `BoundaryBox`.
//! - [`walker`]: `impl Walker for Model`, passing the conformance suite of
//!   `mrr_geom3d::conformance`.
//!
//! Building a model:
//!
//! ```
//! use mrr_csg::{Axis, Model, Surface};
//! use mrr_geom3d::{Bc, BoundaryBox, MaterialId, Vec3};
//!
//! let bbox = BoundaryBox::uniform(Vec3::new(-0.63, -0.63, 0.0), Vec3::new(0.63, 0.63, 1.0), Bc::Reflective).unwrap();
//! let mut model = Model::new(bbox);
//! let pin = model.add_surface(Surface::cylinder(Axis::Z, Vec3::ZERO, 0.41));
//! model.add_cell(vec![-pin], MaterialId(0)).unwrap();          // fuel inside
//! model.add_cell(vec![pin.positive()], MaterialId(1)).unwrap(); // water outside
//! assert_eq!(model.material_at(Vec3::new(0.1, 0.0, 0.5), Vec3::X), Some(MaterialId(0)));
//! ```

pub mod model;
pub mod surface;
pub mod walker;

pub use model::{Cell, CellId, Halfspace, Model, ModelError, PartitionDefect, SurfaceId};
pub use surface::{Axis, Surface};
