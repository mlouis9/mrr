//! Small 3D test problems with analytic or easily checked answers.
//!
//! - [`homogeneous_box`]: one material in a cube. With all faces reflective
//!   it is an infinite medium, so `k = k_inf = νΣ_f / Σ_a` exactly (one group)
//!   and the scalar flux is spatially flat. This is the first validation
//!   target for the 3D gather, and it exercises the method of images heavily.
//! - [`two_region_box`]: two materials split at `x = 0`, for lookups and
//!   emission tests.

use mrr_csg::{Axis, Model, Surface};
use mrr_geom3d::{Bc, BoundaryBox, MaterialId, Vec3};
use mrr_xs::MgMaterialData;

/// One-group isotropic material from Σ_t, Σ_s, νΣ_f (χ = 1)
pub fn one_group_material(
    name: &str,
    sigma_t: f64,
    sigma_s: f64,
    nu_sigma_f: f64,
) -> MgMaterialData {
    MgMaterialData {
        name: name.into(),
        sigma_t: vec![sigma_t],
        scatter: vec![vec![sigma_s]],
        nu_sigma_f: vec![nu_sigma_f],
        chi: vec![1.0],
    }
}

/// The cube `[−h, h]³` filled with material 0, with the same condition on
/// every face
pub fn homogeneous_box(half_width: f64, bc: Bc) -> Model {
    let h = half_width;
    let bbox = BoundaryBox::uniform(Vec3::splat(-h), Vec3::splat(h), bc).expect("valid box");
    let mut m = Model::new(bbox);
    m.add_cell(vec![], MaterialId(0)).expect("valid cell");
    m
}

/// The cube `[−h, h]³` with material 0 for `x < 0` and material 1 for `x > 0`
pub fn two_region_box(half_width: f64, bc: Bc) -> Model {
    let h = half_width;
    let bbox = BoundaryBox::uniform(Vec3::splat(-h), Vec3::splat(h), bc).expect("valid box");
    let mut m = Model::new(bbox);
    let x0 = m.add_surface(Surface::axis_plane(Axis::X, 0.0));
    m.add_cell(vec![-x0], MaterialId(0)).expect("valid cell");
    m.add_cell(vec![x0.positive()], MaterialId(1))
        .expect("valid cell");
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_geom3d::Walker;

    #[test]
    fn boxes_are_built_as_described() {
        let m = homogeneous_box(1.5, Bc::Reflective);
        assert_eq!(
            m.material_at(Vec3::new(1.4, -1.4, 0.0), Vec3::X),
            Some(MaterialId(0))
        );
        assert_eq!(m.boundary().size(), Vec3::splat(3.0));
        let t = two_region_box(1.0, Bc::Vacuum);
        assert_eq!(
            Walker::material_at(&t, Vec3::new(-0.5, 0.0, 0.0), Vec3::X),
            Some(MaterialId(0))
        );
        assert_eq!(
            Walker::material_at(&t, Vec3::new(0.5, 0.0, 0.0), Vec3::X),
            Some(MaterialId(1))
        );
        let mat = one_group_material("u", 1.0, 0.4, 0.9);
        assert_eq!(
            (mat.sigma_t[0], mat.scatter[0][0], mat.nu_sigma_f[0]),
            (1.0, 0.4, 0.9)
        );
    }
}
