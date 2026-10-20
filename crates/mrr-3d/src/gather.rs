//! Exact all-pairs gather.
//!
//! For a node `x`, the flux moments are a sum over atoms and over the
//! admissible images of the node (§2, §9):
//!
//! ```text
//! φ_g,ℓm(x) = Σ_t Σ_i  w_i q_i,g(Ω_e) e^{−τ_g} / r² · Ŷ_ℓm(Ω_a),   r < R
//! ```
//!
//! For atom `x_i` and image `x' = image(t, x)`, the path is
//! `walk(x_i, unit(x' − x_i), |x' − x_i|)`; `r = |x' − x_i|` is its length,
//! `Ω_e` / `Ω_a` its first / last directions and `τ_g` its optical depth.
//! [`PointKernel::add_contribution`] does the per-path arithmetic.
//!
//! # Path-length cutoff `R`
//!
//! Pairs with path length `r >= R` are dropped, for every image including the
//! direct one. In a reflective box the image sum is the infinite-medium
//! kernel, so this is exactly the infinite-medium kernel truncated to a ball
//! of radius `R`: it removes a fraction `e^{−Σ_t R}` of each atom's
//! uncollided flux, a known, controllable bias (e.g. the closed-form
//! `k_R = νΣ_f P / (Σ_t − Σ_s P)`, `P = 1 − e^{−Σ_t R}`, for a homogeneous
//! reflective box). With `R = ∞` (or larger than any path) the gather is exact.
//!
//! Images are computed once per node with `images_within(x, R)`: an image
//! farther than `R` from the box is farther than `R` from every atom.
//!
//! # Coincident points
//!
//! A pair with `r <= tol::LEN` (a node on an atom) is skipped. Its
//! contribution is the `1/r²` pole; with continuously sampled nodes it occurs
//! with probability zero.

use mrr_core::traits::Gather;
use mrr_core::{Atoms, MaterialId, MomentLayout, Segment};
use mrr_geom3d::{Ended, Partition, Path, Vec3, Walker, material_segments, new_partition, tol};
use mrr_xs::{MaterialLibrary, optical_depths};

use crate::{AngularBasis, PointKernel};

/// Exact gather over all atoms and all images, with a path-length cutoff
pub struct ExactGather<'a> {
    lib: &'a MaterialLibrary,
    layout: MomentLayout,
    max_order: usize,
    /// Pairs with path length `>= cutoff` are dropped (see module docs)
    cutoff: f64,
}

/// Reusable buffers for one node's gather
struct Scratch {
    kernel: PointKernel,
    path: Path,
    mats: Partition<MaterialId>,
    segs: Vec<Segment>,
    taus: Vec<f64>,
    images: Vec<[i32; 3]>,
}

impl<'a> ExactGather<'a> {
    /// `cutoff`: path-length cutoff `R` (use `f64::INFINITY` for no cutoff in
    /// vacuum geometries; with reflective faces it must be finite)
    pub fn new(lib: &'a MaterialLibrary, cutoff: f64) -> Self {
        assert!(cutoff > 0.0, "cutoff must be positive, got {cutoff}");
        Self {
            lib,
            layout: lib.harmonic_layout(),
            max_order: lib.max_order(),
            cutoff,
        }
    }

    pub fn layout(&self) -> MomentLayout {
        self.layout
    }

    /// A fresh kernel (with its own scratch) for this gather's layout
    pub fn kernel(&self) -> PointKernel {
        PointKernel::new(AngularBasis::new(self.max_order), self.layout)
    }

    pub fn library(&self) -> &MaterialLibrary {
        self.lib
    }

    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    fn scratch(&self) -> Scratch {
        Scratch {
            kernel: self.kernel(),
            path: Path::new(Vec3::ZERO, Vec3::Z),
            mats: new_partition(),
            segs: Vec::new(),
            taus: vec![0.0; self.layout.n_groups()],
            images: Vec::new(),
        }
    }
}

impl<G: Walker> Gather<G> for ExactGather<'_> {
    fn gather(&self, geometry: &G, atoms: &Atoms<Vec3>, target: Vec3) -> Vec<f64> {
        assert_eq!(
            atoms.layout().block_len(),
            self.layout.block_len(),
            "atoms and gather use different moment layouts"
        );
        let bbox = geometry.boundary();
        let mut s = self.scratch();
        let mut phi = self.layout.zero_block();

        // Images of the node that can be within the cutoff of some atom.
        // With all-vacuum faces this is just the identity.
        if self.cutoff.is_finite() {
            bbox.images_within(target, self.cutoff, &mut s.images);
        } else {
            assert!(
                (0..3).all(|t| {
                    let mut e = [0; 3];
                    e[t] = 1;
                    !bbox.admissible(e) && !bbox.admissible([-e[0], -e[1], -e[2]])
                }),
                "an infinite cutoff needs an all-vacuum boundary"
            );
            s.images.push([0, 0, 0]);
        }

        for &t in &s.images {
            let image = bbox.image(t, target);
            for (i, &x_i) in atoms.points().iter().enumerate() {
                let delta = image - x_i;
                let r = delta.length();
                if r <= tol::LEN || r >= self.cutoff {
                    continue;
                }
                let ended = geometry.walk(x_i, delta / r, r, &mut s.path, &mut s.mats);
                if ended != Ended::Reached {
                    continue; // a vacuum face on the way: no such path
                }
                debug_assert!(
                    (s.path.end_point() - target).length() < 1e-8,
                    "walk toward image {t:?} ended at {:?}, not at the node {target:?}",
                    s.path.end_point()
                );
                material_segments(&s.mats, &mut s.segs);
                optical_depths(self.lib, &s.segs, &mut s.taus);
                s.kernel.add_contribution(
                    atoms.weights()[i],
                    atoms.atom_moments(i),
                    s.path.first_dir(),
                    s.path.last_dir(),
                    r,
                    &s.taus,
                    &mut phi,
                );
            }
        }
        phi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point_kernel;
    use crate::problems::{homogeneous_box, one_group_material, two_region_box};
    use mrr_csg::{Model, Surface};
    use mrr_geom3d::testing::*;
    use mrr_geom3d::{Bc, BoundaryBox, MaterialId};
    use mrr_math::harmonics::index;
    use mrr_math::{GaussLegendre, SphereQuadrature};
    use mrr_xs::EnergyGroups;
    use std::f64::consts::PI;

    fn one_group(order: usize, mats: &[(f64, f64, f64)]) -> MaterialLibrary {
        let mut lib = MaterialLibrary::new(EnergyGroups::new(vec![2e7, 1e-5]).unwrap(), order);
        for (k, &(st, ss, nf)) in mats.iter().enumerate() {
            let mut m = one_group_material(&format!("m{k}"), st, ss, nf);
            m.scatter = (0..=order)
                .map(|l| vec![if l == 0 { ss } else { 0.0 }])
                .collect();
            lib.add(m).unwrap();
        }
        lib
    }

    /// Atoms with one atom at `x`, weight `w`, the given moment block
    fn atom(layout: MomentLayout, x: Vec3, w: f64, q: &[f64]) -> Atoms<Vec3> {
        let mut a = Atoms::new(1, layout);
        a.push(x, w, q);
        a
    }

    fn isotropic(layout: MomentLayout, s: f64) -> Vec<f64> {
        let mut q = layout.zero_block();
        q[0] = s;
        q
    }

    // ---------------- single paths by hand ----------------

    #[test]
    fn isotropic_atom_in_vacuum_is_a_pencil() {
        let lib = one_group(1, &[(0.8, 0.0, 0.0)]);
        let model = homogeneous_box(2.0, Bc::Vacuum);
        let g = ExactGather::new(&lib, f64::INFINITY);
        let x_a = Vec3::new(-0.5, 0.2, 0.1);
        let node = x_a + Vec3::new(0.0, 0.6, 0.8) * 1.5; // r = 1.5 along (0, .6, .8)
        let phi = g.gather(
            &model,
            &atom(g.layout(), x_a, 0.5, &isotropic(g.layout(), 3.0)),
            node,
        );
        let phi00 = 0.5 * 3.0 / (4.0 * PI) * point_kernel(1.5, 0.8 * 1.5);
        assert_close(phi[0], phi00);
        assert_close(phi[index(1, -1)], phi00 * 0.6);
        assert_close(phi[index(1, 0)], phi00 * 0.8);
        assert_close_tol(phi[index(1, 1)], 0.0, Tol::NUMERICAL);
    }

    #[test]
    fn optical_depth_crosses_regions() {
        // Σ = 0.5 for x < 0, 2.0 for x > 0; path from x = -0.4 to x = 0.3 along x
        let lib = one_group(0, &[(0.5, 0.0, 0.0), (2.0, 0.0, 0.0)]);
        let model = two_region_box(1.0, Bc::Vacuum);
        let g = ExactGather::new(&lib, f64::INFINITY);
        let phi = g.gather(
            &model,
            &atom(g.layout(), Vec3::new(-0.4, 0.0, 0.0), 1.0, &[1.0]),
            Vec3::new(0.3, 0.0, 0.0),
        );
        let tau: f64 = 0.5 * 0.4 + 2.0 * 0.3;
        assert_close(phi[0], (-tau).exp() / (4.0 * PI * 0.49));
    }

    #[test]
    fn contributions_add_over_atoms() {
        let lib = one_group(1, &[(1.0, 0.0, 0.0)]);
        let model = homogeneous_box(2.0, Bc::Vacuum);
        let g = ExactGather::new(&lib, f64::INFINITY);
        let l = g.layout();
        let node = Vec3::new(0.3, -0.2, 0.5);
        let (a, b) = (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.5, 1.0, -0.5));
        let (qa, qb) = ([1.0, 0.1, -0.2, 0.3], [2.0, 0.0, 0.4, 0.0]);
        let mut both = Atoms::new(2, l);
        both.push(a, 0.7, &qa);
        both.push(b, 1.3, &qb);
        let pa = g.gather(&model, &atom(l, a, 0.7, &qa), node);
        let pb = g.gather(&model, &atom(l, b, 1.3, &qb), node);
        let pab = g.gather(&model, &both, node);
        for k in 0..4 {
            assert_close(pab[k], pa[k] + pb[k]);
        }
    }

    #[test]
    fn point_reflection_flips_odd_moments() {
        // Isotropic atom at the centre: φ at x_a + d and x_a − d differ by (−1)^ℓ
        let lib = one_group(2, &[(0.7, 0.0, 0.0)]);
        let model = homogeneous_box(2.0, Bc::Vacuum);
        let g = ExactGather::new(&lib, f64::INFINITY);
        let at = atom(g.layout(), Vec3::ZERO, 1.0, &isotropic(g.layout(), 1.0));
        let d = Vec3::new(0.3, -0.5, 0.4);
        let (p, m) = (g.gather(&model, &at, d), g.gather(&model, &at, -d));
        for k in 0..9 {
            let l = mrr_math::harmonics::lm(k).0;
            let sign = if l.is_multiple_of(2) { 1.0 } else { -1.0 };
            assert_close(m[k], sign * p[k]);
        }
    }

    #[test]
    fn one_reflective_face_adds_exactly_one_image() {
        // Box [-1, 1]³, reflective only at x = +1. Atom and node on the x axis.
        let bbox = BoundaryBox::new(
            Vec3::splat(-1.0),
            Vec3::ONE,
            [
                [Bc::Vacuum, Bc::Reflective],
                [Bc::Vacuum; 2],
                [Bc::Vacuum; 2],
            ],
        )
        .unwrap();
        let mut model = Model::new(bbox);
        model.add_cell(vec![], MaterialId(0)).unwrap();
        let sigma = 0.6;
        let lib = one_group(1, &[(sigma, 0.0, 0.0)]);
        let g = ExactGather::new(&lib, 50.0);
        let (x_a, node) = (Vec3::new(-0.2, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0));
        // q_10 = 0 here, but q_1,1 (x-component) makes emission direction matter
        let mut q = g.layout().zero_block();
        q[0] = 1.0;
        q[index(1, 1)] = 0.5;
        let phi = g.gather(&model, &atom(g.layout(), x_a, 1.0, &q), node);

        // q(Ω) = 1/(4π) + 3/(4π) · 0.5 · Ω_x
        let q_of = |ox: f64| 1.0 / (4.0 * PI) + 3.0 / (4.0 * PI) * 0.5 * ox;
        // Direct: r = 0.7 along +x, arriving +x
        let direct = q_of(1.0) * point_kernel(0.7, sigma * 0.7);
        // Image in x = 1: x' = 1.5, r = 1.7, emitted +x, arriving −x
        let image = q_of(1.0) * point_kernel(1.7, sigma * 1.7);
        assert_close(phi[0], direct + image);
        assert_close(phi[index(1, 1)], direct - image);
    }

    #[test]
    fn cutoff_drops_long_paths() {
        let lib = one_group(0, &[(1.0, 0.0, 0.0)]);
        let model = homogeneous_box(2.0, Bc::Vacuum);
        let at = atom(lib.harmonic_layout(), Vec3::ZERO, 1.0, &[1.0]);
        let node = Vec3::new(1.0, 0.0, 0.0);
        assert!(ExactGather::new(&lib, 1.5).gather(&model, &at, node)[0] > 0.0);
        assert_eq!(
            ExactGather::new(&lib, 0.9).gather(&model, &at, node)[0],
            0.0
        );
    }

    #[test]
    fn reflective_box_converges_as_the_cutoff_grows() {
        // Pure absorber: the image sum converges, the change from R to R'
        // is bounded by the uncollided flux beyond R
        let sigma = 1.5;
        let lib = one_group(0, &[(sigma, 0.0, 0.0)]);
        let model = homogeneous_box(0.5, Bc::Reflective);
        let at = atom(
            lib.harmonic_layout(),
            Vec3::new(0.1, -0.2, 0.05),
            1.0,
            &[1.0],
        );
        let node = Vec3::new(-0.3, 0.25, 0.2);
        let phi = |r: f64| ExactGather::new(&lib, r).gather(&model, &at, node)[0];
        let (p4, p6, p8) = (phi(4.0), phi(6.0), phi(8.0));
        assert!(p4 < p6 && p6 < p8);
        assert!((p8 - p6) < (p6 - p4));
        // Tail beyond R of an atom in an infinite medium, spread over the box volume
        // (1 cm³): at most e^{−ΣR}/Σ in total, so the node value changes by ~that much
        assert!(p8 - p6 < (-sigma * 6.0f64).exp() / sigma);
    }

    #[test]
    fn node_on_an_atom_is_skipped() {
        let lib = one_group(0, &[(1.0, 0.0, 0.0)]);
        let model = homogeneous_box(1.0, Bc::Vacuum);
        let g = ExactGather::new(&lib, f64::INFINITY);
        let at = atom(g.layout(), Vec3::ZERO, 1.0, &[1.0]);
        assert_eq!(g.gather(&model, &at, Vec3::ZERO), vec![0.0]);
    }

    #[test]
    #[should_panic(expected = "all-vacuum")]
    fn infinite_cutoff_with_reflection_panics() {
        let lib = one_group(0, &[(1.0, 0.0, 0.0)]);
        let model = homogeneous_box(1.0, Bc::Reflective);
        let g = ExactGather::new(&lib, f64::INFINITY);
        g.gather(
            &model,
            &atom(g.layout(), Vec3::ZERO, 1.0, &[1.0]),
            Vec3::X * 0.5,
        );
    }

    // ---------------- the gather's integral, deterministically ----------------

    /// Ball of radius `r0` of an absorber (Σ) in void, inside a vacuum box
    fn absorbing_ball(r0: f64) -> Model {
        let bbox = BoundaryBox::uniform(Vec3::splat(-2.0), Vec3::splat(2.0), Bc::Vacuum).unwrap();
        let mut m = Model::new(bbox);
        let s = m.add_surface(Surface::sphere(Vec3::ZERO, r0));
        m.add_cell(vec![-s], MaterialId(0)).unwrap();
        m.add_cell(vec![s.positive()], MaterialId(1)).unwrap();
        m
    }

    /// ∫_ball φ_k(x) dV in atom-centred coordinates (dV = r² dr dΩ), which
    /// cancels the 1/r² pole: the deterministic analogue of pole matching
    fn ball_integral(g: &ExactGather, model: &Model, at: &Atoms<Vec3>, r0: f64) -> Vec<f64> {
        let gl = GaussLegendre::new(24);
        let sq = SphereQuadrature::product(6, 12);
        let mut total = g.layout().zero_block();
        for (d, wd) in sq.iter() {
            let dir = Vec3::from_array(d);
            for (&x, &wx) in gl.nodes.iter().zip(&gl.weights) {
                let r = 0.5 * r0 * (x + 1.0);
                let phi = g.gather(model, at, r * dir);
                for (t, p) in total.iter_mut().zip(&phi) {
                    *t += wd * 0.5 * r0 * wx * r * r * p;
                }
            }
        }
        total
    }

    #[test]
    fn integral_over_the_ball_is_emission_times_absorption_probability() {
        // Atom at the centre of an absorbing ball: ∫ φ_ℓm dV = q_ℓm (1 − e^{−Σ r0})/Σ
        let (sigma, r0) = (1.3, 1.0);
        let lib = one_group(1, &[(sigma, 0.0, 0.0), (0.0, 0.0, 0.0)]);
        let model = absorbing_ball(r0);
        let g = ExactGather::new(&lib, f64::INFINITY);
        let q = [2.0, 0.3, -0.4, 0.5];
        let at = atom(g.layout(), Vec3::ZERO, 1.0, &q);
        let integral = ball_integral(&g, &model, &at, r0);
        let p = (1.0 - (-sigma * r0).exp()) / sigma;
        let expected: Vec<f64> = q.iter().map(|qk| qk * p).collect();
        assert_all_close_tol(&integral, &expected, Tol::NUMERICAL);
    }
}
