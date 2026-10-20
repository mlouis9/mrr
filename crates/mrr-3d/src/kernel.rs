//! The exact point-to-point kernel (§2 of the method description).
//!
//! An atom at `x_i` with weight `w` and moments `q_g,ℓm` emits
//! `w q_g(Ω)` per unit solid angle in direction Ω. Its uncollided angular
//! flux at a node `x` reached by a (possibly reflected) path of length `r`,
//! leaving the atom along `Ω_e` and arriving along `Ω_a`, is the pencil beam
//!
//! ```text
//! ψ_g(x, Ω) = w q_g(Ω_e) e^{−τ_g} / r² · δ(Ω − Ω_a)
//! ```
//!
//! so its contribution to the node's flux moments is
//!
//! ```text
//! φ_g,ℓm(x) += w q_g(Ω_e) e^{−τ_g} / r² · Ŷ_ℓm(Ω_a)
//! ```
//!
//! - `r` is the **path** length (through reflections), so `1/r²` is the
//!   spreading of the unfolded straight ray.
//! - `Ω_e` (the path's first direction) evaluates the atom's emission; `Ω_a`
//!   (its last direction) projects onto the node's moments. Without
//!   reflections they coincide.
//! - `τ_g` is the optical depth along the path in group `g`.
//!
//! The kernel integrates to the right total: for an isotropic atom (`q_00 = S`)
//! in an infinite medium, `∫ φ_00 dV = ∫ 4π r² · S e^{−Σr}/(4π r²) dr = S/Σ`.
//! It is singular as `1/r²` at the atom; its integral is finite but its
//! square is not, which is why node placement must be pole-matched (§8).

use mrr_core::MomentLayout;
use mrr_geom3d::Vec3;

use crate::AngularBasis;

/// Uncollided point-source kernel `e^{−τ} / r²` (per unit emission per unit
/// solid angle)
pub fn point_kernel(r: f64, tau: f64) -> f64 {
    debug_assert!(r > 0.0, "point kernel at zero distance");
    (-tau).exp() / (r * r)
}

/// Adds atom contributions to flux moments. Holds the angular basis, the
/// moment layout and scratch space, so one per thread (or per chunk).
#[derive(Debug, Clone)]
pub struct PointKernel {
    basis: AngularBasis,
    layout: MomentLayout,
    y_emit: Vec<f64>,
    y_arrive: Vec<f64>,
}

impl PointKernel {
    /// `layout` must have `(L+1)²` moments for the basis order `L`
    pub fn new(basis: AngularBasis, layout: MomentLayout) -> Self {
        assert_eq!(
            layout.n_moments(),
            basis.n_moments(),
            "layout has {} moments but the basis of order {} needs {}",
            layout.n_moments(),
            basis.max_order(),
            basis.n_moments()
        );
        let n = basis.n_moments();
        Self {
            basis,
            layout,
            y_emit: vec![0.0; n],
            y_arrive: vec![0.0; n],
        }
    }

    pub fn layout(&self) -> MomentLayout {
        self.layout
    }

    pub fn basis(&self) -> &AngularBasis {
        &self.basis
    }

    /// Add one atom's contribution, along one path, to the node's flux block:
    ///
    /// `phi[g, ℓm] += weight · q_g(emit_dir) · e^{−τ_g}/r² · Ŷ_ℓm(arrive_dir)`
    ///
    /// - `atom_block`: the atom's full moment block (all groups)
    /// - `taus[g]`: optical depth of the path in group `g`
    /// - `r`: path length (> 0)
    #[allow(clippy::too_many_arguments)]
    pub fn add_contribution(
        &mut self,
        weight: f64,
        atom_block: &[f64],
        emit_dir: Vec3,
        arrive_dir: Vec3,
        r: f64,
        taus: &[f64],
        phi: &mut [f64],
    ) {
        let layout = self.layout;
        debug_assert_eq!(atom_block.len(), layout.block_len());
        debug_assert_eq!(phi.len(), layout.block_len());
        assert_eq!(taus.len(), layout.n_groups(), "one optical depth per group");
        assert!(r > 0.0, "contribution at zero distance (node on an atom)");

        self.basis.eval(emit_dir, &mut self.y_emit);
        self.basis.eval(arrive_dir, &mut self.y_arrive);
        let inv_r2 = 1.0 / (r * r);
        for (g, &tau) in taus.iter().enumerate() {
            let range = layout.group_range(g);
            let emitted = self
                .basis
                .reconstruct(&atom_block[range.clone()], &self.y_emit);
            if emitted == 0.0 {
                continue;
            }
            let value = weight * emitted * (-tau).exp() * inv_r2;
            self.basis
                .add_pencil(value, &self.y_arrive, &mut phi[range]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_geom3d::testing::*;
    use mrr_math::GaussLegendre;
    use mrr_math::harmonics::index;
    use std::f64::consts::PI;

    fn kernel(order: usize, groups: usize) -> PointKernel {
        PointKernel::new(
            AngularBasis::new(order),
            MomentLayout::from_harmonic_order(order, groups),
        )
    }

    #[test]
    fn isotropic_atom_gives_pencil_moments() {
        // q_00 = S per group, weight w: φ_00 = w S e^{−τ}/(4π r²), φ_ℓm = φ_00 Ŷ_ℓm(Ω)
        let mut k = kernel(2, 1);
        let mut q = k.layout().zero_block();
        q[0] = 3.0;
        let d = Vec3::new(0.0, 0.6, 0.8);
        let mut phi = k.layout().zero_block();
        k.add_contribution(0.5, &q, d, d, 2.0, &[0.7], &mut phi);
        let phi00 = 0.5 * 3.0 * (-0.7f64).exp() / (4.0 * PI * 4.0);
        assert_close(phi[0], phi00);
        assert_close(phi[index(1, -1)], phi00 * 0.6);
        assert_close(phi[index(1, 0)], phi00 * 0.8);
        assert_close(phi[index(2, 0)], phi00 * 0.5 * (3.0 * 0.64 - 1.0));
    }

    #[test]
    fn emission_is_evaluated_at_the_emission_direction() {
        // q_10 = a: q(Ω) = 3a z/(4π). Emitted along +z, arriving along +x
        // (as after reflections): the strength uses +z, the moments use +x.
        let mut k = kernel(1, 1);
        let mut q = k.layout().zero_block();
        q[index(1, 0)] = 1.0;
        let mut phi = k.layout().zero_block();
        k.add_contribution(1.0, &q, Vec3::Z, Vec3::X, 1.0, &[0.0], &mut phi);
        let value = 3.0 / (4.0 * PI);
        assert_all_close(&phi, &[value, 0.0, 0.0, value]);
        // Emitted sideways: nothing
        let mut phi = k.layout().zero_block();
        k.add_contribution(1.0, &q, Vec3::X, Vec3::X, 1.0, &[0.0], &mut phi);
        assert_all_close(&phi, &[0.0; 4]);
    }

    #[test]
    fn groups_attenuate_independently() {
        let mut k = kernel(0, 2);
        let q = vec![1.0, 2.0];
        let mut phi = vec![0.0; 2];
        k.add_contribution(1.0, &q, Vec3::Z, Vec3::Z, 1.0, &[0.0, 1.0], &mut phi);
        assert_close(phi[0], 1.0 / (4.0 * PI));
        assert_close(phi[1], 2.0 * (-1.0f64).exp() / (4.0 * PI));
    }

    #[test]
    fn contributions_accumulate() {
        let mut k = kernel(1, 1);
        let mut q = k.layout().zero_block();
        q[0] = 1.0;
        let mut phi = k.layout().zero_block();
        k.add_contribution(1.0, &q, Vec3::Z, Vec3::Z, 1.0, &[0.0], &mut phi);
        k.add_contribution(1.0, &q, -Vec3::Z, -Vec3::Z, 1.0, &[0.0], &mut phi);
        // Opposite pencils: scalar flux doubles, current cancels
        assert_close(phi[0], 2.0 / (4.0 * PI));
        assert_close_tol(phi[index(1, 0)], 0.0, Tol::NUMERICAL);
    }

    #[test]
    fn infinite_medium_total_flux_is_source_over_sigma() {
        // ∫ φ_00 dV over a ball of radius R = S/Σ (1 − e^{−ΣR}), computed by
        // radial quadrature of the kernel (angle is trivial for an isotropic atom)
        let (s, sigma, big_r) = (2.0, 0.8, 30.0);
        let gl = GaussLegendre::new(200);
        let total: f64 = gl.integrate(|x| {
            let r = 0.5 * big_r * (x + 1.0);
            0.5 * big_r * 4.0 * PI * r * r * s / (4.0 * PI) * point_kernel(r, sigma * r)
        });
        assert_close_tol(
            total,
            s / sigma * (1.0 - (-sigma * big_r).exp()),
            Tol::NUMERICAL,
        );
    }

    #[test]
    #[should_panic(expected = "needs")]
    fn mismatched_layout_panics() {
        PointKernel::new(
            AngularBasis::new(2),
            MomentLayout::from_legendre_order(2, 1),
        );
    }

    #[test]
    #[should_panic(expected = "zero distance")]
    fn coincident_points_panic() {
        let mut k = kernel(0, 1);
        k.add_contribution(1.0, &[1.0], Vec3::Z, Vec3::Z, 0.0, &[0.0], &mut [0.0]);
    }
}
