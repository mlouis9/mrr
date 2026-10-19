//! Geometric tolerances shared by every geometry backend and layer.
//!
//! All lengths are in **cm**. These are properties of the geometry
//! algorithms, not test tolerances (tests use `mrr_math::testing`). Every
//! partition, walker and layer must use these same values; two layers that
//! disagree about what counts as "the same point" produce slivers and
//! inconsistent labels.
//!
//! # Scale of the numbers involved
//!
//! - Reactor coordinates reach a few hundred cm. The spacing of f64 values
//!   (one ulp) at 500 cm is about 500 × 2.2e-16 ≈ 1.1e-13 cm.
//! - A computed point or path length carries a few ulps of error per
//!   operation, and the path length s accumulates error over every crossing.
//!   After ~10³ crossings the worst case is ~10³ ulps ≈ 1e-10 cm; a
//!   random-walk estimate is closer to 3e-12 cm.
//! - Crossing distances to quadric surfaces (spheres, cylinders) are roots of
//!   a quadratic. For rays that meet the surface at a reasonable angle their
//!   error is a few ulps times the surface size. For **grazing** rays the root
//!   becomes ill-conditioned: near tangency the error grows like √ε · R ≈
//!   1.5e-8 · R. No fixed tolerance covers that limit; it is handled by the
//!   direction-aware tie rule below, not by making the tolerances larger.
//! - Physically, 1e-8 cm is about the size of an atom; mean free paths are
//!   ≳ 1e-3 cm even in strong resonances. Anything below ~1e-8 cm is
//!   physically meaningless, so tolerances there cost no accuracy.
//!
//! # The tie policy (one rule for every module)
//!
//! - A point exactly on a surface, grid plane or partition edge belongs to the
//!   region the **direction of travel** points into (for a stationary query,
//!   the half-open convention of `Bins` and `Partition`).
//! - Edges closer than [`LEN`] are the same edge. `Partition::push` absorbs the
//!   later one; labels are never re-derived from midpoint positions.
//! - Crossing a surface moves the *lookup point* [`BUMP`] past it. Partition
//!   edges are still placed at the crossing itself, so `BUMP` never creates or
//!   shortens an interval; it only decides which region is looked up next.

/// Minimum interval length: edges closer than this are merged (cm).
///
/// Chosen at ~10³ ulps of a 500 cm coordinate: above the accumulated rounding
/// of path lengths in realistic walks, so rounding never creates a sliver,
/// and ~100× below atomic size, so merging never changes physics. The worst
/// effect of merging is that an interval shorter than `LEN` takes its
/// neighbour's label, changing an optical depth by at most
/// ΔΣ · LEN ≲ 1e3 cm⁻¹ × 1e-10 cm = 1e-7, far below statistical noise.
pub const LEN: f64 = 1e-10;

/// Distance to step past a crossing before looking up the next region (cm).
///
/// Constraints, all satisfied by 1e-9:
/// - **`BUMP > LEN`** (here 10×): a root closer than `LEN` counts as "already
///   crossed", so after a bump the surface just crossed is never found again
///   and the walk cannot stall.
/// - **`BUMP` exceeds the error of non-grazing crossing distances** (a few
///   ulps × size ≈ 1e-13 to 1e-12 cm), so the bumped point reliably lands on
///   the far side.
/// - **`BUMP` is far below the thinnest physical region** (fuel–clad gaps are
///   ~1e-3 cm), so a bump never skips a whole region.
///
/// Grazing crossings can have larger errors than `BUMP`; those are resolved
/// by evaluating the surface sense with the direction of travel (tie policy),
/// not by increasing `BUMP`.
pub const BUMP: f64 = 1e-9;

// The constraints above, checked at compile time: changing either constant
// so that they are violated is a build error, not a test failure.
const _: () = {
    assert!(LEN > 0.0);
    assert!(BUMP > LEN, "a bump must clear the merge tolerance");
    // Well above the rounding of a 500 cm coordinate (~1.1e-13 cm)
    assert!(LEN > 100.0 * 500.0 * f64::EPSILON);
    // Far below the thinnest physical region (~1e-3 cm)
    assert!(BUMP < 1e-6);
};
