//! Shared geometry traits

use crate::{BoundaryBox, Ended, Leg, MaterialId, Partition, Path, Vec3};

/// A geometry backend that can trace straight paths with boundary conditions.
///
/// Implementations must pass [`crate::conformance`].
pub trait Walker {
    /// The outer boundary (and with it the image set for reflections)
    fn boundary(&self) -> &BoundaryBox;

    /// Material at `p`, `None` outside the domain; a point on a surface takes
    /// the side `dir` points into
    fn material_at(&self, p: Vec3, dir: Vec3) -> Option<MaterialId>;

    /// Walk from `origin` along unit `dir` for path length up to `s_max`,
    /// following the boundary conditions. Fills `path` (legs, plus how the walk
    /// ended) and `materials` (material intervals over `[0, path.s_end()]`),
    /// both reset first, and returns the same [`Ended`] stored in `path`.
    ///
    /// A shadow ray to image `t` of a node is
    /// `walk(x, unit(x' - x), |x' - x|)` with `x' = boundary().image(t, node)`.
    fn walk(
        &self,
        origin: Vec3,
        dir: Vec3,
        s_max: f64,
        path: &mut Path,
        materials: &mut Partition<MaterialId>,
    ) -> Ended;
}

/// A labelled partition of space that can segment straight legs
pub trait Layer {
    type Label: Copy;

    /// Label of the region containing `p`, with ties broken by `dir`
    fn locate(&self, p: Vec3, dir: Vec3) -> Option<Self::Label>;

    /// Append the labelled intervals of `leg` to `out` (see the module docs)
    fn segment_leg(&self, leg: &Leg, out: &mut Partition<Self::Label>);

    /// Partition of a whole path: `out` is reset to the path's start and each
    /// leg is segmented in turn. A path with no legs gives an empty partition.
    fn segment_path(&self, path: &Path, out: &mut Partition<Self::Label>) {
        out.reset(path.s_start());
        for leg in path.legs() {
            self.segment_leg(leg, out);
        }
    }
}
