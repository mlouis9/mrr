//! Shared geometry traits

use crate::Vec3;
use crate::boundary::BoundaryBox;
use crate::walk::{Ended, Leg, Path};
use mrr_core::geometry::MaterialId;
use mrr_math::Partition;

pub trait Walker {
    fn boundary(&self) -> &BoundaryBox;

    fn material_at(&self, p: Vec3) -> MaterialId;

    fn walk(
        &self,
        origin: Vec3,
        dir: Vec3,
        s_max: f64,
        path: &mut Path,
        partition: &mut Partition<f64>,
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
