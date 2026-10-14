//! Energy-agnostic geometry description code

/// Index of a material in a material library. A newtype so it can't be
/// confused with a group or atom index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub usize);

/// A piece of a straight path inside one material
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub material: MaterialId,
    pub length: f64,
}
