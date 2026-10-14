//! Errors from building slab geometries and problems

use mrr_math::bins::BinsError;
use mrr_xs::XsError;
use thiserror::Error;

/// Why a slab geometry or slab problem could not be built.
#[derive(Debug, Error)]
pub enum SlabError {
    /// The region edges are not a valid partition (too few, non-finite, or
    /// not strictly increasing)
    #[error("slab region edges: {0}")]
    Bins(#[from] BinsError),

    /// There must be exactly one material per region
    #[error("slab has {expected} regions but {got} materials were given")]
    MaterialCount { expected: usize, got: usize },

    /// A cross-section problem found while assembling a slab problem, e.g. the
    /// geometry refers to a material the library does not contain
    #[error("slab problem: {0}")]
    Xs(#[from] XsError),

    /// A region names a material that is not in the library
    #[error("region {region} uses unknown material '{name}'")]
    UnknownMaterialName { region: usize, name: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_core::MaterialId;
    use mrr_math::Bins;

    /// `?` converts lower-level errors into `SlabError` automatically
    fn build_bins(edges: Vec<f64>) -> Result<Bins, SlabError> {
        Ok(Bins::new(edges)?)
    }

    fn check_material(id: MaterialId) -> Result<(), SlabError> {
        Err(XsError::UnknownMaterial(id))?
    }

    #[test]
    fn bins_errors_convert_with_question_mark() {
        assert!(matches!(
            build_bins(vec![0.0, 0.0]),
            Err(SlabError::Bins(BinsError::NotIncreasing { index: 1 }))
        ));
    }

    #[test]
    fn xs_errors_convert_with_question_mark() {
        assert!(matches!(
            check_material(MaterialId(3)),
            Err(SlabError::Xs(XsError::UnknownMaterial(MaterialId(3))))
        ));
    }

    #[test]
    fn messages_are_readable() {
        assert_eq!(
            SlabError::MaterialCount {
                expected: 2,
                got: 3
            }
            .to_string(),
            "slab has 2 regions but 3 materials were given"
        );
        assert_eq!(
            build_bins(vec![1.0]).unwrap_err().to_string(),
            "slab region edges: need at least 2 bin edges, found 1"
        );
        assert_eq!(
            check_material(MaterialId(3)).unwrap_err().to_string(),
            "slab problem: unknown material ID: MaterialId(3)"
        );
    }

    #[test]
    fn underlying_error_is_kept_as_the_source() {
        use std::error::Error;
        let err = build_bins(vec![]).unwrap_err();
        assert!(err.source().is_some());
    }
}
