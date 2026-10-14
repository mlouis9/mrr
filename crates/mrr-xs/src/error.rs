use mrr_core::MaterialId;
use mrr_math::bins::BinsError;
use mrr_math::group_matrix::GroupMatrixError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum XsError {
    #[error(transparent)]
    Bins(#[from] BinsError),

    #[error(transparent)]
    GroupMatrix(#[from] GroupMatrixError),

    #[error("energy must be positive, got {value}")]
    NonPositiveEnergy { value: f64 },

    #[error("material '{material}': field '{field}' has length {got}, expected {expected}")]
    LengthMismatch {
        material: String,
        field: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("material '{material}': scattering data has {got} orders, expected {expected}")]
    ScatterOrder {
        material: String,
        expected: usize,
        got: usize,
    },

    #[error("material '{material}': field '{field}' has a negative value at index {index}")]
    Negative {
        material: String,
        field: &'static str,
        index: usize,
    },

    #[error("material '{material}': field '{field}' has a non-finite value at index {index}")]
    NonFinite {
        material: String,
        field: &'static str,
        index: usize,
    },

    #[error("material '{material}': scattering order {order}: {source}")]
    ScatterMatrix {
        material: String,
        order: usize,
        source: GroupMatrixError,
    },

    #[error("material '{material}' is already in the library")]
    DuplicateName { material: String },

    #[error("material '{material}': fission spectrum chi sums to {sum}, expected 1")]
    ChiNotNormalized { material: String, sum: f64 },

    #[error("unknown material ID: {0:?}")]
    UnknownMaterial(MaterialId),
}
