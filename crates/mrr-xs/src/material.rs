//! Multigroup materials: raw input data and validated materials

use crate::error::XsError;
use mrr_math::GroupMatrix;

/// Tolerance on sum(chi) = 1 for fissile materials
const CHI_SUM_TOL: f64 = 1e-10;

/// Unvalidated input: what tests and the TOML reader fill in.
#[derive(Debug, Clone)]
pub struct MgMaterialData {
    pub name: String,
    /// Total cross section per group [G]
    pub sigma_t: Vec<f64>,
    /// Scattering matrices for l = 0..=L, each flat G*G with `data[from*G + to]`
    pub scatter: Vec<Vec<f64>>,
    /// nu * fission cross section per group [G]
    pub nu_sigma_f: Vec<f64>,
    /// Fission spectrum [G] (sums to 1 if fissile; may be all zero otherwise)
    pub chi: Vec<f64>,
}

/// Validated material. Fields private: if you hold one, it passed every check.
///
/// Guarantees (for the G and L it was validated against):
/// - `sigma_t`, `nu_sigma_f`, `chi` have length G, are finite and non-negative
/// - there are L + 1 scattering matrices, each G x G and finite (higher-order
///   moments may be negative, so scattering is not sign-checked)
/// - if any `nu_sigma_f > 0`, `chi` sums to 1
///
/// Not checked on purpose: sum_to sigma_s0(from -> to) may exceed sigma_t,
/// which is legitimate with (n,2n) reactions.
#[derive(Debug, Clone)]
pub struct MgMaterial {
    name: String,
    sigma_t: Vec<f64>,
    scatter: Vec<GroupMatrix>,
    nu_sigma_f: Vec<f64>,
    chi: Vec<f64>,
}

impl MgMaterial {
    /// Validate raw data against `n_groups` groups and scattering order
    /// `max_order`. Takes `data` by value and moves its vectors in (no copies).
    pub fn from_data(
        data: MgMaterialData,
        n_groups: usize,
        max_order: usize,
    ) -> Result<Self, XsError> {
        let MgMaterialData {
            name,
            sigma_t,
            scatter,
            nu_sigma_f,
            chi,
        } = data;

        // 1. Lengths
        check_len(&name, "sigma_t", &sigma_t, n_groups)?;
        check_len(&name, "nu_sigma_f", &nu_sigma_f, n_groups)?;
        check_len(&name, "chi", &chi, n_groups)?;

        // 2. Scattering order
        if scatter.len() != max_order + 1 {
            return Err(XsError::ScatterOrder {
                material: name,
                expected: max_order + 1,
                got: scatter.len(),
            });
        }

        // 3. Scattering matrices (keep the material name and order in the error)
        let scatter = scatter
            .into_iter()
            .enumerate()
            .map(|(order, flat)| {
                GroupMatrix::new(n_groups, flat).map_err(|source| XsError::ScatterMatrix {
                    material: name.clone(),
                    order,
                    source,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 4. Finite everywhere
        check_finite(&name, "sigma_t", &sigma_t)?;
        check_finite(&name, "nu_sigma_f", &nu_sigma_f)?;
        check_finite(&name, "chi", &chi)?;
        for m in &scatter {
            check_finite(&name, "scatter", m.as_slice())?;
        }

        // 5. Signs (not for scattering: higher moments are legitimately negative)
        check_non_negative(&name, "sigma_t", &sigma_t)?;
        check_non_negative(&name, "nu_sigma_f", &nu_sigma_f)?;
        check_non_negative(&name, "chi", &chi)?;

        // 6. Fission spectrum normalization, only for fissile materials
        if nu_sigma_f.iter().any(|&v| v > 0.0) {
            let sum: f64 = chi.iter().sum();
            if (sum - 1.0).abs() > CHI_SUM_TOL {
                return Err(XsError::ChiNotNormalized {
                    material: name,
                    sum,
                });
            }
        }

        Ok(Self {
            name,
            sigma_t,
            scatter,
            nu_sigma_f,
            chi,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn n_groups(&self) -> usize {
        self.sigma_t.len()
    }

    /// Highest Legendre order of the scattering data
    pub fn max_order(&self) -> usize {
        self.scatter.len() - 1
    }

    /// Total cross section for every group
    pub fn sigma_t(&self) -> &[f64] {
        &self.sigma_t
    }

    /// Scattering matrix of Legendre order `l` (`get(from, to)`)
    pub fn scatter(&self, l: usize) -> &GroupMatrix {
        &self.scatter[l]
    }

    pub fn nu_sigma_f(&self) -> &[f64] {
        &self.nu_sigma_f
    }

    pub fn chi(&self) -> &[f64] {
        &self.chi
    }

    pub fn is_fissile(&self) -> bool {
        self.nu_sigma_f.iter().any(|&v| v > 0.0)
    }

    /// Total l = 0 scattering out of group `from` (summed over destination groups)
    pub fn scatter_out(&self, from: usize) -> f64 {
        self.scatter[0].row_sum(from)
    }
}

fn check_len(material: &str, field: &'static str, v: &[f64], n: usize) -> Result<(), XsError> {
    if v.len() != n {
        return Err(XsError::LengthMismatch {
            material: material.to_string(),
            field,
            expected: n,
            got: v.len(),
        });
    }
    Ok(())
}

fn check_finite(material: &str, field: &'static str, v: &[f64]) -> Result<(), XsError> {
    match v.iter().position(|x| !x.is_finite()) {
        Some(index) => Err(XsError::NonFinite {
            material: material.to_string(),
            field,
            index,
        }),
        None => Ok(()),
    }
}

fn check_non_negative(material: &str, field: &'static str, v: &[f64]) -> Result<(), XsError> {
    match v.iter().position(|&x| x < 0.0) {
        Some(index) => Err(XsError::Negative {
            material: material.to_string(),
            field,
            index,
        }),
        None => Ok(()),
    }
}

/// A valid 2-group, L = 1 fissile material for tests in this crate.
/// Non-symmetric scattering so transposed indices show up.
#[cfg(test)]
pub(crate) fn test_data(name: &str) -> MgMaterialData {
    MgMaterialData {
        name: name.to_string(),
        sigma_t: vec![0.5, 1.2],
        scatter: vec![
            vec![0.3, 0.1, 0.0, 0.9],     // l = 0: 0->0, 0->1, 1->0, 1->1
            vec![0.05, -0.01, 0.0, 0.02], // l = 1 (negative entry is allowed)
        ],
        nu_sigma_f: vec![0.01, 0.2],
        chi: vec![1.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrr_math::group_matrix::GroupMatrixError;

    fn build(data: MgMaterialData) -> Result<MgMaterial, XsError> {
        MgMaterial::from_data(data, 2, 1)
    }

    #[test]
    fn valid_material_round_trips() {
        let m = build(test_data("fuel")).unwrap();
        assert_eq!(m.name(), "fuel");
        assert_eq!(m.n_groups(), 2);
        assert_eq!(m.max_order(), 1);
        assert_eq!(m.sigma_t(), &[0.5, 1.2]);
        assert_eq!(m.nu_sigma_f(), &[0.01, 0.2]);
        assert_eq!(m.chi(), &[1.0, 0.0]);
        assert!(m.is_fissile());
    }

    #[test]
    fn scattering_uses_from_to_convention() {
        let m = build(test_data("fuel")).unwrap();
        assert_eq!(m.scatter(0).get(0, 1), 0.1); // downscatter fast -> thermal
        assert_eq!(m.scatter(0).get(1, 0), 0.0); // no upscatter
        assert_eq!(m.scatter(1).get(0, 1), -0.01);
        assert!((m.scatter_out(0) - 0.4).abs() < 1e-15);
    }

    #[test]
    fn wrong_vector_lengths_name_the_field() {
        let mut d = test_data("fuel");
        d.sigma_t.push(1.0);
        assert!(matches!(
            build(d),
            Err(XsError::LengthMismatch {
                field: "sigma_t",
                expected: 2,
                got: 3,
                ..
            })
        ));

        let mut d = test_data("fuel");
        d.nu_sigma_f.pop();
        assert!(matches!(
            build(d),
            Err(XsError::LengthMismatch {
                field: "nu_sigma_f",
                expected: 2,
                got: 1,
                ..
            })
        ));

        let mut d = test_data("fuel");
        d.chi.push(0.0);
        assert!(matches!(
            build(d),
            Err(XsError::LengthMismatch {
                field: "chi",
                expected: 2,
                got: 3,
                ..
            })
        ));
    }

    #[test]
    fn wrong_scattering_order_is_rejected() {
        let mut too_few = test_data("fuel");
        too_few.scatter.pop();
        assert!(matches!(
            build(too_few),
            Err(XsError::ScatterOrder {
                expected: 2,
                got: 1,
                ..
            })
        ));

        let mut too_many = test_data("fuel");
        too_many.scatter.push(vec![0.0; 4]);
        assert!(matches!(
            build(too_many),
            Err(XsError::ScatterOrder {
                expected: 2,
                got: 3,
                ..
            })
        ));
    }

    #[test]
    fn wrong_size_scattering_matrix_names_the_order() {
        let mut d = test_data("fuel");
        d.scatter[1] = vec![0.0; 3];
        assert!(matches!(
            build(d),
            Err(XsError::ScatterMatrix {
                order: 1,
                source: GroupMatrixError::LengthMismatch { n: 2, len: 3 },
                ..
            })
        ));
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let mut d = test_data("fuel");
        d.sigma_t[1] = f64::NAN;
        assert!(matches!(
            build(d),
            Err(XsError::NonFinite {
                field: "sigma_t",
                index: 1,
                ..
            })
        ));

        let mut d = test_data("fuel");
        d.scatter[0][2] = f64::INFINITY;
        assert!(matches!(
            build(d),
            Err(XsError::NonFinite {
                field: "scatter",
                index: 2,
                ..
            })
        ));
    }

    #[test]
    fn negative_values_are_rejected() {
        let mut d = test_data("fuel");
        d.nu_sigma_f[0] = -0.1;
        assert!(matches!(
            build(d),
            Err(XsError::Negative {
                field: "nu_sigma_f",
                index: 0,
                ..
            })
        ));

        let mut d = test_data("fuel");
        d.chi = vec![1.5, -0.5];
        assert!(matches!(
            build(d),
            Err(XsError::Negative {
                field: "chi",
                index: 1,
                ..
            })
        ));
    }

    #[test]
    fn fissile_chi_must_sum_to_one() {
        let mut d = test_data("fuel");
        d.chi = vec![0.9, 0.0];
        assert!(matches!(
            build(d),
            Err(XsError::ChiNotNormalized { sum, .. }) if (sum - 0.9).abs() < 1e-15
        ));
    }

    #[test]
    fn non_fissile_material_may_have_zero_chi() {
        let mut d = test_data("water");
        d.nu_sigma_f = vec![0.0, 0.0];
        d.chi = vec![0.0, 0.0];
        let m = build(d).unwrap();
        assert!(!m.is_fissile());
    }

    #[test]
    fn scattering_may_exceed_total() {
        // Legitimate with (n,2n); deliberately not an error
        let mut d = test_data("beryllium");
        d.scatter[0] = vec![0.6, 0.0, 0.0, 0.9]; // 0.6 out of group 0 > sigma_t = 0.5
        assert!(build(d).is_ok());
    }

    #[test]
    fn error_message_names_the_material() {
        let mut d = test_data("fuel");
        d.chi.pop();
        assert_eq!(
            build(d).unwrap_err().to_string(),
            "material 'fuel': field 'chi' has length 1, expected 2"
        );
    }
}
