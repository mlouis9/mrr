//! A collection of materials sharing one energy structure and scattering order

use crate::energy::EnergyGroups;
use crate::error::XsError;
use crate::material::{MgMaterial, MgMaterialData};
use mrr_core::{MaterialId, MomentLayout};

/// All materials of a problem. The energy groups and scattering order are set
/// once at construction; every material is validated against them when added,
/// so all materials in a library agree on G and L by construction.
#[derive(Debug, Clone)]
pub struct MaterialLibrary {
    groups: EnergyGroups,
    max_order: usize,
    materials: Vec<MgMaterial>,
}

impl MaterialLibrary {
    pub fn new(groups: EnergyGroups, max_order: usize) -> Self {
        Self {
            groups,
            max_order,
            materials: Vec::new(),
        }
    }

    /// Validate `data` against this library's groups and order, store it, and
    /// return its ID. On error the library is unchanged.
    pub fn add(&mut self, data: MgMaterialData) -> Result<MaterialId, XsError> {
        if self.find(&data.name).is_some() {
            return Err(XsError::DuplicateName {
                material: data.name,
            });
        }
        let material = MgMaterial::from_data(data, self.n_groups(), self.max_order)?;
        self.materials.push(material);
        Ok(MaterialId(self.materials.len() - 1))
    }

    /// The material with this ID.
    ///
    /// Panics on an unknown ID: IDs come from `add`, and geometry IDs are
    /// checked once up front with `check_ids`.
    pub fn material(&self, id: MaterialId) -> &MgMaterial {
        match self.materials.get(id.0) {
            Some(m) => m,
            None => panic!("unknown material {id:?} (library has {})", self.len()),
        }
    }

    /// ID of the material with this name, if any
    pub fn find(&self, name: &str) -> Option<MaterialId> {
        self.materials
            .iter()
            .position(|m| m.name() == name)
            .map(MaterialId)
    }

    /// Check that every ID refers to a material in this library
    pub fn check_ids(&self, ids: impl IntoIterator<Item = MaterialId>) -> Result<(), XsError> {
        for id in ids {
            if id.0 >= self.len() {
                return Err(XsError::UnknownMaterial(id));
            }
        }
        Ok(())
    }

    pub fn groups(&self) -> &EnergyGroups {
        &self.groups
    }

    pub fn n_groups(&self) -> usize {
        self.groups.n_groups()
    }

    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// Number of materials
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Shape of each atom's moment block: (max_order + 1) moments x G groups.
    /// The single source of truth for the layout used by the run.
    pub fn layout(&self) -> MomentLayout {
        MomentLayout::from_legendre_order(self.max_order, self.n_groups())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::test_data;

    fn library() -> MaterialLibrary {
        let groups = EnergyGroups::new(vec![2e7, 0.625, 1e-5]).unwrap();
        MaterialLibrary::new(groups, 1)
    }

    #[test]
    fn ids_are_sequential() {
        let mut lib = library();
        assert!(lib.is_empty());
        assert_eq!(lib.add(test_data("fuel")).unwrap(), MaterialId(0));
        assert_eq!(lib.add(test_data("water")).unwrap(), MaterialId(1));
        assert_eq!(lib.len(), 2);
        assert_eq!(lib.material(MaterialId(1)).name(), "water");
    }

    #[test]
    fn find_by_name() {
        let mut lib = library();
        lib.add(test_data("fuel")).unwrap();
        lib.add(test_data("water")).unwrap();
        assert_eq!(lib.find("water"), Some(MaterialId(1)));
        assert_eq!(lib.find("steel"), None);
    }

    #[test]
    fn invalid_material_is_rejected_and_not_added() {
        let mut lib = library();
        let mut d = test_data("fuel");
        d.sigma_t.push(1.0); // 3 groups in a 2-group library
        assert!(matches!(lib.add(d), Err(XsError::LengthMismatch { .. })));
        assert!(lib.is_empty());
    }

    #[test]
    fn material_must_match_library_scattering_order() {
        let groups = EnergyGroups::new(vec![2e7, 0.625, 1e-5]).unwrap();
        let mut lib = MaterialLibrary::new(groups, 2); // L = 2, test data has L = 1
        assert!(matches!(
            lib.add(test_data("fuel")),
            Err(XsError::ScatterOrder {
                expected: 3,
                got: 2,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut lib = library();
        lib.add(test_data("fuel")).unwrap();
        assert!(matches!(
            lib.add(test_data("fuel")),
            Err(XsError::DuplicateName { .. })
        ));
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn check_ids_rejects_unknown_ids() {
        let mut lib = library();
        lib.add(test_data("fuel")).unwrap();
        assert!(lib.check_ids([MaterialId(0), MaterialId(0)]).is_ok());
        assert!(matches!(
            lib.check_ids([MaterialId(0), MaterialId(1)]),
            Err(XsError::UnknownMaterial(MaterialId(1)))
        ));
    }

    #[test]
    #[should_panic(expected = "unknown material")]
    fn material_panics_on_unknown_id() {
        library().material(MaterialId(0));
    }

    #[test]
    fn layout_matches_groups_and_order() {
        let lib = library();
        assert_eq!(lib.n_groups(), 2);
        assert_eq!(lib.max_order(), 1);
        assert_eq!(lib.layout().block_len(), (1 + 1) * 2);
    }
}
