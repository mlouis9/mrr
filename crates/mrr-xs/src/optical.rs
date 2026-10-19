//! Optical depth along traced paths.
//!
//! Dimension-independent: a geometry traces a path into `Segment`s
//! (material + length), and this module turns those into optical depths
//! using the library's cross sections. The same code serves 1D and 3D.

use crate::library::MaterialLibrary;
use mrr_core::Segment;

/// Optical depth of a path in every group:
///     out[g] = sum over segments of sigma_t,g(material) * length
///
/// Overwrites `out` (length must equal the library's number of groups).
/// Segments are visited once each, so the cost is O(segments x groups) with a
/// single cross-section lookup per segment.
pub fn optical_depths(lib: &MaterialLibrary, segments: &[Segment], out: &mut [f64]) {
    assert_eq!(
        out.len(),
        lib.n_groups(),
        "output buffer must have one entry per group"
    );
    out.fill(0.0);
    for segment in segments {
        let sigma_t = lib.material(segment.material).sigma_t();
        for (tau, &s) in out.iter_mut().zip(sigma_t) {
            *tau += s * segment.length;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::EnergyGroups;
    use crate::material::test_data;
    use mrr_core::MaterialId;
    use mrr_math::testing::assert_all_close;

    /// Two 2-group materials with different sigma_t in every group:
    /// fuel: [0.5, 1.2], water: [0.3, 2.0]
    fn library() -> MaterialLibrary {
        let groups = EnergyGroups::new(vec![2e7, 0.625, 1e-5]).unwrap();
        let mut lib = MaterialLibrary::new(groups, 1);
        lib.add(test_data("fuel")).unwrap();
        let mut water = test_data("water");
        water.sigma_t = vec![0.3, 2.0];
        lib.add(water).unwrap();
        lib
    }

    fn seg(material: usize, length: f64) -> Segment {
        Segment {
            material: MaterialId(material),
            length,
        }
    }

    #[test]
    fn single_segment_is_sigma_t_times_length() {
        let mut out = [0.0; 2];
        optical_depths(&library(), &[seg(0, 2.0)], &mut out);
        assert_all_close(&out, &[1.0, 2.4]);
    }

    #[test]
    fn segments_add_per_group() {
        // fuel 2.0 then water 0.5: [0.5*2 + 0.3*0.5, 1.2*2 + 2.0*0.5]
        let mut out = [0.0; 2];
        optical_depths(&library(), &[seg(0, 2.0), seg(1, 0.5)], &mut out);
        assert_all_close(&out, &[1.15, 3.4]);
    }

    #[test]
    fn order_of_segments_does_not_matter() {
        let lib = library();
        let (mut a, mut b) = ([0.0; 2], [0.0; 2]);
        optical_depths(&lib, &[seg(0, 0.7), seg(1, 1.3), seg(0, 0.2)], &mut a);
        optical_depths(&lib, &[seg(1, 1.3), seg(0, 0.9)], &mut b);
        assert_all_close(&a, &b);
    }

    #[test]
    fn empty_path_has_zero_optical_depth() {
        let mut out = [7.0; 2];
        optical_depths(&library(), &[], &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn output_is_overwritten_not_accumulated() {
        let lib = library();
        let mut out = [0.0; 2];
        optical_depths(&lib, &[seg(1, 1.0)], &mut out);
        optical_depths(&lib, &[seg(1, 1.0)], &mut out);
        assert_all_close(&out, &[0.3, 2.0]);
    }

    #[test]
    #[should_panic(expected = "one entry per group")]
    fn wrong_buffer_length_panics() {
        let mut out = [0.0; 3];
        optical_depths(&library(), &[seg(0, 1.0)], &mut out);
    }
}
