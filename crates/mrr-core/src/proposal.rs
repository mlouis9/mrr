//! Proposals used for importance sampling target atom distributions
use crate::sampler::{CHUNK_SIZE, SamplerStreams, UnitSampler};
use crate::traits::{Medium, Proposal, Source, Targets};

pub struct UniformProposal;

impl<M: Medium> Proposal<M> for UniformProposal {
    fn sample_target_atoms(
        &self,
        medium: &M,
        source: &Source<M::Point>,
        streams: &impl SamplerStreams,
    ) -> Targets<M::Point> {
        let n_atoms = source.n_atoms();
        assert!(n_atoms > 0); // Avoid division by zero
        let weight = medium.volume() / n_atoms as f64;
        let n_chunks = n_atoms.div_ceil(CHUNK_SIZE);

        // Sample points for each chunk using that chunk's rng
        (0..n_chunks).fold(
            Targets {
                points: Vec::with_capacity(n_atoms),
                weights: Vec::with_capacity(n_atoms),
            },
            |mut targets, chunk| {
                let n_here = CHUNK_SIZE.min(n_atoms - chunk * CHUNK_SIZE);
                let mut sampler = streams.stream(chunk);
                let mut u = vec![0.0; M::SAMPLE_DIM];

                let mut count = 0;
                while count < n_here {
                    sampler.next_point(&mut u);
                    if let Some(point) = medium.map_uniform(&u) {
                        targets.points.push(point);
                        targets.weights.push(weight);
                        count += 1;
                    }
                }
                targets
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{GridStreams1D, RandomStreams, purpose};
    use crate::test_utils::*;

    fn random_streams(cycle: usize) -> RandomStreams {
        RandomStreams {
            base_seed: 3,
            purpose: purpose::PROPOSAL,
            cycle,
        }
    }

    #[test]
    fn grid_targets_have_midpoints_and_equal_weights() {
        let n = CHUNK_SIZE + 3;
        let medium = TestSlab::new(2.0, 1, 0.0..1.0);
        let Targets { points, weights } = UniformProposal.sample_target_atoms(
            &medium,
            &Source::Initialize(n),
            &GridStreams1D::new(n),
        );
        assert_eq!(points.len(), n);
        for (i, &x) in points.iter().enumerate() {
            assert_eq!(x, (i as f64 + 0.5) / n as f64 * 2.0);
        }
        assert!(weights.iter().all(|&w| w == medium.volume() / n as f64));
        assert_close(weights.iter().sum::<f64>(), medium.volume());
    }

    #[test]
    fn rejection_keeps_exactly_n_points_inside_the_domain() {
        // Accept only u in [0.25, 0.75): domain is [0.5, 1.5) for width 2.
        // Grid streams would loop forever here, so use random streams.
        let n = CHUNK_SIZE + 3;
        let medium = TestSlab::new(2.0, 1, 0.25..0.75);
        let Targets { points, weights } = UniformProposal.sample_target_atoms(
            &medium,
            &Source::Initialize(n),
            &random_streams(0),
        );
        assert_eq!(points.len(), n);
        assert!(points.iter().all(|&x| (0.5..1.5).contains(&x)));
        assert_close(weights.iter().sum::<f64>(), medium.volume());
    }

    #[test]
    fn same_streams_give_same_targets_and_new_cycle_gives_new_ones() {
        let medium = TestSlab::new(2.0, 1, 0.0..1.0);
        let source = Source::Initialize(100);
        let a = UniformProposal.sample_target_atoms(&medium, &source, &random_streams(0));
        let b = UniformProposal.sample_target_atoms(&medium, &source, &random_streams(0));
        let c = UniformProposal.sample_target_atoms(&medium, &source, &random_streams(1));
        assert_eq!(a.points, b.points);
        assert_ne!(a.points, c.points);
    }
}
