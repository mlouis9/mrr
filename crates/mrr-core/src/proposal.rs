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
