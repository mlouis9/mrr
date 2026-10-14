//! Dense square matrices for group-to-group transfer (e.g. scattering)

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum GroupMatrixError {
    /// `data.len()` must equal `n * n`
    LengthMismatch { n: usize, len: usize },
}

impl fmt::Display for GroupMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GroupMatrixError::LengthMismatch { n, len } => write!(
                f,
                "group matrix of size {n}x{n} needs {} entries, got {len}",
                n * n
            ),
        }
    }
}

impl std::error::Error for GroupMatrixError {}

/// An n x n transfer matrix between energy groups.
///
/// **Convention:** entry `(from, to)` is the transfer *from* group `from`
/// *into* group `to` (e.g. Σ_s,ℓ(g' → g) is `get(g', g)`). Storage is flat and
/// row-major by `from`: `data[from * n + to]`. This matches the usual layout of
/// scattering matrices in nuclear data, where each row is one incoming group.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupMatrix {
    n: usize,
    data: Vec<f64>,
}

impl GroupMatrix {
    /// Build from row-major data (`data[from * n + to]`)
    pub fn new(n: usize, data: Vec<f64>) -> Result<Self, GroupMatrixError> {
        if data.len() != n * n {
            return Err(GroupMatrixError::LengthMismatch { n, len: data.len() });
        }
        Ok(Self { n, data })
    }

    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n],
        }
    }

    /// Number of groups
    pub fn n(&self) -> usize {
        self.n
    }

    /// Transfer from group `from` into group `to`
    pub fn get(&self, from: usize, to: usize) -> f64 {
        self.data[from * self.n + to]
    }

    pub fn set(&mut self, from: usize, to: usize, value: f64) {
        self.data[from * self.n + to] = value;
    }

    /// Row-major storage, `data[from * n + to]`
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Total transfer out of group `from` (sum of its row), e.g. Σ_s0 out of g'
    pub fn row_sum(&self, from: usize) -> f64 {
        self.data[from * self.n..(from + 1) * self.n].iter().sum()
    }

    /// Apply the transfer to a group vector:
    ///     out[to] = Σ_from M(from, to) * x[from]
    /// e.g. the scattering source in each group from the flux in all groups.
    /// Overwrites `out`; allocation-free.
    pub fn transfer(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(x.len(), self.n, "input vector length");
        assert_eq!(out.len(), self.n, "output vector length");
        out.fill(0.0);
        for (row, &x_from) in self.data.chunks_exact(self.n).zip(x) {
            for (o, &m) in out.iter_mut().zip(row) {
                *o += m * x_from;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Non-symmetric on purpose, so a transposed index shows up immediately:
    // from 0: [1, 2]   (0->0 = 1, 0->1 = 2)
    // from 1: [3, 4]   (1->0 = 3, 1->1 = 4)
    fn m() -> GroupMatrix {
        GroupMatrix::new(2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()
    }

    #[test]
    fn rejects_wrong_length() {
        assert_eq!(
            GroupMatrix::new(2, vec![1.0; 3]),
            Err(GroupMatrixError::LengthMismatch { n: 2, len: 3 })
        );
    }

    #[test]
    fn get_uses_from_to_convention() {
        let m = m();
        assert_eq!(m.get(0, 1), 2.0); // from 0 into 1
        assert_eq!(m.get(1, 0), 3.0); // from 1 into 0
        assert_eq!(m.row_sum(1), 7.0);
    }

    #[test]
    fn transfer_sums_over_incoming_groups() {
        // out[0] = 1*x0 + 3*x1, out[1] = 2*x0 + 4*x1
        let mut out = [99.0, 99.0];
        m().transfer(&[10.0, 1.0], &mut out);
        assert_eq!(out, [13.0, 24.0]);
    }

    #[test]
    fn set_and_zeros() {
        let mut z = GroupMatrix::zeros(3);
        z.set(2, 0, 5.0);
        assert_eq!(z.get(2, 0), 5.0);
        assert_eq!(z.get(0, 2), 0.0);
        assert_eq!(z.n(), 3);
    }

    #[test]
    fn readable_error() {
        assert_eq!(
            GroupMatrixError::LengthMismatch { n: 2, len: 3 }.to_string(),
            "group matrix of size 2x2 needs 4 entries, got 3"
        );
    }
}
