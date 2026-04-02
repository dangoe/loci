/// An embedding vector represented as a sequence of `f32` values.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Creates an `Embedding` from the given values.
    pub fn new(values: Vec<f32>) -> Self {
        Self(values)
    }

    /// Returns the raw embedding values.
    pub fn values(&self) -> &[f32] {
        &self.0
    }

    /// Returns the number of dimensions in this embedding.
    pub fn dimension(&self) -> usize {
        self.0.len()
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(values: Vec<f32>) -> Self {
        Self::new(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension() {
        assert_eq!(Embedding::new(vec![1.0, 2.0, 3.0]).dimension(), 3);
    }

    #[test]
    fn test_values() {
        let values = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(Embedding::new(values.clone()).values(), values.as_slice());
    }

    #[test]
    fn test_from_vec() {
        let e: Embedding = vec![1.0_f32, 2.0].into();
        assert_eq!(e.dimension(), 2);
    }
}
