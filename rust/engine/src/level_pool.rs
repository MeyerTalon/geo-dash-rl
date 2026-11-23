use std::sync::Arc;

use rand::RngExt;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

use crate::level::Level;

#[derive(Clone, Debug)]
pub struct LevelPool {
    imported: Vec<Arc<Level>>,
    imported_fraction: f32,
}

impl LevelPool {
    #[must_use]
    pub fn new(imported: Vec<Level>, imported_fraction: f32) -> Self {
        Self {
            imported: imported.into_iter().map(Arc::new).collect(),
            imported_fraction,
        }
    }

    #[must_use]
    pub fn pick(&self, generated_length_units: f32, rng: &mut StdRng) -> Arc<Level> {
        let imported = self.imported.choose(rng);
        match imported {
            Some(level) if rng.random_range(0.0..1.0) < self.imported_fraction => Arc::clone(level),
            _ => Arc::new(Level::generate(generated_length_units, rng)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const GENERATED_LENGTH_UNITS: f32 = 200.0;
    const IMPORTED_LENGTH_UNITS: f32 = 50.0;
    const PICKS: usize = 50;

    fn imported_level() -> Level {
        Level::new(IMPORTED_LENGTH_UNITS, Vec::new(), Vec::new())
    }

    fn all_equal_to(lengths: &[f32], expected: f32) -> bool {
        lengths
            .iter()
            .all(|length| (length - expected).abs() < f32::EPSILON)
    }

    fn picked_lengths(pool: &LevelPool) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(0);
        (0..PICKS)
            .map(|_| pool.pick(GENERATED_LENGTH_UNITS, &mut rng).length_units())
            .collect()
    }

    #[test]
    fn empty_pool_only_generates() {
        let lengths = picked_lengths(&LevelPool::new(Vec::new(), 1.0));
        assert!(all_equal_to(&lengths, GENERATED_LENGTH_UNITS));
    }

    #[test]
    fn fraction_one_only_picks_imported_levels() {
        let lengths = picked_lengths(&LevelPool::new(vec![imported_level()], 1.0));
        assert!(all_equal_to(&lengths, IMPORTED_LENGTH_UNITS));
    }

    #[test]
    fn fraction_zero_only_generates() {
        let lengths = picked_lengths(&LevelPool::new(vec![imported_level()], 0.0));
        assert!(all_equal_to(&lengths, GENERATED_LENGTH_UNITS));
    }

    #[test]
    fn fraction_between_mixes_both_sources() {
        let lengths = picked_lengths(&LevelPool::new(vec![imported_level()], 0.5));
        assert!(lengths.contains(&GENERATED_LENGTH_UNITS));
        assert!(lengths.contains(&IMPORTED_LENGTH_UNITS));
    }
}
