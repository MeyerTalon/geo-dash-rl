use rand::RngExt;
use rand::rngs::StdRng;

const FIRST_SPIKE_X_UNITS: f32 = 12.0;
const MIN_GAP_UNITS: f32 = 6.0;
const MAX_GAP_UNITS: f32 = 12.0;
const APEX_REL_X: f32 = 0.5;

pub const SPIKE_WIDTH_UNITS: f32 = 1.0;
pub const SPIKE_HEIGHT_UNITS: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spike {
    pub x_units: f32,
}

impl Spike {
    #[must_use]
    pub fn covers(self, x_units: f32, y_units: f32) -> bool {
        let rel_x = (x_units - self.x_units) / SPIKE_WIDTH_UNITS;
        if !(0.0..=1.0).contains(&rel_x) {
            return false;
        }
        let height_at_x = SPIKE_HEIGHT_UNITS * (1.0 - (rel_x - APEX_REL_X).abs() / APEX_REL_X);
        (0.0..=height_at_x).contains(&y_units)
    }

    #[must_use]
    pub fn overlaps_box(self, left_units: f32, bottom_units: f32, size_units: f32) -> bool {
        left_units + size_units > self.x_units
            && left_units < self.x_units + SPIKE_WIDTH_UNITS
            && bottom_units < SPIKE_HEIGHT_UNITS
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Level {
    length_units: f32,
    spikes: Vec<Spike>,
}

impl Level {
    #[must_use]
    pub fn generate(length_units: f32, rng: &mut StdRng) -> Self {
        let mut spikes = Vec::new();
        let mut x_units = FIRST_SPIKE_X_UNITS;
        while x_units + SPIKE_WIDTH_UNITS < length_units {
            spikes.push(Spike { x_units });
            x_units += SPIKE_WIDTH_UNITS + rng.random_range(MIN_GAP_UNITS..MAX_GAP_UNITS);
        }
        Self {
            length_units,
            spikes,
        }
    }

    #[must_use]
    pub fn length_units(&self) -> f32 {
        self.length_units
    }

    #[must_use]
    pub fn spikes(&self) -> &[Spike] {
        &self.spikes
    }

    #[must_use]
    pub fn is_deadly_box(&self, left_units: f32, bottom_units: f32, size_units: f32) -> bool {
        self.spikes
            .iter()
            .any(|spike| spike.overlaps_box(left_units, bottom_units, size_units))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const LEVEL_LENGTH_UNITS: f32 = 200.0;

    #[test]
    fn same_seed_generates_the_same_level() {
        let first = Level::generate(LEVEL_LENGTH_UNITS, &mut StdRng::seed_from_u64(7));
        let second = Level::generate(LEVEL_LENGTH_UNITS, &mut StdRng::seed_from_u64(7));
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_generate_different_levels() {
        let first = Level::generate(LEVEL_LENGTH_UNITS, &mut StdRng::seed_from_u64(1));
        let second = Level::generate(LEVEL_LENGTH_UNITS, &mut StdRng::seed_from_u64(2));
        assert_ne!(first, second);
    }

    #[test]
    fn spikes_keep_the_minimum_gap_and_stay_inside_the_level() {
        let level = Level::generate(LEVEL_LENGTH_UNITS, &mut StdRng::seed_from_u64(3));
        assert!(!level.spikes().is_empty());
        for pair in level.spikes().windows(2) {
            let gap_units = pair[1].x_units - pair[0].x_units - SPIKE_WIDTH_UNITS;
            assert!(gap_units >= MIN_GAP_UNITS);
        }
        let last = level.spikes().last().unwrap();
        assert!(last.x_units + SPIKE_WIDTH_UNITS < LEVEL_LENGTH_UNITS);
    }

    #[test]
    fn spike_covers_only_its_triangle() {
        let spike = Spike { x_units: 10.0 };
        assert!(spike.covers(10.5, 0.9));
        assert!(!spike.covers(10.05, 0.9));
        assert!(!spike.covers(9.9, 0.1));
        assert!(!spike.covers(10.5, 1.1));
    }

    #[test]
    fn box_above_the_spike_is_safe() {
        let spike = Spike { x_units: 10.0 };
        assert!(spike.overlaps_box(9.5, 0.0, 1.0));
        assert!(!spike.overlaps_box(9.5, SPIKE_HEIGHT_UNITS, 1.0));
        assert!(!spike.overlaps_box(8.0, 0.0, 1.0));
    }
}
