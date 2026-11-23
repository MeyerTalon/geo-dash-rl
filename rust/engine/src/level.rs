use rand::RngExt;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

const FIRST_SPIKE_X_UNITS: f32 = 12.0;
const MIN_GAP_UNITS: f32 = 6.0;
const MAX_GAP_UNITS: f32 = 12.0;
const APEX_REL_X: f32 = 0.5;

pub const SPIKE_WIDTH_UNITS: f32 = 1.0;
pub const SPIKE_HEIGHT_UNITS: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x_units: f32,
    pub y_units: f32,
    pub width_units: f32,
    pub height_units: f32,
}

impl Rect {
    #[must_use]
    pub fn right_units(self) -> f32 {
        self.x_units + self.width_units
    }

    #[must_use]
    pub fn top_units(self) -> f32 {
        self.y_units + self.height_units
    }

    #[must_use]
    pub fn contains(self, x_units: f32, y_units: f32) -> bool {
        (self.x_units..=self.right_units()).contains(&x_units)
            && (self.y_units..=self.top_units()).contains(&y_units)
    }

    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.x_units < other.right_units()
            && other.x_units < self.right_units()
            && self.y_units < other.top_units()
            && other.y_units < self.top_units()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spike {
    pub bounds: Rect,
    pub points_down: bool,
}

impl Spike {
    #[must_use]
    pub fn on_ground(x_units: f32) -> Self {
        Self {
            bounds: Rect {
                x_units,
                y_units: 0.0,
                width_units: SPIKE_WIDTH_UNITS,
                height_units: SPIKE_HEIGHT_UNITS,
            },
            points_down: false,
        }
    }

    #[must_use]
    pub fn covers(self, x_units: f32, y_units: f32) -> bool {
        let rel_x = (x_units - self.bounds.x_units) / self.bounds.width_units;
        if !(0.0..=1.0).contains(&rel_x) {
            return false;
        }
        let height_at_x =
            self.bounds.height_units * (1.0 - (rel_x - APEX_REL_X).abs() / APEX_REL_X);
        let height_above_base_units = if self.points_down {
            self.bounds.top_units() - y_units
        } else {
            y_units - self.bounds.y_units
        };
        (0.0..=height_at_x).contains(&height_above_base_units)
    }

    #[must_use]
    pub fn overlaps_box(self, left_units: f32, bottom_units: f32, size_units: f32) -> bool {
        self.bounds.overlaps(Rect {
            x_units: left_units,
            y_units: bottom_units,
            width_units: size_units,
            height_units: size_units,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Level {
    length_units: f32,
    spikes: Vec<Spike>,
    solids: Vec<Rect>,
}

impl Level {
    #[must_use]
    pub fn new(length_units: f32, spikes: Vec<Spike>, solids: Vec<Rect>) -> Self {
        Self {
            length_units,
            spikes,
            solids,
        }
    }

    #[must_use]
    pub fn generate(length_units: f32, rng: &mut StdRng) -> Self {
        let mut spikes = Vec::new();
        let mut x_units = FIRST_SPIKE_X_UNITS;
        while x_units + SPIKE_WIDTH_UNITS < length_units {
            spikes.push(Spike::on_ground(x_units));
            x_units += SPIKE_WIDTH_UNITS + rng.random_range(MIN_GAP_UNITS..MAX_GAP_UNITS);
        }
        Self::new(length_units, spikes, Vec::new())
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
    pub fn solids(&self) -> &[Rect] {
        &self.solids
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
            let gap_units = pair[1].bounds.x_units - pair[0].bounds.x_units - SPIKE_WIDTH_UNITS;
            assert!(gap_units >= MIN_GAP_UNITS);
        }
        let last = level.spikes().last().unwrap();
        assert!(last.bounds.x_units + SPIKE_WIDTH_UNITS < LEVEL_LENGTH_UNITS);
    }

    #[test]
    fn spike_covers_only_its_triangle() {
        let spike = Spike::on_ground(10.0);
        assert!(spike.covers(10.5, 0.9));
        assert!(!spike.covers(10.05, 0.9));
        assert!(!spike.covers(9.9, 0.1));
        assert!(!spike.covers(10.5, 1.1));
    }

    #[test]
    fn box_above_the_spike_is_safe() {
        let spike = Spike::on_ground(10.0);
        assert!(spike.overlaps_box(9.5, 0.0, 1.0));
        assert!(!spike.overlaps_box(9.5, SPIKE_HEIGHT_UNITS, 1.0));
        assert!(!spike.overlaps_box(8.0, 0.0, 1.0));
    }

    #[test]
    fn hanging_spike_covers_its_triangle_from_the_top_edge() {
        let spike = Spike {
            bounds: Rect {
                x_units: 10.0,
                y_units: 4.0,
                width_units: 1.0,
                height_units: 1.0,
            },
            points_down: true,
        };
        assert!(spike.covers(10.5, 4.1));
        assert!(spike.covers(10.05, 4.95));
        assert!(!spike.covers(10.05, 4.1));
        assert!(!spike.covers(10.5, 5.1));
        assert!(!spike.overlaps_box(10.0, 0.0, 1.0));
        assert!(spike.overlaps_box(10.0, 3.5, 1.0));
    }

    #[test]
    fn rects_overlap_only_when_they_share_area() {
        let rect = Rect {
            x_units: 0.0,
            y_units: 0.0,
            width_units: 2.0,
            height_units: 1.0,
        };
        let touching = Rect {
            x_units: 2.0,
            ..rect
        };
        let inside = Rect {
            x_units: 1.0,
            ..rect
        };
        assert!(!rect.overlaps(touching));
        assert!(rect.overlaps(inside));
        assert!(rect.contains(1.0, 0.5));
        assert!(!rect.contains(1.0, 1.5));
    }
}
