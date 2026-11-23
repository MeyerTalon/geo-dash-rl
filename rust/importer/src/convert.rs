use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use geo_dash_engine::level::{Level, Rect, Spike};

use crate::parse::RawObject;
use crate::table::{ObjectKind, SizePx, object_kind};

const PX_PER_UNIT: f32 = 30.0;
const HALF: f32 = 0.5;
const FULL_TURN_DEG: f32 = 360.0;
const ROTATION_TOLERANCE_DEG: f32 = 0.5;
const QUARTER_TURNS_DEG: [(f32, QuarterTurns); 4] = [
    (0.0, QuarterTurns::None),
    (90.0, QuarterTurns::One),
    (180.0, QuarterTurns::Two),
    (270.0, QuarterTurns::Three),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuarterTurns {
    None,
    One,
    Two,
    Three,
}

impl QuarterTurns {
    fn from_degrees(rotation_deg: f32) -> Option<Self> {
        let wrapped_deg = rotation_deg.rem_euclid(FULL_TURN_DEG);
        QUARTER_TURNS_DEG
            .iter()
            .find(|(turn_deg, _)| (wrapped_deg - turn_deg).abs() < ROTATION_TOLERANCE_DEG)
            .map(|&(_, turns)| turns)
    }

    fn is_sideways(self) -> bool {
        matches!(self, Self::One | Self::Three)
    }
}

#[derive(Debug, PartialEq)]
pub struct ImportReport {
    pub length_units: f32,
    pub solids: usize,
    pub spikes: usize,
    pub decoration: usize,
    pub not_simulated: BTreeMap<u32, usize>,
}

impl fmt::Display for ImportReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "length {:.1} units: {} solids, {} spikes, {} decoration objects dropped, \
             objects not simulated yet (id: count) {:?}",
            self.length_units, self.solids, self.spikes, self.decoration, self.not_simulated
        )
    }
}

pub fn import_level(objects: &[RawObject]) -> Result<(Level, ImportReport), String> {
    let unknown_ids: BTreeSet<u32> = objects
        .iter()
        .map(|object| object.id)
        .filter(|&id| object_kind(id).is_none())
        .collect();
    if !unknown_ids.is_empty() {
        return Err(format!(
            "object ids missing from the table: {unknown_ids:?}"
        ));
    }
    let length_units = objects.iter().map(|object| object.x_px).fold(0.0, f32::max) / PX_PER_UNIT;
    if length_units <= 0.0 {
        return Err("level has no object right of the start".to_owned());
    }
    let mut spikes = Vec::new();
    let mut solids = Vec::new();
    let mut report = ImportReport {
        length_units,
        solids: 0,
        spikes: 0,
        decoration: 0,
        not_simulated: BTreeMap::new(),
    };
    for object in objects {
        match object_kind(object.id) {
            Some(ObjectKind::Solid(size)) => solids.push(solid_bounds(object, size)?),
            Some(ObjectKind::Spike(size)) => spikes.push(spike(object, size)?),
            Some(ObjectKind::Decoration) => report.decoration += 1,
            Some(ObjectKind::NotSimulated) => {
                *report.not_simulated.entry(object.id).or_default() += 1;
            }
            None => return Err(format!("object id {} missing from the table", object.id)),
        }
    }
    report.solids = solids.len();
    report.spikes = spikes.len();
    Ok((Level::new(length_units, spikes, solids), report))
}

fn turns_of(object: &RawObject) -> Result<QuarterTurns, String> {
    QuarterTurns::from_degrees(object.rotation_deg).ok_or_else(|| {
        format!(
            "object {} at x {} px is rotated {} degrees, not a quarter turn",
            object.id, object.x_px, object.rotation_deg
        )
    })
}

fn bounds_around_center(object: &RawObject, size: SizePx, turns: QuarterTurns) -> Rect {
    let (width_px, height_px) = if turns.is_sideways() {
        (size.height_px, size.width_px)
    } else {
        (size.width_px, size.height_px)
    };
    Rect {
        x_units: (object.x_px - HALF * width_px) / PX_PER_UNIT,
        y_units: (object.y_px - HALF * height_px) / PX_PER_UNIT,
        width_units: width_px / PX_PER_UNIT,
        height_units: height_px / PX_PER_UNIT,
    }
}

fn solid_bounds(object: &RawObject, size: SizePx) -> Result<Rect, String> {
    Ok(bounds_around_center(object, size, turns_of(object)?))
}

fn spike(object: &RawObject, size: SizePx) -> Result<Spike, String> {
    let turns = turns_of(object)?;
    if turns.is_sideways() {
        return Err(format!(
            "spike {} at x {} px points sideways, which the engine does not support",
            object.id, object.x_px
        ));
    }
    Ok(Spike {
        bounds: bounds_around_center(object, size, turns),
        points_down: object.flip_y != (turns == QuarterTurns::Two),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-4;
    const BLOCK_ID: u32 = 1;
    const SPIKE_ID: u32 = 8;
    const HALF_SPIKE_ID: u32 = 39;
    const SLAB_ID: u32 = 40;
    const PORTAL_ID: u32 = 13;
    const DECORATION_ID: u32 = 15;
    const UNLISTED_ID: u32 = 9_999;

    fn object(id: u32, x_px: f32, y_px: f32) -> RawObject {
        RawObject {
            id,
            x_px,
            y_px,
            flip_y: false,
            rotation_deg: 0.0,
        }
    }

    fn assert_near(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "{actual} vs {expected}"
        );
    }

    #[test]
    fn spike_center_in_pixels_becomes_left_and_bottom_in_units() {
        let (level, _) = import_level(&[object(SPIKE_ID, 525.0, 15.0)]).unwrap();
        let bounds = level.spikes().first().unwrap().bounds;
        assert_near(bounds.x_units, 17.0);
        assert_near(bounds.y_units, 0.0);
        assert_near(bounds.width_units, 1.0);
        assert_near(bounds.height_units, 1.0);
    }

    #[test]
    fn half_spike_and_slab_keep_their_sprite_heights() {
        let (level, _) = import_level(&[
            object(HALF_SPIKE_ID, 60.0, 6.0),
            object(SLAB_ID, 90.0, 23.0),
        ])
        .unwrap();
        assert_near(
            level.spikes().first().unwrap().bounds.height_units,
            14.0 / 30.0,
        );
        let slab = level.solids().first().unwrap();
        assert_near(slab.y_units, 16.0 / 30.0);
        assert_near(slab.height_units, 14.0 / 30.0);
    }

    #[test]
    fn length_is_the_furthest_object_including_dropped_ones() {
        let (level, _) = import_level(&[
            object(BLOCK_ID, 300.0, 15.0),
            object(DECORATION_ID, 900.0, 15.0),
        ])
        .unwrap();
        assert_near(level.length_units(), 30.0);
    }

    #[test]
    fn flip_or_half_turn_points_a_spike_down_and_both_cancel() {
        let flipped = RawObject {
            flip_y: true,
            ..object(SPIKE_ID, 60.0, 15.0)
        };
        let turned = RawObject {
            rotation_deg: -180.0,
            ..object(SPIKE_ID, 60.0, 15.0)
        };
        let both = RawObject {
            flip_y: true,
            rotation_deg: 180.0,
            ..object(SPIKE_ID, 60.0, 15.0)
        };
        let (level, _) = import_level(&[flipped, turned, both]).unwrap();
        let directions: Vec<bool> = level
            .spikes()
            .iter()
            .map(|spike| spike.points_down)
            .collect();
        assert_eq!(directions, vec![true, true, false]);
    }

    #[test]
    fn quarter_turn_swaps_a_solid_s_width_and_height() {
        let turned = RawObject {
            rotation_deg: 90.0,
            ..object(SLAB_ID, 90.0, 15.0)
        };
        let (level, _) = import_level(&[turned]).unwrap();
        let slab = level.solids().first().unwrap();
        assert_near(slab.width_units, 14.0 / 30.0);
        assert_near(slab.height_units, 1.0);
    }

    #[test]
    fn sideways_spike_and_odd_angles_are_rejected() {
        let sideways = RawObject {
            rotation_deg: 90.0,
            ..object(SPIKE_ID, 60.0, 15.0)
        };
        let slanted = RawObject {
            rotation_deg: 45.0,
            ..object(BLOCK_ID, 60.0, 15.0)
        };
        assert!(import_level(&[sideways]).unwrap_err().contains("sideways"));
        assert!(
            import_level(&[slanted])
                .unwrap_err()
                .contains("quarter turn")
        );
    }

    #[test]
    fn unlisted_ids_fail_the_import_and_are_all_named() {
        let error = import_level(&[
            object(UNLISTED_ID, 60.0, 15.0),
            object(BLOCK_ID, 90.0, 15.0),
        ])
        .unwrap_err();
        assert!(error.contains(&UNLISTED_ID.to_string()));
    }

    #[test]
    fn portals_and_decoration_are_counted_but_not_added_to_the_level() {
        let (level, report) = import_level(&[
            object(PORTAL_ID, 60.0, 15.0),
            object(PORTAL_ID, 90.0, 15.0),
            object(DECORATION_ID, 120.0, 15.0),
        ])
        .unwrap();
        assert!(level.spikes().is_empty());
        assert!(level.solids().is_empty());
        assert_eq!(report.decoration, 1);
        assert_eq!(report.not_simulated.get(&PORTAL_ID), Some(&2));
    }

    #[test]
    fn level_without_objects_is_rejected() {
        assert!(import_level(&[]).is_err());
    }
}
