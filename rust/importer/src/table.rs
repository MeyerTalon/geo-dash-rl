use std::ops::RangeInclusive;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizePx {
    pub width_px: f32,
    pub height_px: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObjectKind {
    Solid(SizePx),
    Spike(SizePx),
    Decoration,
    NotSimulated,
}

const BLOCK_SIZE_PX: SizePx = SizePx {
    width_px: 30.0,
    height_px: 30.0,
};
const SLAB_SIZE_PX: SizePx = SizePx {
    width_px: 30.0,
    height_px: 14.0,
};
const SPIKE_SIZE_PX: SizePx = SizePx {
    width_px: 30.0,
    height_px: 30.0,
};
const PIT_SPIKE_SIZE_PX: SizePx = SizePx {
    width_px: 30.0,
    height_px: 27.0,
};
const HALF_SPIKE_SIZE_PX: SizePx = SizePx {
    width_px: 30.0,
    height_px: 14.0,
};
const SMALL_SPIKE_SIZE_PX: SizePx = SizePx {
    width_px: 20.0,
    height_px: 19.0,
};

const BLOCK_IDS: RangeInclusive<u32> = 1..=7;
const SLAB_ID: u32 = 40;
const SPIKE_ID: u32 = 8;
const PIT_SPIKE_ID: u32 = 9;
const HALF_SPIKE_ID: u32 = 39;
const SMALL_SPIKE_ID: u32 = 103;
const PORTAL_IDS: [u32; 2] = [12, 13];
const DECORATION_IDS: [u32; 17] = [
    15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 29, 30, 41, 54, 62, 65, 142,
];

pub fn object_kind(id: u32) -> Option<ObjectKind> {
    if BLOCK_IDS.contains(&id) {
        Some(ObjectKind::Solid(BLOCK_SIZE_PX))
    } else if id == SLAB_ID {
        Some(ObjectKind::Solid(SLAB_SIZE_PX))
    } else if id == SPIKE_ID {
        Some(ObjectKind::Spike(SPIKE_SIZE_PX))
    } else if id == PIT_SPIKE_ID {
        Some(ObjectKind::Spike(PIT_SPIKE_SIZE_PX))
    } else if id == HALF_SPIKE_ID {
        Some(ObjectKind::Spike(HALF_SPIKE_SIZE_PX))
    } else if id == SMALL_SPIKE_ID {
        Some(ObjectKind::Spike(SMALL_SPIKE_SIZE_PX))
    } else if PORTAL_IDS.contains(&id) {
        Some(ObjectKind::NotSimulated)
    } else if DECORATION_IDS.contains(&id) {
        Some(ObjectKind::Decoration)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNLISTED_ID: u32 = 9_999;

    #[test]
    fn every_block_id_is_a_full_square_solid() {
        for id in BLOCK_IDS {
            assert_eq!(object_kind(id), Some(ObjectKind::Solid(BLOCK_SIZE_PX)));
        }
    }

    #[test]
    fn spikes_carry_their_own_sizes() {
        assert_eq!(
            object_kind(SPIKE_ID),
            Some(ObjectKind::Spike(SPIKE_SIZE_PX))
        );
        assert_eq!(
            object_kind(SMALL_SPIKE_ID),
            Some(ObjectKind::Spike(SMALL_SPIKE_SIZE_PX))
        );
    }

    #[test]
    fn portals_are_known_but_not_simulated() {
        for id in PORTAL_IDS {
            assert_eq!(object_kind(id), Some(ObjectKind::NotSimulated));
        }
    }

    #[test]
    fn decoration_ids_do_not_collide_with_gameplay_ids() {
        for id in DECORATION_IDS {
            assert_eq!(object_kind(id), Some(ObjectKind::Decoration));
        }
    }

    #[test]
    fn unlisted_ids_are_unknown() {
        assert_eq!(object_kind(UNLISTED_ID), None);
    }
}
