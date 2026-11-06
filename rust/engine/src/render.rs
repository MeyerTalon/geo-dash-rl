use std::num::NonZeroU16;

use crate::level::Level;
use crate::physics::{PLAYER_SIZE_UNITS, Player};

const VIEW_WIDTH_UNITS: f32 = 16.0;
const VIEW_TOP_UNITS: f32 = 8.0;
const VIEW_HEIGHT_UNITS: f32 = 9.0;
const PLAYER_SCREEN_X_FRACTION: f32 = 0.25;
const PIXEL_CENTER_OFFSET: f32 = 0.5;

const BACKGROUND_VALUE: u8 = 0;
const GROUND_VALUE: u8 = 64;
const SPIKE_VALUE: u8 = 192;
const PLAYER_VALUE: u8 = 255;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObsSpec {
    pub width_px: NonZeroU16,
    pub height_px: NonZeroU16,
}

impl ObsSpec {
    #[must_use]
    pub fn frame_len(self) -> usize {
        usize::from(self.width_px.get()) * usize::from(self.height_px.get())
    }
}

pub fn render_into(player: &Player, level: &Level, spec: ObsSpec, frame: &mut [u8]) {
    let width_px = spec.width_px.get();
    let height_px = spec.height_px.get();
    let camera_x_units = player.x_units - PLAYER_SCREEN_X_FRACTION * VIEW_WIDTH_UNITS;
    let pixels = (0..height_px).flat_map(|row| (0..width_px).map(move |col| (row, col)));
    for (value, (row, col)) in frame.iter_mut().zip(pixels) {
        let x_units = camera_x_units
            + (f32::from(col) + PIXEL_CENTER_OFFSET) / f32::from(width_px) * VIEW_WIDTH_UNITS;
        let y_units = VIEW_TOP_UNITS
            - (f32::from(row) + PIXEL_CENTER_OFFSET) / f32::from(height_px) * VIEW_HEIGHT_UNITS;
        *value = pixel_value(player, level, x_units, y_units);
    }
}

fn pixel_value(player: &Player, level: &Level, x_units: f32, y_units: f32) -> u8 {
    let inside_player = (player.x_units..player.x_units + PLAYER_SIZE_UNITS).contains(&x_units)
        && (player.y_units..player.y_units + PLAYER_SIZE_UNITS).contains(&y_units);
    if inside_player {
        PLAYER_VALUE
    } else if level
        .spikes()
        .iter()
        .any(|spike| spike.covers(x_units, y_units))
    {
        SPIKE_VALUE
    } else if y_units < 0.0 {
        GROUND_VALUE
    } else {
        BACKGROUND_VALUE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const WIDTH_PX: u16 = 64;
    const HEIGHT_PX: u16 = 36;
    const LEVEL_LENGTH_UNITS: f32 = 200.0;

    fn spec() -> ObsSpec {
        ObsSpec {
            width_px: NonZeroU16::new(WIDTH_PX).unwrap(),
            height_px: NonZeroU16::new(HEIGHT_PX).unwrap(),
        }
    }

    fn render(player: &Player) -> Vec<u8> {
        let level = Level::generate(LEVEL_LENGTH_UNITS, &mut StdRng::seed_from_u64(0));
        let mut frame = vec![BACKGROUND_VALUE; spec().frame_len()];
        render_into(player, &level, spec(), &mut frame);
        frame
    }

    #[test]
    fn frame_has_player_ground_and_background_pixels() {
        let frame = render(&Player::at_start());
        assert!(frame.contains(&PLAYER_VALUE));
        assert!(frame.contains(&GROUND_VALUE));
        assert!(frame.contains(&BACKGROUND_VALUE));
    }

    #[test]
    fn ground_fills_the_bottom_row_and_background_fills_the_top_row() {
        let frame = render(&Player::at_start());
        let (top_row, rest) = frame.split_at(usize::from(WIDTH_PX));
        assert!(top_row.iter().all(|&value| value == BACKGROUND_VALUE));
        let bottom_row = rest.rchunks(usize::from(WIDTH_PX)).next().unwrap();
        assert!(bottom_row.iter().all(|&value| value == GROUND_VALUE));
    }

    #[test]
    fn jumping_moves_the_player_pixels_up() {
        let mut jumped = Player::at_start();
        jumped.y_units = 2.0;
        let first_player_row = |frame: &[u8]| {
            frame
                .chunks(usize::from(WIDTH_PX))
                .position(|row| row.contains(&PLAYER_VALUE))
        };
        assert!(
            first_player_row(&render(&jumped)) < first_player_row(&render(&Player::at_start()))
        );
    }
}
