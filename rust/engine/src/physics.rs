use crate::action::Action;

pub const TICKS_PER_SECOND: f32 = 240.0;
pub const PLAYER_SIZE_UNITS: f32 = 1.0;

const TICK_S: f32 = 1.0 / TICKS_PER_SECOND;
const RUN_SPEED_UNITS_PER_S: f32 = 10.386;
const GRAVITY_UNITS_PER_S2: f32 = 94.0;
const JUMP_SPEED_UNITS_PER_S: f32 = 19.9;
const GROUND_Y_UNITS: f32 = 0.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Player {
    pub x_units: f32,
    pub y_units: f32,
    pub vertical_speed_units_per_s: f32,
    pub on_ground: bool,
}

impl Player {
    #[must_use]
    pub fn at_start() -> Self {
        Self {
            x_units: 0.0,
            y_units: GROUND_Y_UNITS,
            vertical_speed_units_per_s: 0.0,
            on_ground: true,
        }
    }

    pub fn tick(&mut self, action: Action) {
        if action == Action::Hold && self.on_ground {
            self.vertical_speed_units_per_s = JUMP_SPEED_UNITS_PER_S;
            self.on_ground = false;
        }
        self.vertical_speed_units_per_s -= GRAVITY_UNITS_PER_S2 * TICK_S;
        self.y_units += self.vertical_speed_units_per_s * TICK_S;
        self.x_units += RUN_SPEED_UNITS_PER_S * TICK_S;
        if self.y_units <= GROUND_Y_UNITS {
            self.y_units = GROUND_Y_UNITS;
            self.vertical_speed_units_per_s = 0.0;
            self.on_ground = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE_UNITS: f32 = 0.05;
    const ONE_SECOND_TICKS: u16 = 240;

    #[test]
    fn running_without_input_stays_on_the_ground_and_moves_forward() {
        let mut player = Player::at_start();
        for _ in 0..ONE_SECOND_TICKS {
            player.tick(Action::Release);
        }
        assert!(player.on_ground);
        assert!((player.x_units - RUN_SPEED_UNITS_PER_S).abs() < TOLERANCE_UNITS);
    }

    #[test]
    fn holding_jumps_to_the_ballistic_apex_and_lands() {
        let mut player = Player::at_start();
        let apex_units = JUMP_SPEED_UNITS_PER_S.powi(2) / (2.0 * GRAVITY_UNITS_PER_S2);
        let mut peak_units = 0.0_f32;
        player.tick(Action::Hold);
        assert!(!player.on_ground);
        while !player.on_ground {
            peak_units = peak_units.max(player.y_units);
            player.tick(Action::Release);
        }
        assert!((peak_units - apex_units).abs() < TOLERANCE_UNITS);
        assert!(player.y_units.abs() < f32::EPSILON);
    }
}
