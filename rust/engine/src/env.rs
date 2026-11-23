use std::num::NonZeroU8;
use std::sync::Arc;

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::action::Action;
use crate::level::Level;
use crate::level_pool::LevelPool;
use crate::physics::{PLAYER_SIZE_UNITS, Player};
use crate::render::{ObsSpec, render_into};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvConfig {
    pub frame_skip: NonZeroU8,
    pub level_length_units: f32,
    pub obs: ObsSpec,
    pub death_penalty: f32,
    pub completion_bonus: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    Running,
    Died,
    Completed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOutcome {
    pub reward: f32,
    pub termination: Termination,
    pub progress_fraction: f32,
}

pub struct Env {
    config: EnvConfig,
    pool: Arc<LevelPool>,
    rng: StdRng,
    level: Arc<Level>,
    player: Player,
}

impl Env {
    #[must_use]
    pub fn new(config: EnvConfig, pool: Arc<LevelPool>, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let level = pool.pick(config.level_length_units, &mut rng);
        Self {
            config,
            pool,
            rng,
            level,
            player: Player::at_start(),
        }
    }

    pub fn reset(&mut self) {
        self.level = self
            .pool
            .pick(self.config.level_length_units, &mut self.rng);
        self.player = Player::at_start();
    }

    pub fn step(&mut self, action: Action) -> StepOutcome {
        let progress_before = self.progress_fraction();
        let mut termination = Termination::Running;
        for _ in 0..self.config.frame_skip.get() {
            self.player.tick(action);
            termination = self.termination();
            if termination != Termination::Running {
                break;
            }
        }
        let progress_fraction = self.progress_fraction();
        let terminal_reward = match termination {
            Termination::Running => 0.0,
            Termination::Died => -self.config.death_penalty,
            Termination::Completed => self.config.completion_bonus,
        };
        StepOutcome {
            reward: progress_fraction - progress_before + terminal_reward,
            termination,
            progress_fraction,
        }
    }

    pub fn render_into(&self, frame: &mut [u8]) {
        render_into(&self.player, &self.level, self.config.obs, frame);
    }

    fn progress_fraction(&self) -> f32 {
        (self.player.x_units / self.level.length_units()).min(1.0)
    }

    fn termination(&self) -> Termination {
        if self
            .level
            .is_deadly_box(self.player.x_units, self.player.y_units, PLAYER_SIZE_UNITS)
        {
            Termination::Died
        } else if self.player.x_units >= self.level.length_units() {
            Termination::Completed
        } else {
            Termination::Running
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU16;

    const TOLERANCE: f32 = 1e-4;
    const DEATH_PENALTY: f32 = 1.0;
    const COMPLETION_BONUS: f32 = 2.0;
    const SPIKE_FREE_LENGTH_UNITS: f32 = 5.0;
    const SPIKED_LENGTH_UNITS: f32 = 200.0;
    const MAX_STEPS: u32 = 10_000;

    fn config(level_length_units: f32) -> EnvConfig {
        EnvConfig {
            frame_skip: NonZeroU8::new(4).unwrap(),
            level_length_units,
            obs: ObsSpec {
                width_px: NonZeroU16::new(8).unwrap(),
                height_px: NonZeroU16::new(8).unwrap(),
                frame_stack: NonZeroU8::MIN,
            },
            death_penalty: DEATH_PENALTY,
            completion_bonus: COMPLETION_BONUS,
        }
    }

    fn generated_pool() -> Arc<LevelPool> {
        Arc::new(LevelPool::new(Vec::new(), 0.0))
    }

    fn run_until_done(env: &mut Env, action: Action) -> (StepOutcome, f32) {
        let mut total_reward = 0.0;
        for _ in 0..MAX_STEPS {
            let outcome = env.step(action);
            total_reward += outcome.reward;
            if outcome.termination != Termination::Running {
                return (outcome, total_reward);
            }
        }
        panic!("episode never terminated");
    }

    #[test]
    fn idling_into_a_spike_dies_with_the_penalty() {
        let mut env = Env::new(config(SPIKED_LENGTH_UNITS), generated_pool(), 0);
        let (outcome, total_reward) = run_until_done(&mut env, Action::Release);
        assert_eq!(outcome.termination, Termination::Died);
        assert!(outcome.progress_fraction < 1.0);
        let expected = outcome.progress_fraction - DEATH_PENALTY;
        assert!((total_reward - expected).abs() < TOLERANCE);
    }

    #[test]
    fn reaching_the_end_of_a_spike_free_level_completes_with_the_bonus() {
        let mut env = Env::new(config(SPIKE_FREE_LENGTH_UNITS), generated_pool(), 0);
        let (outcome, total_reward) = run_until_done(&mut env, Action::Release);
        assert_eq!(outcome.termination, Termination::Completed);
        assert!((outcome.progress_fraction - 1.0).abs() < TOLERANCE);
        assert!((total_reward - (1.0 + COMPLETION_BONUS)).abs() < TOLERANCE);
    }

    #[test]
    fn reset_returns_the_player_to_the_start() {
        let mut env = Env::new(config(SPIKED_LENGTH_UNITS), generated_pool(), 0);
        run_until_done(&mut env, Action::Release);
        env.reset();
        let outcome = env.step(Action::Release);
        assert_eq!(outcome.termination, Termination::Running);
        assert!(outcome.progress_fraction < 0.01);
    }
}
