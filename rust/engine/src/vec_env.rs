use std::num::NonZeroUsize;
use std::sync::Arc;

use rayon::prelude::*;

use crate::action::Action;
use crate::env::{Env, EnvConfig, StepOutcome, Termination};
use crate::frame_stack::{fill_frames, push_frame};
use crate::level_pool::LevelPool;

pub struct VecEnv {
    envs: Vec<Env>,
    frame_len: usize,
    stack_len: usize,
    obs: Vec<u8>,
}

impl VecEnv {
    #[must_use]
    pub fn new(config: EnvConfig, pool: LevelPool, num_envs: NonZeroUsize, seed: u64) -> Self {
        let pool = Arc::new(pool);
        let frame_len = config.obs.frame_len();
        let stack_len = config.obs.stack_len();
        let envs: Vec<Env> = (seed..)
            .take(num_envs.get())
            .map(|env_seed| Env::new(config, Arc::clone(&pool), env_seed))
            .collect();
        let obs = vec![0; stack_len * envs.len()];
        let mut vec_env = Self {
            envs,
            frame_len,
            stack_len,
            obs,
        };
        vec_env.reset();
        vec_env
    }

    #[must_use]
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    #[must_use]
    pub fn obs(&self) -> &[u8] {
        &self.obs
    }

    pub fn reset(&mut self) {
        let frame_len = self.frame_len;
        self.envs
            .par_iter_mut()
            .zip(self.obs.par_chunks_mut(self.stack_len))
            .for_each(|(env, stack)| {
                env.reset();
                fill_frames(stack, frame_len, |frame| env.render_into(frame));
            });
    }

    pub fn step(&mut self, actions: &[Action]) -> Vec<StepOutcome> {
        let frame_len = self.frame_len;
        self.envs
            .par_iter_mut()
            .zip(self.obs.par_chunks_mut(self.stack_len))
            .zip(actions.par_iter())
            .map(|((env, stack), &action)| {
                let outcome = env.step(action);
                if outcome.termination == Termination::Running {
                    push_frame(stack, frame_len, |frame| env.render_into(frame));
                } else {
                    env.reset();
                    fill_frames(stack, frame_len, |frame| env.render_into(frame));
                }
                outcome
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::ObsSpec;
    use std::num::{NonZeroU8, NonZeroU16};

    const NUM_ENVS: usize = 3;
    const WIDTH_PX: u16 = 8;
    const HEIGHT_PX: u16 = 8;
    const FRAME_STACK: u8 = 4;
    const FRAME_LEN: usize = WIDTH_PX as usize * HEIGHT_PX as usize;
    const STACK_LEN: usize = FRAME_LEN * FRAME_STACK as usize;
    const MAX_STEPS: u32 = 10_000;

    fn config() -> EnvConfig {
        EnvConfig {
            frame_skip: NonZeroU8::new(4).unwrap(),
            level_length_units: 200.0,
            obs: ObsSpec {
                width_px: NonZeroU16::new(WIDTH_PX).unwrap(),
                height_px: NonZeroU16::new(HEIGHT_PX).unwrap(),
                frame_stack: NonZeroU8::new(FRAME_STACK).unwrap(),
            },
            death_penalty: 1.0,
            completion_bonus: 1.0,
        }
    }

    fn vec_env(seed: u64) -> VecEnv {
        VecEnv::new(
            config(),
            LevelPool::new(Vec::new(), 0.0),
            NonZeroUsize::new(NUM_ENVS).unwrap(),
            seed,
        )
    }

    #[test]
    fn obs_holds_one_stack_per_env() {
        let env = vec_env(0);
        assert_eq!(env.num_envs(), NUM_ENVS);
        assert_eq!(env.obs().len(), NUM_ENVS * STACK_LEN);
    }

    fn assert_every_frame_matches_the_newest(stack: &[u8]) {
        let frames: Vec<&[u8]> = stack.chunks_exact(FRAME_LEN).collect();
        assert!(frames.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn reset_fills_each_stack_with_the_first_frame() {
        let env = vec_env(0);
        for stack in env.obs().chunks_exact(STACK_LEN) {
            assert_every_frame_matches_the_newest(stack);
        }
    }

    #[test]
    fn step_shifts_each_stack_towards_the_oldest_slot() {
        let mut env = vec_env(0);
        let actions = [Action::Hold; NUM_ENVS];
        env.step(&actions);
        let before = env.obs().to_vec();
        let outcomes = env.step(&actions);
        assert!(
            outcomes
                .iter()
                .all(|outcome| outcome.termination == Termination::Running)
        );
        for (old, new) in before
            .chunks_exact(STACK_LEN)
            .zip(env.obs().chunks_exact(STACK_LEN))
        {
            assert_eq!(new[..STACK_LEN - FRAME_LEN], old[FRAME_LEN..]);
        }
    }

    #[test]
    fn terminated_env_refills_its_stack_so_frames_never_cross_episodes() {
        let mut env = vec_env(0);
        let actions = [Action::Release; NUM_ENVS];
        for _ in 0..MAX_STEPS {
            let outcomes = env.step(&actions);
            let finished = outcomes
                .iter()
                .position(|outcome| outcome.termination != Termination::Running);
            if let Some(index) = finished {
                let stack = env.obs().chunks_exact(STACK_LEN).nth(index).unwrap();
                assert_every_frame_matches_the_newest(stack);
                return;
            }
        }
        panic!("no env finished an episode");
    }

    #[test]
    fn terminated_envs_reset_and_keep_running() {
        let mut env = vec_env(0);
        let actions = [Action::Release; NUM_ENVS];
        let mut deaths = 0;
        for _ in 0..MAX_STEPS {
            let outcomes = env.step(&actions);
            assert_eq!(outcomes.len(), NUM_ENVS);
            deaths += outcomes
                .iter()
                .filter(|outcome| outcome.termination == Termination::Died)
                .count();
            if deaths >= 2 * NUM_ENVS {
                return;
            }
        }
        panic!("envs did not die twice each after auto reset");
    }

    #[test]
    fn same_seed_gives_the_same_frames() {
        let mut first = vec_env(5);
        let mut second = vec_env(5);
        let actions = [Action::Hold, Action::Release, Action::Hold];
        for _ in 0..10 {
            first.step(&actions);
            second.step(&actions);
        }
        assert_eq!(first.obs(), second.obs());
    }
}
