use std::num::NonZeroUsize;

use rayon::prelude::*;

use crate::action::Action;
use crate::env::{Env, EnvConfig, StepOutcome, Termination};

pub struct VecEnv {
    envs: Vec<Env>,
    frame_len: usize,
    obs: Vec<u8>,
}

impl VecEnv {
    #[must_use]
    pub fn new(config: EnvConfig, num_envs: NonZeroUsize, seed: u64) -> Self {
        let frame_len = config.obs.frame_len();
        let envs: Vec<Env> = (seed..)
            .take(num_envs.get())
            .map(|env_seed| Env::new(config, env_seed))
            .collect();
        let obs = vec![0; frame_len * envs.len()];
        let mut vec_env = Self {
            envs,
            frame_len,
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
        self.envs
            .par_iter_mut()
            .zip(self.obs.par_chunks_mut(self.frame_len))
            .for_each(|(env, frame)| {
                env.reset();
                env.render_into(frame);
            });
    }

    pub fn step(&mut self, actions: &[Action]) -> Vec<StepOutcome> {
        self.envs
            .par_iter_mut()
            .zip(self.obs.par_chunks_mut(self.frame_len))
            .zip(actions.par_iter())
            .map(|((env, frame), &action)| {
                let outcome = env.step(action);
                if outcome.termination != Termination::Running {
                    env.reset();
                }
                env.render_into(frame);
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
    const MAX_STEPS: u32 = 10_000;

    fn config() -> EnvConfig {
        EnvConfig {
            frame_skip: NonZeroU8::new(4).unwrap(),
            level_length_units: 200.0,
            obs: ObsSpec {
                width_px: NonZeroU16::new(WIDTH_PX).unwrap(),
                height_px: NonZeroU16::new(HEIGHT_PX).unwrap(),
            },
            death_penalty: 1.0,
            completion_bonus: 1.0,
        }
    }

    fn vec_env(seed: u64) -> VecEnv {
        VecEnv::new(config(), NonZeroUsize::new(NUM_ENVS).unwrap(), seed)
    }

    #[test]
    fn obs_holds_one_frame_per_env() {
        let env = vec_env(0);
        assert_eq!(env.num_envs(), NUM_ENVS);
        assert_eq!(
            env.obs().len(),
            NUM_ENVS * usize::from(WIDTH_PX) * usize::from(HEIGHT_PX)
        );
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
