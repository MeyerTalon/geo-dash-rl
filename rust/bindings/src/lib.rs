use std::fs;
use std::num::{NonZeroU8, NonZeroU16, NonZeroUsize};

use geo_dash_engine::action::Action;
use geo_dash_engine::env::{EnvConfig as EngineEnvConfig, Termination};
use geo_dash_engine::level::Level;
use geo_dash_engine::level_pool::LevelPool;
use geo_dash_engine::render::ObsSpec;
use geo_dash_engine::vec_env::VecEnv as EngineVecEnv;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

type StepResult<'py> = (Bound<'py, PyBytes>, Vec<f32>, Vec<bool>, Vec<f32>);

#[pyclass(frozen)]
struct EnvConfig {
    frame_skip: u8,
    frame_stack: u8,
    level_length_units: f32,
    obs_width_px: u16,
    obs_height_px: u16,
    death_penalty: f32,
    completion_bonus: f32,
}

#[pymethods]
impl EnvConfig {
    #[new]
    fn new(
        frame_skip: u8,
        frame_stack: u8,
        level_length_units: f32,
        obs_width_px: u16,
        obs_height_px: u16,
        death_penalty: f32,
        completion_bonus: f32,
    ) -> Self {
        Self {
            frame_skip,
            frame_stack,
            level_length_units,
            obs_width_px,
            obs_height_px,
            death_penalty,
            completion_bonus,
        }
    }
}

impl EnvConfig {
    fn to_engine(&self) -> PyResult<EngineEnvConfig> {
        let non_zero_error = |name: &str| PyValueError::new_err(format!("{name} must be positive"));
        if self.level_length_units <= 0.0 {
            return Err(non_zero_error("level_length_units"));
        }
        Ok(EngineEnvConfig {
            frame_skip: NonZeroU8::new(self.frame_skip)
                .ok_or_else(|| non_zero_error("frame_skip"))?,
            level_length_units: self.level_length_units,
            obs: ObsSpec {
                width_px: NonZeroU16::new(self.obs_width_px)
                    .ok_or_else(|| non_zero_error("obs_width_px"))?,
                height_px: NonZeroU16::new(self.obs_height_px)
                    .ok_or_else(|| non_zero_error("obs_height_px"))?,
                frame_stack: NonZeroU8::new(self.frame_stack)
                    .ok_or_else(|| non_zero_error("frame_stack"))?,
            },
            death_penalty: self.death_penalty,
            completion_bonus: self.completion_bonus,
        })
    }
}

fn load_level(path: &str) -> PyResult<Level> {
    let json = fs::read_to_string(path)
        .map_err(|error| PyValueError::new_err(format!("cannot read level {path}: {error}")))?;
    let level: Level = serde_json::from_str(&json)
        .map_err(|error| PyValueError::new_err(format!("invalid level {path}: {error}")))?;
    if !(level.length_units() > 0.0 && level.length_units().is_finite()) {
        return Err(PyValueError::new_err(format!(
            "level {path} must have a positive finite length"
        )));
    }
    Ok(level)
}

fn build_pool(level_paths: Vec<String>, imported_fraction: f32) -> PyResult<LevelPool> {
    if !(0.0..=1.0).contains(&imported_fraction) {
        return Err(PyValueError::new_err("imported_fraction must be in [0, 1]"));
    }
    let levels = level_paths
        .into_iter()
        .map(|path| load_level(&path))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(LevelPool::new(levels, imported_fraction))
}

#[pyclass]
struct VecEnv {
    inner: EngineVecEnv,
}

#[pymethods]
impl VecEnv {
    #[new]
    fn new(
        num_envs: usize,
        seed: u64,
        config: &EnvConfig,
        level_paths: Vec<String>,
        imported_fraction: f32,
    ) -> PyResult<Self> {
        let num_envs = NonZeroUsize::new(num_envs)
            .ok_or_else(|| PyValueError::new_err("num_envs must be positive"))?;
        Ok(Self {
            inner: EngineVecEnv::new(
                config.to_engine()?,
                build_pool(level_paths, imported_fraction)?,
                num_envs,
                seed,
            ),
        })
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        py.detach(|| self.inner.reset());
        PyBytes::new(py, self.inner.obs())
    }

    fn step<'py>(&mut self, py: Python<'py>, actions: &[u8]) -> PyResult<StepResult<'py>> {
        if actions.len() != self.inner.num_envs() {
            return Err(PyValueError::new_err(format!(
                "expected {} actions, got {}",
                self.inner.num_envs(),
                actions.len()
            )));
        }
        let actions = actions
            .iter()
            .map(|&index| Action::try_from(index))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let outcomes = py.detach(|| self.inner.step(&actions));
        Ok((
            PyBytes::new(py, self.inner.obs()),
            outcomes.iter().map(|outcome| outcome.reward).collect(),
            outcomes
                .iter()
                .map(|outcome| outcome.termination != Termination::Running)
                .collect(),
            outcomes
                .iter()
                .map(|outcome| outcome.progress_fraction)
                .collect(),
        ))
    }
}

#[pymodule]
fn geo_dash_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<EnvConfig>()?;
    module.add_class::<VecEnv>()?;
    Ok(())
}
