"""LIBERO environment helpers (via LeRobot's env integration).

Everything here works on the *raw robosuite env* underneath LeRobot's gym
wrapper, because we need state save/restore and fine-grained control over
terminal flags (the vector env auto-resets on termination).

Key facts (verified on the target machine, lerobot 0.6.1 / robosuite 1.4.0):
  - ``make_env`` returns {suite: {task_id: SyncVectorEnv}}; flatten to a list.
  - raw env:         gym_env.envs[0]._env               (OffScreenRenderEnv)
  - inner robosuite: gym_env.envs[0]._env.env           (_get_observations)
  - sim state:       raw.sim.get_state().flatten()      -> float64 array
                     MjSimState.from_flattened(flat, raw.sim)
  - success:         raw.check_success()
  - observation:     gym_env._format_raw_obs(raw.env._get_observations())
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

__all__ = [
    "make_task_envs", "raw_env", "inner_env", "sim_of", "clear_done",
    "save_sim_state", "load_sim_state", "obs_from_state", "task_description",
    "step_raw", "step_checked", "is_success", "env_horizon",
]


def make_task_envs(task: str, n_envs: int = 1) -> list[tuple[str, Any]]:
    """Create one LIBERO vector env per task of a suite.

    Returns a list of ``(task_name, gym_env)`` pairs, e.g. ``('libero_spatial/0', SyncVectorEnv)``.
    """
    import libero.libero  # noqa: F401  (registers the environments)
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env

    env_cfg = LiberoEnv(task=task)
    envs = make_env(env_cfg, n_envs=n_envs, use_async_envs=False)

    flat: list[tuple[str, Any]] = []

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}/{k}" if prefix else str(k))
        else:
            flat.append((prefix, obj))

    walk(envs)
    return flat


def raw_env(gym_env: Any) -> Any:
    return gym_env.envs[0]._env


def inner_env(gym_env: Any) -> Any:
    return raw_env(gym_env).env


def _batchify(x: Any) -> Any:
    """Add the leading (env) dimension so observations match the vector-env format."""
    if isinstance(x, dict):
        return {k: _batchify(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[None, ...]
    return x


def _format_obs(gym_env: Any, raw_obs: dict) -> dict:
    """Build the LeRobot observation dict using the inner wrapper's formatter.

    The result is batched (leading dim 1), matching what the LeRobot vector env
    returns, which is what the env pre-processors expect.
    """
    return _batchify(gym_env.envs[0]._format_raw_obs(raw_obs))


def sim_of(gym_env: Any) -> Any:
    return raw_env(gym_env).sim


def clear_done(gym_env: Any) -> None:
    """Clear terminal flags along the wrapper chain (probing triggers them)."""
    obj = gym_env
    for _ in range(6):
        if obj is None:
            break
        if hasattr(obj, "done"):
            try:
                obj.done = False
            except Exception:
                pass
        obj = getattr(obj, "env", None)


def save_sim_state(gym_env: Any) -> np.ndarray:
    return np.asarray(sim_of(gym_env).get_state().flatten(), dtype=np.float64)


def load_sim_state(gym_env: Any, flat: np.ndarray) -> None:
    from robosuite.utils.binding_utils import MjSimState

    sim = sim_of(gym_env)
    state = MjSimState.from_flattened(np.asarray(flat, dtype=np.float64), sim)
    sim.set_state(state)
    sim.forward()
    clear_done(gym_env)


def obs_from_state(gym_env: Any) -> dict:
    """Observation dict in LeRobot format (pixels / robot_state) at the current sim state."""
    raw = raw_env(gym_env)
    raw_obs = raw.env._get_observations()
    # controller orientation matrix is refreshed from the restored sim state
    try:
        raw.env.robots[0].controller.update()
    except Exception:
        pass
    return _format_obs(gym_env, raw_obs)


def task_description(gym_env: Any) -> str:
    return list(gym_env.call("task_description"))[0]


def step_raw(gym_env: Any, action: np.ndarray) -> tuple[dict, bool, bool]:
    """Step the raw robosuite env with a 1-D action.

    Returns ``(obs, success, done)`` where ``obs`` is in LeRobot format.
    """
    raw = raw_env(gym_env)
    clear_done(gym_env)
    raw_obs, _reward, done, _info = raw.step(np.asarray(action, dtype=np.float64).reshape(-1))
    success = bool(raw.check_success())
    obs = _format_obs(gym_env, raw_obs)
    return obs, success, bool(done or success)


def step_checked(gym_env: Any, action: np.ndarray) -> tuple[dict | None, bool, bool]:
    """Robust raw step: works on a *live* env and tolerates already-terminated states.

    Returns ``(obs, success, done)``; ``obs`` is None when the env was already
    terminated (the step was refused by robosuite).
    """
    raw = raw_env(gym_env)
    try:
        raw_obs, _reward, done, _info = raw.step(np.asarray(action, dtype=np.float64).reshape(-1))
    except ValueError as exc:  # robosuite: "executing action in terminated episode"
        if "terminated" in str(exc):
            return None, bool(raw.check_success()), True
        raise
    success = bool(raw.check_success())
    obs = _format_obs(gym_env, raw_obs)
    return obs, success, bool(done or success)


def is_success(info: dict) -> bool:
    v = info.get("is_success", False)
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return any(bool(x) for x in v)
    return bool(v)


_LIBERO_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_long": 520,
    "libero_90": 400,
}


def env_horizon(gym_env: Any, task: str | None = None) -> int:
    """Max steps per episode under the standard LIBERO protocol.

    The raw robosuite env reports horizon=1000 (its default); LIBERO evaluation
    caps episodes per suite (280/300/400/520).  We follow the LIBERO convention.
    """
    if task in _LIBERO_MAX_STEPS:
        return _LIBERO_MAX_STEPS[task]
    inner = inner_env(gym_env)
    for attr in ("horizon", "episode_length", "max_episode_steps"):
        v = getattr(inner, attr, None)
        if isinstance(v, (int, float)) and 0 < v < 1000:
            return int(v)
    return 300
