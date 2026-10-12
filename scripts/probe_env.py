"""接口探测：确认 lerobot LIBERO 环境、观测格式、robosuite 状态序列化 API。

只打印信息，不做任何实验。
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

import libero.libero  # noqa: F401
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

POLICY_PATH = "/media/rzc/PS3000/embodied-ai/models/smolvla_libero"

def main() -> None:
    env_cfg = LiberoEnv(task="libero_spatial")
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    print("type(make_env):", type(envs))
    task_envs = []
    def flat(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                flat(v, f"{p}/{k}" if p else str(k))
        else:
            task_envs.append((p, o))
    flat(envs)
    print("task_envs:", [(p, type(e).__name__) for p, e in task_envs][:4], f"... total={len(task_envs)}")

    name, tenv = task_envs[0]
    print("wrapper type:", type(tenv))
    print("wrapper attrs:", [a for a in dir(tenv) if not a.startswith("__")][:40])
    obs, info = tenv.reset(seed=123)
    print("obs type:", type(obs), "keys:", list(obs.keys()) if isinstance(obs, dict) else obs)
    for k, v in (obs.items() if isinstance(obs, dict) else []):
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype} min={v.min()} max={v.max()}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype} min={float(v.min())} max={float(v.max())}")
        else:
            print(f"  {k}: {type(v)}")

    print("task desc:", list(tenv.call("task_description"))[0])
    raw = tenv.envs[0]._env
    print("raw env type:", type(raw))
    sim = raw.sim
    st = sim.get_state()
    print("MjSimState type:", type(st), "| flatten shape:", st.flatten().shape)
    from robosuite.utils.binding_utils import MjSimState
    st2 = MjSimState.from_flattened(st.flatten(), sim)
    sim.set_state(st2)
    sim.forward()
    print("state round-trip OK")

    # 再取一步 obs，观察 preprocess_observation 处理
    print("pixels keys:", list(obs["pixels"].keys()), "| robot_state keys:", list(obs["robot_state"].keys()))
    for k, v in obs["pixels"].items():
        print(f"  pixels/{k}: {getattr(v, 'shape', None)} {getattr(v, 'dtype', None)}")
    for k, v in obs["robot_state"].items():
        print(f"  robot_state/{k}: {getattr(v, 'shape', None)} {getattr(v, 'dtype', None)}")
    a = np.zeros((1, 7), dtype=np.float32)
    obs2, rew, term, trunc, info2 = tenv.step(a)
    print("step info keys:", list(info2.keys()) if isinstance(info2, dict) else info2)
    o = preprocess_observation(obs2)
    print("preprocess_observation keys:", list(o.keys()))
    for k, v in o.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray {v.shape} {v.dtype}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")

    # policy 侧
    cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    cfg.device = "cuda"; cfg.pretrained_path = POLICY_PATH
    rm = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=POLICY_PATH,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": rm}})

    o = preprocess_observation(obs2)
    o["task"] = [list(tenv.call("task_description"))[0]]
    o1 = env_pre(o)
    print("after env_pre keys:", list(o1.keys()))
    for k, v in o1.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray {v.shape} {v.dtype} {v.min():.3f}..{v.max():.3f}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
    o2 = pre(o1)
    print("after pre keys:", list(o2.keys()))
    for k, v in o2.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: tensor {tuple(v.shape)} {v.dtype} min={float(v.min()):.3f} max={float(v.max()):.3f}")
    print("chunk_size:", getattr(cfg, "chunk_size", None), "n_action_steps:", getattr(cfg, "n_action_steps", None))

    policy = make_policy(cfg, env_cfg=env_cfg, rename_map=rm); policy.eval()
    with torch.no_grad():
        chunk = policy.predict_action_chunk(o2)
    print("chunk shape:", tuple(chunk.shape), "| raw(normalized) range:", float(chunk.min()), float(chunk.max()))
    a_raw = post(chunk[:, 0, :])
    print("post(chunk[:,0]) shape:", tuple(a_raw.shape), "range:", float(a_raw.min()), float(a_raw.max()))
    a_env = env_post({"action": a_raw})["action"]
    print("env action shape:", getattr(a_env, "shape", None), "range:", float(a_env.min()), float(a_env.max()), "type:", type(a_env))

    # 归一化统计（用于确认 CEM 动作空间与 WM 训练空间一致）
    import json
    pre_json = json.load(open(f"{POLICY_PATH}/policy_preprocessor.json"))
    steps = pre_json.get("steps", pre_json)
    for s in steps:
        if s.get("registry_name", "").startswith("normalizer") or "normalizer" in str(s.get("registry_name", "")):
            print("normalizer step:", s.get("registry_name"), "state:", s.get("state") if not isinstance(s.get("state"), dict) else "dict")
            st = s.get("state")
            if isinstance(st, dict):
                for k, v in st.items():
                    if isinstance(v, dict):
                        print("  ", k, {kk: (vv if not isinstance(vv, list) else np.round(np.array(vv), 4).tolist()) for kk, vv in v.items() if kk in ("mean", "std", "min", "max")})


if __name__ == "__main__":
    main()
