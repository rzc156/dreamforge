"""Tests for the dataset builder filters (no GPU, tiny synthetic fixtures)."""
import json

import numpy as np

from dreamforge.dataset import BuildConfig, build_dataset


def _write_fixtures(root):
    mine = root / "mine"
    imag = root / "imagine"
    verify = root / "verify"
    for d in (mine, imag, verify):
        d.mkdir(parents=True)
    name = "libero_spatial__task00__ep000.npz"
    S, H, A = 3, 4, 7
    np.savez(mine / name,
             image=np.zeros((S, 8, 8, 3), np.uint8), image2=np.zeros((S, 8, 8, 3), np.uint8),
             state=np.zeros((S, 8), np.float32), task_desc="pick the bowl",
             success=np.array(False), decision_steps=np.arange(S) * 50,
             task_name="libero_spatial/0", task_id=0, episode=0)
    np.savez(imag / name,
             correction=np.ones((S, H, A), np.float32), random_actions=np.zeros((S, H, A), np.float32),
             wm_best_score=np.array([0.6, 0.42, 0.55], np.float32),
             wm_init_score=np.array([0.5, 0.40, 0.40], np.float32),
             progress_curve=np.zeros((S, H), np.float32), history=np.zeros((S, 2), np.float32),
             decision_steps=np.arange(S) * 50, task_name="libero_spatial/0", task_id=0,
             episode=0, success=np.array(False), source_file=name)
    # verify: state 0 lifted, state 1 not, state 2 none missing
    (verify / "none.jsonl").write_text(
        json.dumps({"file": name, "decision": 0, "arm": "none", "success": False}) + "\n" +
        json.dumps({"file": name, "decision": 1, "arm": "none", "success": True}) + "\n")
    (verify / "imagined.jsonl").write_text(
        json.dumps({"file": name, "decision": 0, "arm": "imagined", "success": True}) + "\n" +
        json.dumps({"file": name, "decision": 1, "arm": "imagined", "success": True}) + "\n")
    return mine, imag, verify


def test_filters(tmp_path):
    mine, imag, verify = _write_fixtures(tmp_path)

    all_out = tmp_path / "all"
    s = build_dataset(BuildConfig(str(mine), str(imag), str(all_out), filter_mode="all"))
    assert s["n_samples"] == 3

    gained_out = tmp_path / "gained"
    s = build_dataset(BuildConfig(str(mine), str(imag), str(gained_out), filter_mode="gained",
                                  wm_gain=0.05))
    # gains: +0.10, +0.02, +0.15 -> 2 kept
    assert s["n_samples"] == 2

    lifted_out = tmp_path / "lifted"
    s = build_dataset(BuildConfig(str(mine), str(imag), str(lifted_out), filter_mode="lifted",
                                  verify_dir=str(verify)))
    # state 0: imagined ok & none fail -> keep; state 1: none succeeded -> drop; state 2: no verify -> drop
    assert s["n_samples"] == 1

    d = np.load(sorted(lifted_out.glob("samples_*.npz"))[0], allow_pickle=True)
    assert d["action"].shape == (1, 4, 7)
    assert str(d["task"][0]) == "pick the bowl"
