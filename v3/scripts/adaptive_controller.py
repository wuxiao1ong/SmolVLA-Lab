"""Action-volatility-aware replanning for LeRobot v0.6.1 SmolVLAPolicy.

The model predicts the SAME 50-action chunk. This controller only discards
unused tail actions to change how frequently the policy observes a new scene.
No model weights, normalization steps, or environment dynamics are modified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def volatility_score(chunk: torch.Tensor, window: int = 25) -> float:
    """Mean L2 distance between consecutive predicted 6-DoF actions.

    chunk: [1, H, D] tensor in model output coordinates (BEFORE postprocessor).
    Not a metric of physical distance or calibrated uncertainty.
    """
    if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[-1] < 6:
        raise ValueError(f"expected [1, H, D>=6], got {tuple(chunk.shape)}")
    k = min(int(window), chunk.shape[1])
    if k < 2:
        return 0.0
    six_d = chunk[0, :k, :6].float()
    return torch.linalg.vector_norm(six_d[1:] - six_d[:-1], dim=-1).mean().item()


def select_horizon(mode: str, score: float, short: int, long: int, threshold: float) -> int:
    if mode in {"fixed", "probe"}:
        return long
    if mode == "adaptive":
        return short if score >= threshold else long
    raise ValueError(f"unknown mode: {mode}")


def install_controller(
    policy_cls: type,
    action_key: str,
    *,
    mode: str,
    short: int,
    long: int,
    threshold: float,
    trace: str,
):
    """Patch class-level queue management only within the eval process.

    Works for batch_size=1 and disabled RTC. Fails loudly if the upstream
    SmolVLA implementation changes. Patched original select_action still
    handles all preprocessing and model sampling.
    """
    if not (1 <= short <= long <= 50):
        raise ValueError("require 1 <= short <= long <= 50")
    if mode not in {"fixed", "probe", "adaptive"}:
        raise ValueError("mode must be fixed, probe, or adaptive")
    p = Path(trace)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"trace already exists: {p}")

    original_action = policy_cls.select_action
    original_reset = policy_cls.reset

    def record(obj: dict[str, Any]) -> None:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def patched_reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        self._planner_rollout_idx = getattr(self, "_planner_rollout_idx", -1) + 1
        self._planner_t = 0
        record({"event": "reset", "rollout": self._planner_rollout_idx})
        return result

    def patched_select(self, batch, *args, **kwargs):
        if getattr(self.config, "rtc_config", None) is not None and getattr(self.config.rtc_config, "enabled", False):
            raise RuntimeError("RTC mode not supported by this controller")
        if int(self.config.n_action_steps) != int(self.config.chunk_size):
            raise RuntimeError("set --policy.n_action_steps=50 to expose full chunk")
        q = self._queues[action_key]
        new_chunk = len(q) == 0
        result = original_action(self, batch, *args, **kwargs)
        if new_chunk:
            if result.ndim != 2 or result.shape[0] != 1:
                raise ValueError("controller requires eval.batch_size=1")
            chunk = torch.stack([result] + list(self._queues[action_key]), dim=1)
            value = volatility_score(chunk, window=long)
            length = select_horizon(mode, value, short, long, threshold)
            while len(self._queues[action_key]) > length - 1:
                self._queues[action_key].pop()  # discard tail, keep imminent actions
            record({"event": "chunk", "rollout": self._planner_rollout_idx,
                    "step": self._planner_t, "score": round(value, 9),
                    "horizon": length, "mode": mode,
                    "short": short, "long": long, "threshold": threshold})
        self._planner_t = getattr(self, "_planner_t", 0) + 1
        return result

    policy_cls.select_action = patched_select
    policy_cls.reset = patched_reset
    record({"event": "configuration", "mode": mode, "short": short,
            "long": long, "threshold": threshold,
            "score": "mean L2 of consecutive first-six model-space actions"})
