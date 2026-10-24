"""AAC (2026) + BID-inspired candidate reranking for frozen SmolVLA.

AAC uses regularized 3D Gaussian entropies and binary gripper entropy.
BID adaptation uses backward coherence + positive-sample consensus only;
it does NOT implement BID's weak-policy negative contrast.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch


METHODS = {
    "base10": dict(samples=1, adaptive=False, rerank=False),
    "sample4_fixed10": dict(samples=4, adaptive=False, rerank=False),
    "rerank10": dict(samples=4, adaptive=False, rerank=True),
    "aac_first": dict(samples=4, adaptive=True, rerank=False),
    "aac_rerank": dict(samples=4, adaptive=True, rerank=True),
    "aac_nogrip": dict(samples=4, adaptive=True, rerank=False, no_gripper=True),
}


def action_entropy(samples, gripper_boundary=0.0, include_gripper=True, ridge=1e-6):
    """[N,H,7] normalized action candidates -> entropy at each future step.

    Separate 3x3 covariances avoid a rank-deficient 6x6 estimate with N=4.
    Ridge is fixed before evaluations. These are sample-disagreement proxies,
    not calibrated probabilities that a robot action will succeed.
    """
    x = samples.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 3 or x.shape[0] < 4 or x.shape[2] != 7 or not torch.isfinite(x).all():
        raise ValueError("expected finite [N>=4,H,7] action candidates")
    entropy = torch.zeros(x.shape[1], dtype=torch.float64)
    for block in (x[..., :3], x[..., 3:6]):
        centered = block - block.mean(0, keepdim=True)
        cov = torch.einsum("nhi,nhj->hij", centered, centered) / (x.shape[0] - 1)
        cov += ridge * torch.eye(3, dtype=torch.float64)
        sign, logdet = torch.linalg.slogdet(cov)
        if not (sign > 0).all():
            raise ValueError("nonpositive regularized covariance")
        entropy += 0.5 * (3 * math.log(2 * math.pi * math.e) + logdet)
    probability = (x[..., 6] > gripper_boundary).double().mean(0)
    p = probability.clamp(1e-12, 1 - 1e-12)
    gripper = -(p * p.log() + (1 - p) * (1 - p).log())
    if include_gripper:
        entropy += gripper
    return entropy, gripper


def entropy_horizon(entropy, minimum=5, maximum=25):
    """AAC Eq. 5: largest increase in prefix-average entropy, bounded 5..25.

    The maximum bound is this project's locked compute/feedback constraint.
    The paper does not impose this particular SmolVLA-specific maximum.
    """
    if entropy.ndim != 1 or len(entropy) < maximum + 1 or not torch.isfinite(entropy).all():
        raise ValueError("invalid entropy curve")
    prefix = entropy.cumsum(0) / torch.arange(1, len(entropy) + 1, dtype=entropy.dtype)
    boundary = int(torch.argmax(prefix[1:] - prefix[:-1]).item()) + 1
    return max(minimum, min(maximum, boundary))


def rank_candidates(samples, previous=None, elapsed=0, window=10, discount=0.8):
    """Select one whole sample, preserving each sample's gripper actions.

    Score = positive-sample consensus + backward overlap error, equal weights.
    This is a controlled adaptation of BID, not a full BID implementation.
    """
    x = samples.detach().to(device="cpu", dtype=torch.float64)
    n = x.shape[0]
    front = x[:, :window, :6]
    distances = torch.linalg.vector_norm(front[:, None] - front[None, :], dim=-1).mean(-1)
    consensus = distances.sum(1) / max(n - 1, 1)
    backward = torch.zeros(n, dtype=torch.float64)
    if previous is not None and elapsed < previous.shape[0]:
        reference = previous.detach().to(device="cpu", dtype=torch.float64)[elapsed:elapsed + window, :6]
        k = len(reference)
        weights = discount ** torch.arange(k, dtype=torch.float64)
        backward = (torch.linalg.vector_norm(x[:, :k, :6] - reference, dim=-1) * weights).sum(1) / weights.sum()
    scores = consensus + backward
    return int(scores.argmin()), consensus.tolist(), backward.tolist()


def install_controller(policy_cls, action_key, *, method, trace, gripper_boundary, seed):
    if method not in METHODS:
        raise ValueError(method)
    config = METHODS[method]
    path = Path(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    original_select, original_reset = policy_cls.select_action, policy_cls.reset

    def record(row):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        self._recent_rollout = getattr(self, "_recent_rollout", -1) + 1
        self._recent_t = 0
        self._recent_chunks = 0
        self._recent_previous = None
        self._recent_previous_start = 0
        self._recent_last_action = None
        record(dict(event="reset", rollout=self._recent_rollout))
        return result

    @torch.no_grad()
    def select(self, batch, *args, **kwargs):
        if self.config.chunk_size != 50 or self.config.n_action_steps != 50:
            raise RuntimeError("requires chunk_size=n_action_steps=50")
        if getattr(self.config, "rtc_config", None) is not None and self.config.rtc_config.enabled:
            raise RuntimeError("RTC unsupported")
        is_new = len(self._queues[action_key]) == 0
        if is_new and self.config.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_select(self, batch, *args, **kwargs)
        if is_new:
            first = torch.stack([result] + list(self._queues[action_key]), dim=1)
            if first.shape != (1, 50, 7):
                raise RuntimeError(f"unexpected chunk {first.shape}")
            candidates = [first]
            # Auxiliary draws are isolated: N=4 first-sample fixed10 must retain
            # exactly the same primary RNG stream as N=1 fixed10.
            if config["samples"] > 1:
                devices = [first.device.index or 0] if first.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    aux_seed = seed + 100003 * self._recent_rollout + 101 * self._recent_chunks
                    torch.manual_seed(aux_seed)
                    for _ in range(config["samples"] - 1):
                        candidates.append(self.predict_action_chunk(batch))
            samples = torch.cat(candidates, 0)
            if first.is_cuda:
                torch.cuda.synchronize()
            sampling_s = time.perf_counter() - start
            elapsed = self._recent_t - self._recent_previous_start
            chosen, consensus, backward = rank_candidates(samples, self._recent_previous, elapsed)
            if not config["rerank"]:
                chosen = 0
            horizon = 10
            entropy_min = entropy_max = gripper_mean = 0.0
            without_gripper = 10
            if config["adaptive"]:
                continuous, grip = action_entropy(samples, gripper_boundary, include_gripper=False)
                without_gripper = entropy_horizon(continuous)
                curve = continuous if config.get("no_gripper", False) else continuous + grip
                horizon = entropy_horizon(curve)
                entropy_min, entropy_max = float(curve.min()), float(curve.max())
                gripper_mean = float(grip.mean())
            selected = samples[chosen]
            queue = self._queues[action_key]
            queue.clear()
            queue.extend(selected[1:horizon].unsqueeze(1))
            result = selected[0].unsqueeze(0)
            seam = 0.0 if self._recent_last_action is None else float(torch.linalg.vector_norm(result[0, :6] - self._recent_last_action[:6]))
            self._recent_previous = selected.clone()
            self._recent_previous_start = self._recent_t
            self._recent_chunks += 1
            record(dict(event="chunk", method=method, rollout=self._recent_rollout,
                        step=self._recent_t, horizon=horizon, horizon_without_gripper=without_gripper,
                        model_calls=config["samples"], candidate=chosen,
                        consensus=consensus, backward=backward, seam_norm=seam,
                        entropy_min=entropy_min, entropy_max=entropy_max,
                        gripper_entropy_mean=gripper_mean, sampling_s=sampling_s,
                        peak_cuda_mb=torch.cuda.max_memory_allocated() / 2**20 if first.is_cuda else 0))
        self._recent_last_action = result[0].clone()
        self._recent_t += 1
        return result

    policy_cls.select_action, policy_cls.reset = select, reset
    record(dict(event="configuration", method=method, gripper_boundary=gripper_boundary,
                seed=seed, **config))
