from collections import deque
from types import SimpleNamespace
import json

import pytest
import torch

from recent_controller import action_entropy, entropy_horizon, install_controller, rank_candidates


def test_gaussian_entropy_tracks_candidate_spread():
    torch.manual_seed(5)
    samples = torch.randn(8, 50, 7)
    low, _ = action_entropy(samples, include_gripper=False)
    high, _ = action_entropy(samples * 2, include_gripper=False)
    assert torch.all(high > low)
    with pytest.raises(ValueError):
        action_entropy(samples[:2])


def test_gripper_boundary_is_normalized_physical_zero():
    samples = torch.zeros(4, 50, 7)
    samples[:2, :, 6] = 0.1
    samples[2:, :, 6] = 0.3
    continuous, _ = action_entropy(samples, 0.2, include_gripper=False)
    combined, grip = action_entropy(samples, 0.2)
    assert torch.allclose(grip, torch.full_like(grip, torch.log(torch.tensor(2.0)).item()))
    assert torch.allclose(combined - continuous, grip)
    _, wrong_boundary = action_entropy(samples, 0.0)
    assert torch.all(wrong_boundary < 1e-8)


def test_entropy_horizon_detects_uncertainty_onset():
    curve = torch.zeros(50, dtype=torch.float64)
    curve[10:] = 10
    assert entropy_horizon(curve) == 10
    curve = torch.zeros(50, dtype=torch.float64)
    curve[40:] = 10
    assert entropy_horizon(curve) == 25


def test_backward_coherence_changes_whole_candidate_choice():
    samples = torch.arange(4).reshape(4, 1, 1).expand(4, 50, 7).float()
    medoid, _, _ = rank_candidates(samples)
    coherent, _, _ = rank_candidates(samples, previous=samples[3], elapsed=10)
    assert medoid == 1
    assert coherent == 3
    assert samples[coherent, 0, 6] == 3  # gripper is not averaged


def test_backward_overlap_is_aligned_to_elapsed_actions():
    ramp = torch.arange(50).reshape(50, 1).expand(50, 7).float()
    samples = torch.stack([ramp + offset for offset in (0, 10, 20, 30)])
    choice, _, backward = rank_candidates(samples, previous=ramp, elapsed=10)
    assert choice == 1
    assert backward[1] == 0 and backward[0] > 20


def make_policy(method, trace):
    class Fake:
        def __init__(self):
            self.config = SimpleNamespace(chunk_size=50, n_action_steps=50, rtc_config=None, device="cpu")
            self.reset()

        def reset(self):
            self._queues = {"action": deque(maxlen=50)}

        def predict_action_chunk(self, batch):
            return torch.randn(1, 50, 7)

        def select_action(self, batch):
            if not self._queues["action"]:
                self._queues["action"].extend(self.predict_action_chunk(batch).transpose(0, 1))
            return self._queues["action"].popleft()

    install_controller(Fake, "action", method=method, trace=str(trace), gripper_boundary=0.05, seed=3000)
    return Fake()


def test_compute_control_preserves_primary_rng_and_queue(tmp_path):
    results = []
    for method in ("base10", "sample4_fixed10"):
        torch.manual_seed(42)
        policy = make_policy(method, tmp_path / f"{method}.jsonl")
        results.append(torch.stack([policy.select_action({}) for _ in range(23)]))
        assert len(policy._queues["action"]) == 7
    assert torch.equal(*results)
    events = [json.loads(line) for line in (tmp_path / "sample4_fixed10.jsonl").read_text().splitlines()]
    chunks = [row for row in events if row["event"] == "chunk"]
    assert len(chunks) == 3
    assert sum(row["model_calls"] for row in chunks) == 12


def test_adaptive_queue_and_reset(tmp_path):
    policy = make_policy("aac_rerank", tmp_path / "aac.jsonl")
    for _ in range(30):
        policy.select_action({})
    assert 0 <= len(policy._queues["action"]) <= 24
    policy.reset()
    assert policy._recent_t == 0 and policy._recent_previous is None
    assert not policy._queues["action"]
    events = [json.loads(line) for line in (tmp_path / "aac.jsonl").read_text().splitlines()]
    chunks = [row for row in events if row["event"] == "chunk"]
    assert chunks and all(5 <= row["horizon"] <= 25 for row in chunks)
