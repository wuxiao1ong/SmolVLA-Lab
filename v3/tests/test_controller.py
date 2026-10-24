from collections import deque
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from adaptive_controller import install_controller, select_horizon, volatility_score


def test_score():
    chunk = torch.zeros(1, 25, 7)
    assert volatility_score(chunk, 25) == 0.0
    chunk[0, :, 0] = torch.arange(25)
    assert abs(volatility_score(chunk, 25) - 1.0) < 1e-6


def test_selection():
    assert select_horizon("adaptive", .21, 5, 25, .2) == 5
    assert select_horizon("adaptive", .19, 5, 25, .2) == 25
    assert select_horizon("fixed", .21, 5, 10, .2) == 10


def test_queue_trim(tmp_path):
    class Fake:
        def __init__(self):
            self.config = type("cfg", (), {"n_action_steps":50,"chunk_size":50, "rtc_config":None})()
            self.reset()
            self.calls=0
        def reset(self):
            self._queues={"action":deque(maxlen=50)}
        def select_action(self, batch, noise=None):
            if not self._queues["action"]:
                self.calls+=1
                chunk=torch.arange(50).float().reshape(50,1).repeat(1,7)
                self._queues["action"].extend(chunk.unsqueeze(1))
            return self._queues["action"].popleft()
    install_controller(Fake, "action", mode="fixed", short=5, long=10,
                       threshold=0.0, trace=str(tmp_path/"trace.jsonl"))
    policy=Fake()
    for _ in range(22):
        policy.select_action({})
    assert policy.calls==3
    assert len(policy._queues["action"])==8
    assert len((tmp_path/"trace.jsonl").read_text().splitlines()) >= 4
