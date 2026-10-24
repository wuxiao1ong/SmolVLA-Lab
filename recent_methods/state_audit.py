"""Explicit initial-state offset; record actual state indices used by workers."""
import json
from pathlib import Path


def install_state_audit(*, offset, trace, env_class=None):
    if offset < 0:
        raise ValueError("negative initial-state offset")
    if env_class is None:
        from lerobot.envs.libero import LiberoEnv
        env_class = LiberoEnv
    path = Path(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    original_init, original_reset = env_class.__init__, env_class.reset

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not self.init_states or self._init_states is None:
            raise RuntimeError("state audit requires fixed init states")
        if offset >= len(self._init_states):
            raise ValueError("offset exceeds available initial states")
        self.init_state_id += offset

    def reset(self, seed=None, **kwargs):
        index = int(self.init_state_id % len(self._init_states))
        result = original_reset(self, seed=seed, **kwargs)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(init_state_index=index, seed=int(seed) if seed is not None else None)) + "\n")
        return result

    env_class.__init__, env_class.reset = init, reset
