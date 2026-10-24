import json
import pytest
from state_audit import install_state_audit


def test_state_offset_and_actual_index_recording(tmp_path):
    class Fake:
        def __init__(self):
            self.init_states=True
            self._init_states=list(range(50))
            self.init_state_id=0
        def reset(self,seed=None):
            state=self._init_states[self.init_state_id]
            self.init_state_id+=1
            return state
    path=tmp_path/"states.jsonl"
    install_state_audit(offset=10,trace=path,env_class=Fake)
    env=Fake()
    assert env.reset(seed=4000)==10
    assert env.reset(seed=4001)==11
    assert [json.loads(line)["init_state_index"] for line in path.read_text().splitlines()]==[10,11]


def test_reject_invalid_offset(tmp_path):
    with pytest.raises(ValueError):
        install_state_audit(offset=-1,trace=tmp_path/"states.jsonl",env_class=object)
