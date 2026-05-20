import numpy as np
from crisp.memory import MemoryBank
from crisp.memory_policies import HerdingMemoryPolicy


def test_herding_limit():
    memory = MemoryBank()
    for i in range(10):
        vector = np.array([1.0, i / 100.0], dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        memory.add(vector, label="A")

    policy = HerdingMemoryPolicy(max_exemplars_per_class=3)
    policy.apply(memory)

    assert len(memory) == 3
