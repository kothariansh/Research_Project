import torch
from problems.tsp.state_tsp import StateTSP

# Create a small dummy batch: batch_size=4, 5 locations, 2-D coords
loc = torch.randn(4, 5, 2)
state = StateTSP.initialize(loc)

# Try a few different index types:
for key in [0, [1, 3], slice(2, 4), torch.tensor([0, 2, 3], dtype=torch.long)]:
    try:
        sliced = state[key]
        print(f"state[{key!r}] ➜ ids.shape = {sliced.ids.shape}")
    except Exception as e:
        print(f"state[{key!r}] raised {type(e).__name__}: {e}")
