import sys
import os
import numpy as np
import torch
import pytest

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import config
from Models.MuZero_torch_model import MuZeroNetwork, BOARD_DIM, BENCH_CHAMP_DIM, BENCH_ITEM_DIM, SHOP_CHAMP_DIM, PER_SLOT_DIM

def test_unnormalized_embedding_indexing():
    """Verify that observations containing unnormalized indexing integer values do not cause out of bounds errors on model forward pass."""
    model = MuZeroNetwork()
    
    # Create an observation with large but valid indices
    obs = np.zeros((1, config.OBSERVATION_SIZE), dtype=np.float32)
    
    # Set unnormalized indices in slot 0 (Board slot 0 is indices 0:122)
    # Champion index = 55.0
    obs[0, 0:32] = 55.0
    # Item 0 index = 35.0 (at 32:56)
    obs[0, 32:56] = 35.0
    # Item 1 index = 10.0 (at 56:80)
    obs[0, 56:80] = 10.0
    # Item 2 index = 2.0 (at 80:104)
    obs[0, 80:104] = 2.0
    # Trait index = 18.0 (at 104:112)
    obs[0, 104:112] = 18.0
    # Origin index = 8.0 (at 112:120)
    obs[0, 112:120] = 8.0
    # Star level = 3.0 (at 120)
    obs[0, 120] = 3.0
    # Chosen = 1.0 (at 121)
    obs[0, 121] = 1.0
    
    # Also set some indices on bench items
    # Bench items start at BOARD_DIM + BENCH_CHAMP_DIM = 3416 + 1098 = 4514
    bench_items_start = BOARD_DIM + BENCH_CHAMP_DIM
    obs[0, bench_items_start : bench_items_start + 24] = 36.0 # Max item index
    
    # Also set some indices on shop champions
    # Shop starts at BOARD_DIM + BENCH_CHAMP_DIM + BENCH_ITEM_DIM = 4514 + 240 = 4754
    shop_start = bench_items_start + BENCH_ITEM_DIM
    obs[0, shop_start : shop_start + 32] = 57.0 # Max champion index
    
    # Test initial_inference with CUDA or CPU
    # We should run it on CUDA if available to verify PyTorch ATen CUDA assertions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    obs_tensor = torch.from_numpy(obs).float().to(device)
    
    try:
        outputs = model.initial_inference(obs_tensor)
        assert outputs["hidden_state"] is not None
        print("✓ Successfully passed unnormalized integer IDs through network without indexSelect errors!")
    except Exception as e:
        pytest.fail(f"Failed to run model initial_inference with unnormalized IDs: {e}")

if __name__ == "__main__":
    test_unnormalized_embedding_indexing()
