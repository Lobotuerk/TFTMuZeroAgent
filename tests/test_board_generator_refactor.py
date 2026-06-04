import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Models.MuZero_torch_model import BoardGenerator, encode_champion_availability_torch, NUM_CLASSES, BOARD_HEIGHT, BOARD_WIDTH

def test_board_generator():
    ngf = 128
    bg = BoardGenerator(input_dim=116)
    bg.eval()

    batch_size = 2
    avail = torch.randn(batch_size, 116)
    with torch.no_grad():
        output = bg(avail)
    assert output.shape == (batch_size, NUM_CLASSES, BOARD_HEIGHT, BOARD_WIDTH), \
        f"Expected ({batch_size}, {NUM_CLASSES}, {BOARD_HEIGHT}, {BOARD_WIDTH}), got {output.shape}"
    print(f"PASS: Output shape correct: {output.shape}")

    batch_size = 2
    full_obs = torch.randn(batch_size, 5152)
    avail = encode_champion_availability_torch(full_obs)
    assert avail.shape == (batch_size, 116), f"Expected (2, 116), got {avail.shape}"
    with torch.no_grad():
        output = bg(avail)
    assert output.shape == (batch_size, NUM_CLASSES, BOARD_HEIGHT, BOARD_WIDTH), \
        f"Expected ({batch_size}, {NUM_CLASSES}, {BOARD_HEIGHT}, {BOARD_WIDTH}), got {output.shape}"
    print(f"PASS: Observation encoding produces correct output shape")

    with torch.no_grad():
        out1 = bg(encode_champion_availability_torch(full_obs))
        obs2 = full_obs.clone()
        obs2[:, 3304:] = 999.0
        out2 = bg(encode_champion_availability_torch(obs2))
    assert torch.allclose(out1, out2, atol=1e-6), "Outputs should be identical when only non-board features differ"
    print("PASS: Slicing works correctly (features after index 3304 are ignored)")

    print("\nAll tests passed!")
    return True

if __name__ == "__main__":
    test_board_generator()
