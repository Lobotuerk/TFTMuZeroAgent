"""Test script to verify BoardGenerator refactoring."""
import sys
sys.path.insert(0, '.')

import torch
from Models.MuZero_torch_model import BoardGenerator

def test_board_generator():
    """Test BoardGenerator with various input sizes."""
    ngf = 128
    bg = BoardGenerator(ngf)
    bg.eval()
    
    # Test with full observation size (5152) - should slice to 3304
    batch_size = 2
    full_obs = torch.randn(batch_size, 5152)
    with torch.no_grad():
        output = bg(full_obs)
    print(f"Input shape (full obs): {full_obs.shape}")
    print(f"Output shape (full obs): {output.shape}")
    assert output.shape == (batch_size, 58, 4, 7), f"Expected (2, 58, 4, 7), got {output.shape}"
    print("PASS: Full observation input produces correct output shape")
    
    # Test with exactly 3304 features
    partial_obs = torch.randn(batch_size, 3304)
    with torch.no_grad():
        output = bg(partial_obs)
    print(f"Input shape (3304 features): {partial_obs.shape}")
    print(f"Output shape (3304 features): {output.shape}")
    assert output.shape == (batch_size, 58, 4, 7), f"Expected (2, 58, 4, 7), got {output.shape}"
    print("PASS: 3304-feature input produces correct output shape")
    
    # Test that first ConvTranspose2d expects 3304 input channels
    first_conv = bg.main[0]
    assert first_conv.in_channels == 3304, f"Expected in_channels=3304, got {first_conv.in_channels}"
    print("PASS: First ConvTranspose2d has correct input channels (3304)")
    
    # Test slicing behavior: inputs that differ only after index 3304 should produce same output
    obs1 = torch.randn(1, 5152)
    obs2 = obs1.clone()
    obs2[:, 3304:] = 999.0  # Modify everything after 3304
    with torch.no_grad():
        out1 = bg(obs1)
        out2 = bg(obs2)
    assert torch.allclose(out1, out2, atol=1e-6), "Outputs should be identical when only non-board features differ"
    print("PASS: Slicing works correctly (features after index 3304 are ignored)")
    
    print("\nAll tests passed!")
    return True

if __name__ == "__main__":
    test_board_generator()
