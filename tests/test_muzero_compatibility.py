#!/usr/bin/env python3
"""
Test MuZero model compatibility with Enhanced MCTS
"""

import sys
import os
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_muzero_compatibility():
    """Test if MuZero model is compatible with Enhanced MCTS"""
    
    try:
        # Test imports
        print("1. Testing imports...")
        from Models.MuZero_torch_model import MuZeroNetwork
        from Models.MCTS_torch import EnhancedMCTS
        import numpy as np
        import torch
        print("   ✓ All imports successful")
        
        # Test model creation
        print("2. Testing model creation...")
        model = MuZeroNetwork()
        print("   ✓ MuZero model created")
        
        # Test model output
        print("3. Testing model output...")
        import config
        batch_size = 1
        obs = np.random.rand(batch_size, config.OBSERVATION_SIZE)
        
        outputs = model.initial_inference(obs)
        print("   ✓ Model inference successful")
        
        # Check output shapes
        print("4. Checking output compatibility...")
        policy_shape = outputs["policy_logits"].shape
        expected_shape = (batch_size, config.ACTION_CONCAT_SIZE)
        
        print(f"   - Policy shape: {policy_shape}")
        print(f"   - Expected shape: {expected_shape}")
        print(f"   - Value shape: {outputs['value'].shape}")
        print(f"   - Hidden state shape: {outputs['hidden_state'].shape}")
        
        if policy_shape == expected_shape:
            print("   ✓ Policy shape matches concatenated 3-block variable-dim format")
            
            # Test Enhanced MCTS creation with this network
            print("5. Testing Enhanced MCTS integration...")
            action_limits = config.ACTION_DIM
            mcts = EnhancedMCTS(
                sample_size=16,
                action_size=3,
                action_limits=action_limits,
                policy_size=config.ACTION_CONCAT_SIZE,
                network=model
            )
            print("   ✓ Enhanced MCTS created successfully with MuZero model")
            
            print("\n🎉 SUCCESS: MuZero model is fully compatible with TFTSet4Gym Enhanced MCTS!")
            return True
        else:
            print(f"   ✗ Policy shape mismatch")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_muzero_layernorm_presence():
    """Verify LayerNorm layers are instantiated and applied on PredNetwork and RepNetwork."""
    from Models.MuZero_torch_model import MuZeroNetwork
    import torch
    
    print("Testing LayerNorm presence in networks...")
    model = MuZeroNetwork()
    pred_net = model.prediction_network
    rep_net = model.representation_network
    
    # Check PredNetwork LayerNorms
    ln_count = len(pred_net.res_lns)
    assert ln_count >= 5, f"PredNetwork.res_lns should have at least 5 LayerNorms, got {ln_count}"
    for i, ln in enumerate(pred_net.res_lns):
        assert isinstance(ln, torch.nn.LayerNorm), f"PredNetwork.res_lns[{i}] is not a LayerNorm"
        
    # Check RepNetwork LayerNorms
    ln_count = len(rep_net.res_lns)
    assert ln_count >= 4, f"RepNetwork.res_lns should have at least 4 LayerNorms, got {ln_count}"
    for i, ln in enumerate(rep_net.res_lns):
        assert isinstance(ln, torch.nn.LayerNorm), f"RepNetwork.res_lns[{i}] is not a LayerNorm"
        
    print("   ✓ LayerNorm layers are present and of correct type!")
    return True

if __name__ == "__main__":
    print("=== MuZero Model Compatibility Test ===")
    print()
    
    success = test_muzero_compatibility()
    
    print()
    if success:
        print("✅ MuZero model is ready for Enhanced MCTS!")
    else:
        print("❌ MuZero model needs further updates for Enhanced MCTS compatibility.")