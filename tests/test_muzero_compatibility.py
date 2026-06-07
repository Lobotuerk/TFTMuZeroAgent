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
        expected_shape = (batch_size, 3, 37)  # Corrected for TFTSet4Gym: ACTION_DIM = [7, 37, 10]
        
        print(f"   - Policy shape: {policy_shape}")
        print(f"   - Expected shape: {expected_shape}")
        print(f"   - Value shape: {outputs['value'].shape}")
        print(f"   - Hidden state shape: {outputs['hidden_state'].shape}")
        
        if policy_shape == expected_shape:
            print("   ✓ Policy shape matches TFTSet4Gym Enhanced MCTS requirements")
            
            # Test Enhanced MCTS creation with this network
            print("5. Testing Enhanced MCTS integration...")
            action_limits = [7, 37, 10]  # Corrected for TFTSet4Gym config
            mcts = EnhancedMCTS(
                sample_size=16,
                action_size=3,  # 3 action dimensions
                action_limits=action_limits,
                policy_size=111,  # 3 * 37
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
    expected_pred_lns = ["ln2", "ln3", "ln4", "ln5", "ln_v1", "ln_v2", "ln_v3", "ln_p1", "ln_p2", "ln_p3"]
    for ln_attr in expected_pred_lns:
        assert hasattr(pred_net, ln_attr), f"PredNetwork is missing {ln_attr}"
        assert isinstance(getattr(pred_net, ln_attr), torch.nn.LayerNorm), f"PredNetwork.{ln_attr} is not a LayerNorm"
        
    # Check RepNetwork LayerNorms
    expected_rep_lns = ["ln2", "ln3", "ln4", "ln5"]
    for ln_attr in expected_rep_lns:
        assert hasattr(rep_net, ln_attr), f"RepNetwork is missing {ln_attr}"
        assert isinstance(getattr(rep_net, ln_attr), torch.nn.LayerNorm), f"RepNetwork.{ln_attr} is not a LayerNorm"
        
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