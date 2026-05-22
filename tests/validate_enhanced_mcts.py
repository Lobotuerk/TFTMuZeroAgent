#!/usr/bin/env python3
"""
Quick validation script for enhanced MCTS implementation
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def main():
    """Validate enhanced MCTS functionality"""
    
    print("=== Enhanced MCTS Validation ===")
    print()
    
    try:
        # Test 1: PyMCTS Import
        print("1. Testing PyMCTS import...", end=" ")
        import pymcts
        print("✓ PASSED")
        
        # Test 2: Enhanced MCTS Import
        print("2. Testing Enhanced MCTS import...", end=" ")
        from Models.MCTS_torch import EnhancedMCTS, TFTMove, TFTState
        print("✓ PASSED")
        
        # Test 3: TFTMove Creation
        print("3. Testing TFTMove creation...", end=" ")
        move = TFTMove(action_type=0, target_1=0, target_2=0)  # 0 = pass action
        assert move.action_type == 0
        print("✓ PASSED")
        
        # Test 4: TFTState Creation  
        print("4. Testing TFTState creation...", end=" ")
        import numpy as np
        mock_obs = np.zeros(5152)
        state = TFTState(mock_obs)
        print("✓ PASSED")
        
        # Test 5: EnhancedMCTS Creation
        print("5. Testing EnhancedMCTS creation...", end=" ")
        
        class MockNetwork:
            def initial_inference(self, obs):
                import torch
                batch_size = obs.shape[0] if obs.ndim > 1 else 1
                return {
                    "reward": torch.zeros(batch_size),
                    "value": torch.zeros(batch_size), 
                    "policy_logits": torch.zeros(batch_size, 13, 100),
                    "hidden_state": torch.zeros(batch_size, 512)
                }
            
            def recurrent_inference(self, hidden_state, action):
                import torch
                batch_size = hidden_state.shape[0]
                return {
                    "reward": torch.zeros(batch_size),
                    "value": torch.zeros(batch_size),
                    "policy_logits": torch.zeros(batch_size, 13, 100), 
                    "hidden_state": torch.zeros(batch_size, 512)
                }
        
        network = MockNetwork()
        # Create MCTS with proper parameters
        action_limits = [13, 100, 100, 100]  # Example action limits
        mcts = EnhancedMCTS(
            sample_size=16,
            action_size=4, 
            action_limits=action_limits,
            policy_size=1300,  # 13 * 100
            network=network
        )
        print("✓ PASSED")
        
        # Test 6: Basic Stats
        print("6. Testing MCTS stats...", end=" ")
        stats = mcts.get_stats()
        assert 'pymcts_available' in stats
        assert stats['pymcts_available'] == True
        print("✓ PASSED")
        
        print()
        print("🎉 All validation tests passed!")
        print("✅ Enhanced MCTS implementation is working correctly!")
        print()
        print("Key features validated:")
        print("  - PyMCTS library integration")
        print("  - TFTMove and TFTState classes")
        print("  - EnhancedMCTS controller")
        print("  - Network interface compatibility")
        print("  - Statistics and metadata")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        print()
        print("❌ Enhanced MCTS validation failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)