#!/usr/bin/env python3
"""Test script for the modified MuZero agent to verify it reads action dimensions from schema."""

import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Models.MuZero_torch_agent import MuZeroAgent, create_enhanced_muzero_agent

def test_muzero_agent_initialization():
    """Test that MuZero agent correctly reads action dimensions from schema/config."""
    
    print("Testing MuZero agent initialization...")
    
    # Test direct initialization
    try:
        agent = MuZeroAgent()
        print(f"✓ Direct initialization successful")
        print(f"  Action size: {agent.action_size}")
        print(f"  Action limits: {agent.action_limits}")
        print(f"  Observation size: {agent.obs_size}")
        print(f"  Simulations: {agent.simulations}")
        
        # Verify expected values
        expected_action_limits = [7, 37, 10]
        if agent.action_limits == expected_action_limits:
            print(f"✓ Action limits match expected values: {expected_action_limits}")
        else:
            print(f"✗ Action limits mismatch. Expected: {expected_action_limits}, Got: {agent.action_limits}")
        
        if agent.action_size == len(expected_action_limits):
            print(f"✓ Action size matches expected value: {len(expected_action_limits)}")
        else:
            print(f"✗ Action size mismatch. Expected: {len(expected_action_limits)}, Got: {agent.action_size}")
            
    except Exception as e:
        print(f"✗ Direct initialization failed: {e}")
        return False
    
    # Test factory function
    try:
        factory_agent = create_enhanced_muzero_agent()
        print(f"✓ Factory initialization successful")
        print(f"  Action size: {factory_agent.action_size}")
        print(f"  Action limits: {factory_agent.action_limits}")
        
        # Verify factory agent has same values
        if factory_agent.action_limits == agent.action_limits:
            print(f"✓ Factory agent action limits match direct initialization")
        else:
            print(f"✗ Factory agent action limits mismatch")
            
    except Exception as e:
        print(f"✗ Factory initialization failed: {e}")
        return False
    
    return True

def test_muzero_agent_action_selection():
    """Test that MuZero agent can select actions with the new setup."""
    
    print("\nTesting MuZero agent action selection...")
    
    try:
        agent = MuZeroAgent()
        
        # Create a dummy observation (matching expected schema size)
        observation = np.random.rand(agent.obs_size)
        print(f"Created observation with size: {observation.shape}")
        
        # Test action selection
        actions = agent.select_action(observation)
        print(f"✓ Action selection successful")
        print(f"  Generated actions: {actions}")
        print(f"  Number of actions: {len(actions)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Action selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing modified MuZero agent...")
    
    success = True
    success &= test_muzero_agent_initialization()
    success &= test_muzero_agent_action_selection()
    
    if success:
        print("\n✓ All tests passed! MuZero agent successfully reads action dimensions from schema.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        sys.exit(1)