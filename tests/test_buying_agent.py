#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from Models.Common_agents import BuyingAgent

def test_unit_counting():
    """Test the unit counting functionality."""
    
    # Create a simple mock observation for testing
    # This is a simplified version - in reality the observation is much more complex
    observation = np.zeros((184, 4, 7))
    
    # Test the buying agent
    agent = BuyingAgent(["yasuo", "fiora", "vayne", "nidalee", "garen"])
    
    print("Testing BuyingAgent unit counting...")
    
    # This will likely fail with the mock observation, but let's see what happens
    try:
        unit_counts = agent.get_unit_counts(observation)
        print(f"Unit counts: {unit_counts}")
        
        units_needed = agent.count_units_needed_for_three_star(unit_counts)
        print(f"Units needed for 3-star: {units_needed}")
        
        is_full = agent.is_board_and_bench_full(observation)
        print(f"Board and bench full: {is_full}")
        
        print("Basic functionality test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        print("This is expected with a mock observation - the agent needs real TFT data")

def test_unit_priority_logic():
    """Test the priority logic for selling units."""
    
    # Test priority calculation
    unit_counts = {
        "yasuo": 8,  # Needs 1 more for 3-star (high priority to keep)
        "fiora": 5,  # Needs 4 more for 3-star (medium priority)
        "vayne": 2,  # Needs 7 more for 3-star (low priority)
        "nidalee": 1  # Needs 8 more for 3-star (lowest priority)
    }
    
    agent = BuyingAgent(["yasuo", "fiora", "vayne", "nidalee", "garen"])
    units_needed = agent.count_units_needed_for_three_star(unit_counts)
    
    print("\nTesting priority logic:")
    print(f"Unit counts: {unit_counts}")
    print(f"Units needed for 3-star: {units_needed}")
    
    # Units should be prioritized by how few they need (yasuo highest priority, nidalee lowest)
    expected_order = ["yasuo", "fiora", "vayne", "nidalee"]
    actual_order = sorted(units_needed.keys(), key=lambda x: units_needed[x])
    
    print(f"Expected priority order (highest to lowest): {expected_order}")
    print(f"Actual priority order: {actual_order}")
    
    if expected_order == actual_order:
        print("✓ Priority logic working correctly!")
    else:
        print("✗ Priority logic needs adjustment")

if __name__ == "__main__":
    test_unit_counting()
    test_unit_priority_logic()