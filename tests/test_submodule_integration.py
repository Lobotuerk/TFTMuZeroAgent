#!/usr/bin/env python3
"""
Quick test to verify that the TFTSet4Gym package integration works correctly.
"""

try:
    from tft_set4_gym.tft_simulator import parallel_env
    print("✅ Successfully imported parallel_env from installed package")
    
    # Test creating an environment
    env = parallel_env()
    print("✅ Successfully created TFT environment")
    
    # Test reset
    observations, infos = env.reset()
    print(f"✅ Environment reset successful. Agents: {list(observations.keys())}")
    
    print("\n🎉 Package integration test PASSED!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the TFTSet4Gym package is installed: pip install -e TFTSet4Gym/")
    exit(1)
except Exception as e:
    print(f"❌ Runtime error: {e}")
    exit(1)