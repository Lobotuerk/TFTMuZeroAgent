"""
Example usage of the Enhanced Agent Interface from outside the Models directory.

This demonstrates how to properly import and use the enhanced agent system
when the file is located in the Models directory.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Example of using the enhanced agent interface from the root directory"""
    
    try:
        # Import the enhanced interface from Models
        from Models.enhanced_agent_interface import (
            create_enhanced_setup,
            create_custom_agent_setup,
            AsyncGameEnvironment
        )
        
        # Import required modules
        from Models.MuZero_torch_agent import MuZeroAgent
        from Models.Common_agents import RandomAgent, CultistAgent, DivineAgent
        import config
        
        print("✅ Successfully imported enhanced agent interface!")
        
        # Example 1: Default setup
        print("\n=== Creating Default Setup ===")
        agent_manager, batch_processor = create_enhanced_setup()
        print(f"Created agent manager with {len(agent_manager.agents)} agent types")
        
        # Example 2: Custom setup
        print("\n=== Creating Custom Setup ===")
        custom_agents = [
            (RandomAgent("CustomRandom1"), 3),
            (CultistAgent(), 2),
            (DivineAgent(), 3)
        ]
        
        custom_manager, custom_processor = create_custom_agent_setup(
            agents_and_counts=custom_agents,
            max_batch_size=8,
            batch_timeout_ms=10.0,
            gpu_memory_fraction=0.6
        )
        print(f"Created custom manager with {len(custom_manager.agents)} agent types")
        print(f"Player to agent mapping: {len(custom_manager.player_to_agent)} players")
        
        # Example 3: Show agent configurations
        print("\n=== Agent Configurations ===")
        for player_id, agent_type in custom_manager.player_to_agent.items():
            print(f"{player_id} -> {agent_type.__name__}")
        
        print("\n✅ Enhanced agent interface is working correctly!")
        print("\nTo use in your training loop:")
        print("1. Import: from Models.enhanced_agent_interface import create_enhanced_setup")
        print("2. Setup: manager, processor = create_enhanced_setup(your_agent_configs)")
        print("3. Use: actions = await manager.get_actions(observations, rewards, terminated)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available:")
        print("- Models/enhanced_agent_interface.py")
        print("- Models/MuZero_torch_agent.py") 
        print("- Models/Common_agents.py")
        print("- config.py")
        print("- torch, numpy, asyncio")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready to use enhanced agent interface!")
    else:
        print("\n💥 Please fix the issues above before proceeding.")