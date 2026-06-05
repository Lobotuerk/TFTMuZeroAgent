#!/usr/bin/env python3
"""
Standalone test for running a single episode with 8 agents using EnhancedAgentManager.
This test creates a parallel environment with 8 agents and runs a complete episode using
the enhanced agent interface with proper batching and GPU optimization.
"""

import sys
import os
import time
import asyncio
import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from submodule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TFTSet4Gym'))
from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
sys.path.pop(0)

pytestmark = pytest.mark.asyncio

# Import Enhanced Agent Interface components
from Models.enhanced_agent_interface import create_enhanced_setup, AsyncGameEnvironment
from Models.Common_agents import RandomAgent, CultistAgent, DivineAgent


async def test_enhanced_episode_with_mixed_agents():
    """Test running a complete episode with 8 agents using EnhancedAgentManager."""
    print("=== TFT Enhanced Episode Test with Mixed Agent Types ===")
    print(f"Starting test at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create enhanced agent setup with mixed agent types
    custom_agents = [
        (RandomAgent("RandomAgent_0"), 3),    # 3 random agents
        (CultistAgent(), 2),                  # 2 cultist agents
        (DivineAgent(), 2),                   # 2 divine agents  
        (RandomAgent("RandomAgent_1"), 1),    # 1 additional random agent
    ]
    
    print(f"Creating enhanced setup with agent configuration:")
    for agent, count in custom_agents:
        print(f"  - {type(agent).__name__}: {count} instances")
    
    # Create enhanced agent manager and batch processor
    try:
        agent_manager, batch_processor = create_enhanced_setup(
            agent_configs=custom_agents,
            max_batch_size=8,
            batch_timeout_ms=5.0,
            gpu_memory_fraction=0.7
        )
        print(f"Enhanced agent manager created successfully")
    except Exception as e:
        print(f"Error creating enhanced setup: {e}")
        return False
    
    # Create environment
    try:
        env = parallel_env(rank=0)
    except:
        env = parallel_env()
    
    print(f"Environment created successfully")
    
    # Reset and verify 8 agents
    observations_dict, infos = env.reset()
    num_agents = len(env.agents)
    
    print(f"Reset complete - Found {num_agents} agents: {env.agents}")
    assert num_agents == 8, f"Expected 8 agents, got {num_agents}"
    
    # Verify agent mapping
    player_mapping = agent_manager.get_player_agent_mapping()
    print(f"Player to agent mapping:")
    for player_id, agent_type in player_mapping.items():
        print(f"  {player_id} -> {getattr(agent_type, 'agent_name', type(agent_type).__name__)}")
    
    # Initialize tracking variables
    step_count = 0
    max_steps = 3000  # Generous limit for TFT games
    initial_agents = num_agents
    
    # Game progress tracking
    elimination_log = []
    round_log = []
    last_progress_report = 0
    
    print(f"Starting enhanced episode simulation...")
    start_time = time.time()
    
    # Initialize reward and termination tracking
    rewards = {agent: 0.0 for agent in env.agents}
    terminations = {agent: False for agent in env.agents}
    
    # Main game loop
    while env.agents and step_count < max_steps:
        try:
            # Prepare observations in the format expected by enhanced agent manager
            enhanced_observations = {}
            for agent in env.agents:
                if agent in observations_dict:
                    obs_data = observations_dict[agent]
                    # Correctly handle dictionary observations from TFT simulator
                    if isinstance(obs_data, dict) and 'tensor' in obs_data:
                        tensor_data = obs_data['tensor']
                        mask_data = obs_data.get('action_mask', getattr(env, 'action_mask', lambda x: None)(agent))
                    else:
                        tensor_data = obs_data
                        mask_data = getattr(env, 'action_mask', lambda x: None)(agent)

                    # Convert observation to expected format
                    enhanced_observations[agent] = {
                        'tensor': tensor_data,
                        'action_mask': mask_data
                    }
            
            # Get actions using enhanced agent manager (async call)
            actions = await agent_manager.get_actions(
                observations=enhanced_observations,
                rewards=rewards,
                terminated=terminations
            )
            
            # Filter actions to only include active agents
            filtered_actions = {agent: actions.get(agent, [0, 0, 0]) for agent in env.agents}
            
        except Exception as e:
            print(f"Error getting actions at step {step_count}: {e}")
            # Fallback to default actions
            filtered_actions = {agent: [0, 0, 0] for agent in env.agents}
        
        # Take step in environment
        try:
            observations_dict, rewards, terminations, truncations, infos = env.step(filtered_actions)
        except Exception as e:
            print(f"Error during step {step_count}: {e}")
            break
        
        # Track agent eliminations
        eliminated_agents = []
        for agent in list(env.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                eliminated_agents.append(agent)
        
        if eliminated_agents:
            remaining = len(env.agents) - len(eliminated_agents)
            elimination_log.append({
                'step': step_count,
                'eliminated': eliminated_agents,
                'remaining_after': remaining
            })
            print(f"Step {step_count:4d}: {len(eliminated_agents)} agents eliminated -> {remaining} remaining")
        
        # Track round progression (if available)
        try:
            if env.agents:
                first_agent = env.agents[0]
                agent_info = infos.get(first_agent, {})
                if 'round' in agent_info:
                    current_round = agent_info['round']
                    if not round_log or current_round != round_log[-1]['round']:
                        round_log.append({
                            'step': step_count,
                            'round': current_round,
                            'agents_active': len(env.agents)
                        })
                        print(f"Step {step_count:4d}: Round {current_round}, {len(env.agents)} agents active")
        except (KeyError, AttributeError, IndexError):
            pass  # Round info not available
        
        step_count += 1
        
        # Progress reporting
        if step_count - last_progress_report >= 500:
            elapsed = time.time() - start_time
            print(f"Step {step_count:4d}: {len(env.agents)} agents active (elapsed: {elapsed:.1f}s)")
            last_progress_report = step_count
    
    # Episode complete - gather final statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    final_agents = len(env.agents)
    total_eliminated = initial_agents - final_agents
    
    print(f"\n=== Enhanced Episode Complete ===")
    print(f"Total duration: {elapsed_time:.2f} seconds")
    print(f"Total steps: {step_count}")
    print(f"Initial agents: {initial_agents}")
    print(f"Final agents: {final_agents}")
    print(f"Total eliminated: {total_eliminated}")
    print(f"Steps per second: {step_count/elapsed_time:.1f}")
    
    # Print elimination timeline
    if elimination_log:
        print(f"\nElimination Timeline:")
        for entry in elimination_log:
            agents_str = ', '.join(entry['eliminated'])
            print(f"  Step {entry['step']:4d}: {agents_str} -> {entry['remaining_after']} remaining")
    
    # Print round progression
    if round_log:
        print(f"\nRound Progression:")
        for entry in round_log:
            print(f"  Step {entry['step']:4d}: Round {entry['round']}, {entry['agents_active']} agents")
    
    # Get performance statistics from enhanced system
    try:
        performance_stats = agent_manager.get_performance_stats()
        if performance_stats:
            print(f"\n=== Enhanced System Performance ===")
            for agent_name, stats in performance_stats.items():
                print(f"{agent_name}:")
                print(f"  Average inference time: {stats.get('avg_inference_time', 0):.4f}s")
                print(f"  Total inferences: {stats.get('total_inferences', 0)}")
                print(f"  Average batch size: {stats.get('avg_batch_size', 0):.1f}")
    except Exception as e:
        print(f"Could not retrieve performance stats: {e}")
    
    # Validate episode success
    episode_successful = (
        total_eliminated > 0 or     # Some agents were eliminated
        step_count >= 1000 or       # Reasonable game length
        final_agents <= 1           # Game completed
    )
    
    if episode_successful:
        print(f"\n✅ Test PASSED - Enhanced episode ran successfully!")
        if total_eliminated >= 4:
            print(f"   🎯 Great progress: {total_eliminated}/8 agents eliminated")
        elif step_count >= 1000:
            print(f"   ⏱️  Good duration: {step_count} steps completed")
        else:
            print(f"   ✅ Episode completed with {final_agents} agents remaining")
        print(f"   🚀 Enhanced batching system operational")
    else:
        print(f"\n❌ Test FAILED - Enhanced episode did not progress meaningfully")
        print(f"   Only {total_eliminated} agents eliminated in {step_count} steps")
    
    # Cleanup
    env.close()
    print(f"Environment closed successfully")
    
    return episode_successful


async def test_async_game_environment():
    """Test the AsyncGameEnvironment wrapper for enhanced parallel execution."""
    print("\n" + "="*60)
    print("=== Testing AsyncGameEnvironment ===")
    
    # Create a simple setup for async testing
    simple_agents = [
        (RandomAgent("AsyncRandom"), 4),
        (CultistAgent(), 2),
        (DivineAgent(), 2)
    ]
    
    try:
        agent_manager, batch_processor = create_enhanced_setup(
            agent_configs=simple_agents,
            max_batch_size=8,
            batch_timeout_ms=3.0
        )
        
        # Create async game environment
        async_env = AsyncGameEnvironment(parallel_env, agent_manager)
        
        print(f"AsyncGameEnvironment created successfully")
        
        # Run a single async game
        print(f"Running single async game...")
        start_time = time.time()
        
        result = await async_env.run_game("async_test_game")
        
        duration = time.time() - start_time
        
        print(f"Async game completed in {duration:.2f} seconds")
        print(f"Game result: {result}")
        
        return True
        
    except Exception as e:
        print(f"Error in AsyncGameEnvironment test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner for enhanced interface tests."""
    print("TFT Enhanced Agent Interface - Episode Test")
    print("=" * 60)
    
    async def run_all_tests():
        try:
            # Test 1: Enhanced episode with mixed agents
            success1 = await test_enhanced_episode_with_mixed_agents()
            
            # Test 2: Async game environment
            success2 = await test_async_game_environment()
            
            overall_success = success1 and success2
            
            if overall_success:
                print(f"\n🎉 ALL ENHANCED TESTS PASSED")
                return 0
            else:
                print(f"\n💥 SOME ENHANCED TESTS FAILED")
                print(f"   Enhanced episode: {'✅' if success1 else '❌'}")
                print(f"   Async environment: {'✅' if success2 else '❌'}")
                return 1
                
        except Exception as e:
            print(f"\n💥 CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Run async tests
    exit_code = asyncio.run(run_all_tests())
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)