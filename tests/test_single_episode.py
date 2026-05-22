#!/usr/bin/env python3
"""
Standalone test for running a single episode with 8 random agents in TFT parallel environment.
This test creates a parallel environment with 8 agents and runs a complete episode using RandomAgent from Common_agents.
"""

import sys
import os
import time

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from submodule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TFTSet4Gym'))
from tft_set4_gym.tft_simulator import parallel_env
sys.path.pop(0)

# Import RandomAgent from Common_agents
from Models.Common_agents import RandomAgent


def test_single_episode_with_random_agents():
    """Test running a complete episode with 8 RandomAgent instances from Common_agents."""
    print("=== TFT Single Episode Test with 8 RandomAgent Instances ===")
    print(f"Starting test at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create environment
    try:
        env = parallel_env(rank=0)
    except:
        env = parallel_env()
    
    print(f"Environment created successfully")
    
    # Reset and verify 8 agents
    observations, infos = env.reset()
    num_agents = len(env.agents)
    
    print(f"Reset complete - Found {num_agents} agents: {env.agents}")
    assert num_agents == 8, f"Expected 8 agents, got {num_agents}"
    
    # Create RandomAgent instances for each agent
    agent_instances = {}
    for agent_id in env.agents:
        agent_instances[agent_id] = RandomAgent(agent_name=f"RandomAgent_{agent_id}")
    
    print(f"Created {len(agent_instances)} RandomAgent instances")
    
    # Initialize tracking variables
    step_count = 0
    max_steps = 3000  # Generous limit for TFT games
    initial_agents = num_agents
    
    # Game progress tracking
    elimination_log = []
    round_log = []
    last_progress_report = 0
    
    print(f"Starting episode simulation...")
    start_time = time.time()
    
    # Main game loop
    while env.agents and step_count < max_steps:
        # Generate actions using RandomAgent instances for all active agents
        actions = {}
        for agent in env.agents:
            try:
                observation = observations.get(agent, None)
                action_space = env.action_space(agent)
                
                # Use RandomAgent to select action
                if agent in agent_instances and observation is not None:
                    # RandomAgent expects action_mask but we'll pass None since it doesn't use it
                    action = agent_instances[agent].select_action(observation, action_mask=None)
                else:
                    # Fallback to random sampling if agent instance not available
                    action = action_space.sample()
                
                actions[agent] = action
            except Exception as e:
                print(f"Warning: Could not generate action for {agent}: {e}")
                actions[agent] = [0, 0, 0]  # Default action
        
        # Take step in environment
        try:
            observations, rewards, terminations, truncations, infos = env.step(actions)
        except Exception as e:
            print(f"Error during step {step_count}: {e}")
            break
        
        # Track agent eliminations
        eliminated_agents = []
        for agent in list(env.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                eliminated_agents.append(agent)
                # Clean up agent instance
                if agent in agent_instances:
                    # Get final reward from rewards dict, default to 0 if not available
                    final_reward = rewards.get(agent, 0)
                    agent_instances[agent].terminate(final_reward)
                    del agent_instances[agent]
        
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
    
    print(f"\n=== Episode Complete ===")
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
    
    # Validate episode success
    episode_successful = (
        total_eliminated > 0 or     # Some agents were eliminated
        step_count >= 1000 or       # Reasonable game length
        final_agents <= 1           # Game completed
    )
    
    if episode_successful:
        print(f"\n✅ Test PASSED - Episode ran successfully with RandomAgent instances!")
        if total_eliminated >= 4:
            print(f"   🎯 Great progress: {total_eliminated}/8 agents eliminated")
        elif step_count >= 1000:
            print(f"   ⏱️  Good duration: {step_count} steps completed")
        else:
            print(f"   ✅ Episode completed with {final_agents} agents remaining")
    else:
        print(f"\n❌ Test FAILED - Episode did not progress meaningfully")
        print(f"   Only {total_eliminated} agents eliminated in {step_count} steps")
    
    # Cleanup
    env.close()
    print(f"Environment closed successfully")
    
    return episode_successful


def main():
    """Main test runner."""
    print("TFT Parallel Environment - Single Episode Test")
    print("=" * 50)
    
    try:
        success = test_single_episode_with_random_agents()
        
        if success:
            print(f"\n🎉 TEST SUITE PASSED")
            return 0
        else:
            print(f"\n💥 TEST SUITE FAILED") 
            return 1
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)