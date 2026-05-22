"""
Single-threaded training routine for TFT MuZero Agent

This is a simplified training script that runs games sequentially in a single thread
to help debug issues without the complexity of parallel execution.

Uses the direct parallel_env approach with simple agent management.

Basic training loop:
1. Create environment and agents directly
2. Run episodes sequentially using direct agent calls
3. Collect experience and train periodically
4. Monitor progress with simple logging
"""

import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional
import torch
import copy

# Core imports
import config
from Models.global_buffer import GlobalBuffer
from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env

# Agent imports
from Models.MuZero_torch_agent import MuZeroAgent
from Models.Common_agents import RandomAgent, CultistAgent, DivineAgent, FastLevelAgent, RerollAgent

def create_training_agents():
    """Create direct agent instances for training"""
    print("Creating direct agent instances...")
    
    # Create global buffer for agents
    global_buffer = GlobalBuffer(config.BATCH_SIZE)
    
    # Create agent instances
    muzero_agent = MuZeroAgent(
        global_buffer=global_buffer
    )
    
    # Create different agent types for variety
    agents = {
        'player_0': RandomAgent("RandomAgent_0", global_buffer),
        'player_1': RandomAgent("RandomAgent_1", global_buffer),
        'player_2': RandomAgent("RandomAgent_2", global_buffer),
        'player_3': muzero_agent,
        'player_4': CultistAgent(global_buffer),
        'player_5': DivineAgent(global_buffer),
        'player_6': FastLevelAgent(global_buffer),
        'player_7': RerollAgent(global_buffer),
    }
    
    print(f"Created {len(agents)} agents:")
    for player_id, agent in agents.items():
        print(f"  {player_id}: {type(agent).__name__}")
    
    return agents, global_buffer

def create_simple_environment():
    """Create a simple TFT environment"""
    print("Creating TFT environment...")
    env = parallel_env(rank=0)
    print(f"Environment created with {len(env.possible_agents)} possible agents")
    return env

def run_single_episode(env, agents: Dict[str, Any], episode_num: int) -> Dict[str, Any]:
    """
    Run a single episode/hand of TFT using direct agent calls
    
    Args:
        env: TFT environment
        agents: Dictionary mapping player_id to agent instances
        episode_num: Episode number for tracking
        
    Returns:
        Episode statistics
    """
    start_time = time.time()
    
    # Start a new hand
    observations, info = env.reset()
    terminated = {player_id: False for player_id in env.possible_agents}
    rewards = {player_id: 0.0 for player_id in env.possible_agents}
    final_rewards = {player_id: 0.0 for player_id in env.possible_agents}
    
    step_count = 0
    
    print(f"Episode {episode_num}: Starting with {len(observations)} players")
    
    # Play one complete hand
    while env.agents:
        step_count += 1
        
        if step_count % 100 == 0:  # Periodic logging
            print(f"  Step {step_count}, {len(env.agents)} players still active")
        
        # Get actions from each agent directly
        actions = {}
        for agent_id in env.agents:
            if agent_id in agents and agent_id in observations:
                agent = agents[agent_id]
                obs_dict = observations[agent_id]
                action = agent.select_action(obs_dict)
                actions[agent_id] = action
        
        # Take action and observe result
        observations, rewards, terminations, truncations, info = env.step(actions)
        
        # Handle agent terminations - check if any eliminations occurred first
        eliminated_agents = []
        if any(terminations.values()) or any(truncations.values()):
            print(rewards)
            for agent_id in list(terminations.keys()):
                if terminations.get(agent_id, False) or truncations.get(agent_id, False):
                    eliminated_agents.append(agent_id)
                    final_reward = rewards.get(agent_id, 0.0)
                    print(f'Agent {agent_id} terminated with reward {final_reward}')
                    agents[agent_id].terminate(final_reward)
                    final_rewards[agent_id] = final_reward
        
        if eliminated_agents:
            remaining = len(env.agents)
            print(f"    Step {step_count}: {eliminated_agents} eliminated, {remaining} remaining")
    
    duration = time.time() - start_time
    
    # Calculate final results focused on MuZero agent
    all_rewards = list(final_rewards.values())
    print("  Final rewards:", all_rewards)
    avg_reward = np.mean(all_rewards) if all_rewards else 0.0
    
    # Get MuZero agent's performance (player_3)
    muzero_player_id = 'player_3'
    muzero_reward = final_rewards.get(muzero_player_id, 0.0)
    
    # Calculate MuZero agent's placement based on reward ranking
    sorted_rewards = sorted(final_rewards.values(), reverse=True)
    muzero_placement = sorted_rewards.index(muzero_reward) + 1 if muzero_reward in sorted_rewards else 8
    
    episode_stats = {
        'episode': episode_num,
        'duration': duration,
        'steps': step_count,
        'avg_reward': avg_reward,
        'muzero_reward': muzero_reward,
        'muzero_placement': muzero_placement,
        'total_players': len(rewards)
    }
    
    print(f"Episode {episode_num} completed: {duration:.2f}s, {step_count} steps, "
          f"avg reward: {avg_reward:.2f}, MuZero reward: {muzero_reward:.2f}, MuZero placement: {muzero_placement}")
    
    return episode_stats

def train_single_threaded(n_episodes: int = 10, train_every: int = 5):
    """
    Main single-threaded training routine using direct agent calls
    
    Args:
        n_episodes: Number of episodes to run
        train_every: Train the model every N episodes
    """
    print("=" * 60)
    print("TFT MuZero Single-Threaded Training (Direct Agents)")
    print("=" * 60)
    print(f"Episodes to run: {n_episodes}")
    print(f"Training frequency: every {train_every} episodes")
    print(f"GPU available: {torch.cuda.is_available()}")
    print()
    
    # Create environment and agents
    env = create_simple_environment()
    agents, global_buffer = create_training_agents()
    
    # Track statistics
    episode_stats = []
    total_start_time = time.time()
    
    # Main training loop
    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        try:
            # Run single episode using direct agent calls
            stats = run_single_episode(env, agents, episode + 1)
            episode_stats.append(stats)
            exit()
            
            # Periodic training
            if (episode + 1) % train_every == 0:
                print(f"\nTraining check at episode {episode + 1}...")
                
                # Check if we have enough data to train
                if hasattr(global_buffer, 'available_gameplay_batch'):
                    if global_buffer.available_gameplay_batch():
                        print("  Training data available - would train here")
                        # TODO: Add actual training step
                        # trainer.train_network(batch, muzero_agent.model, episode)
                    else:
                        print("  Not enough training data yet")
                else:
                    print("  Buffer doesn't support batch checking")
                
                # Print recent performance
                recent_stats = episode_stats[-train_every:]
                avg_reward = np.mean([s['avg_reward'] for s in recent_stats])
                muzero_avg_reward = np.mean([s['muzero_reward'] for s in recent_stats])
                muzero_avg_placement = np.mean([s['muzero_placement'] for s in recent_stats])
                avg_duration = np.mean([s['duration'] for s in recent_stats])
                avg_steps = np.mean([s['steps'] for s in recent_stats])
                
                print(f"  Recent performance (last {len(recent_stats)} episodes):")
                print(f"    Overall avg reward: {avg_reward:.2f}")
                print(f"    MuZero avg reward: {muzero_avg_reward:.2f}")
                print(f"    MuZero avg placement: {muzero_avg_placement:.1f}")
                print(f"    Avg duration: {avg_duration:.2f}s")
                print(f"    Avg steps: {avg_steps:.1f}")
                print()
        
        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user at episode {episode + 1}")
            break
        except Exception as e:
            print(f"\nError in episode {episode + 1}: {e}")
            import traceback
            traceback.print_exc()
            break  # Continue with next episode
    
    # Final statistics
    total_duration = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Total episodes: {len(episode_stats)}")
    print(f"Total time: {total_duration:.2f}s")
    
    if episode_stats:
        all_rewards = [s['avg_reward'] for s in episode_stats]
        muzero_rewards = [s['muzero_reward'] for s in episode_stats]
        muzero_placements = [s['muzero_placement'] for s in episode_stats]
        all_durations = [s['duration'] for s in episode_stats]
        all_steps = [s['steps'] for s in episode_stats]
        
        print(f"Overall average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        print(f"MuZero average reward: {np.mean(muzero_rewards):.2f} ± {np.std(muzero_rewards):.2f}")
        print(f"MuZero average placement: {np.mean(muzero_placements):.1f} ± {np.std(muzero_placements):.1f}")
        print(f"Average episode duration: {np.mean(all_durations):.2f}s ± {np.std(all_durations):.2f}")
        print(f"Average steps per episode: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
        
        # Show MuZero improvement trend
        if len(episode_stats) >= 10:
            early_rewards = np.mean([s['muzero_reward'] for s in episode_stats[:5]])
            late_rewards = np.mean([s['muzero_reward'] for s in episode_stats[-5:]])
            early_placement = np.mean([s['muzero_placement'] for s in episode_stats[:5]])
            late_placement = np.mean([s['muzero_placement'] for s in episode_stats[-5:]])
            print(f"MuZero reward trend: {early_rewards:.2f} -> {late_rewards:.2f} "
                  f"({late_rewards - early_rewards:+.2f})")
            print(f"MuZero placement trend: {early_placement:.1f} -> {late_placement:.1f} "
                  f"({late_placement - early_placement:+.1f})")
    
    return episode_stats

def quick_debug_test():
    """Quick test to verify everything works"""
    print("Running quick debug test...")
    
    try:
        # Test environment creation
        env = create_simple_environment()
        print("✓ Environment creation successful")
        
        # Test direct agent setup
        agents, global_buffer = create_training_agents()
        print("✓ Direct agent setup successful")
        
        # Test single episode
        print("Running one test episode...")
        stats = run_single_episode(env, agents, 0)
        print("✓ Single episode completed successfully")
        print(f"  Episode stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Single-threaded TFT training')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--train_every', '-t', type=int, default=5,
                        help='Train every N episodes')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Run quick debug test only')
    
    args = parser.parse_args()
    
    if args.debug:
        success = quick_debug_test()
        if success:
            print("\nDebug test passed! Ready for full training.")
        else:
            print("\nDebug test failed. Check errors above.")
    else:
        train_single_threaded(args.episodes, args.train_every)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()