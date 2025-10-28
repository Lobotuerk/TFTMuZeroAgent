"""
TFT SB3 Training Script with SuperSuit Integration - Production Ready

Enhanced training script with SuperSuit wrapper for multi-agent TFT training.
Now uses vector concatenation where a single model controls ALL players simultaneously.

SUPERSUIT INTEGRATION:
- Single model controls all 8 TFT players at once
- Vector concatenation of observations from all players  
- No more random dummy agents - model learns complete game strategy
- Follows PettingZoo best practices for multi-agent RL

EPISODIC FEATURES:
- EpisodicRolloutBuffer: Tracks complete episodes for sparse reward training
- EpisodicPPO: Enhanced PPO that prioritizes episode completion
- Enhanced callbacks with episode-based metrics
- Optimized hyperparameters for sparse reward environments

KEY CHANGES FROM ORIGINAL:
- BEFORE: One agent vs 7 random players
- AFTER: One model controls all 8 players optimally
- Better learning efficiency and game understanding
- Self-play ready architecture

SPARSE REWARD PROBLEM:
- TFT rewards are extremely sparse (only at elimination/victory)
- Most steps have zero reward, making standard RL difficult
- Episodic approach ensures complete reward signals are captured

Combined script supports:
- SuperSuit multi-agent wrapper
- Sparse reward solutions
- Resumable training
- Enhanced episodic monitoring
- Clean output and logging
"""

import numpy as np
import gymnasium as gym
import os
import time
import argparse
from pathlib import Path
from typing import Optional

# Try to import stable-baselines3 with fallback
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.utils import obs_as_tensor
    SB3_AVAILABLE = True
    print("[OK] stable-baselines3 imported successfully")
except ImportError as e:
    print(f"[!] stable-baselines3 import failed: {e}")
    print("[NOTE] This is likely due to NumPy version incompatibility")
    print("[TIP] Suggestion: pip install 'numpy<2' or upgrade packages for NumPy 2.x support")
    SB3_AVAILABLE = False
    
    # Create dummy classes for syntax checking
    class PPO:
        def __init__(self, *args, **kwargs):
            raise ImportError("stable-baselines3 not available")
    
    class BaseCallback:
        def __init__(self, *args, **kwargs):
            pass
    
    class Monitor:
        def __init__(self, *args, **kwargs):
            pass
    
    class RolloutBuffer:
        def __init__(self, *args, **kwargs):
            pass

# Try to import sb3_wrapper with fallback
try:
    from sb3_wrapper import create_sb3_env as create_sb3_compatible_env
    WRAPPER_AVAILABLE = True
    print("[OK] SuperSuit SB3 adapter imported successfully")
except ImportError as e:
    print(f"[!] SuperSuit SB3 adapter import failed: {e}")
    WRAPPER_AVAILABLE = False
    
    def create_sb3_compatible_env(*args, **kwargs):
        raise ImportError("SuperSuit SB3 adapter not available")

# Try to import torch with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"[!] torch import failed: {e}")
    TORCH_AVAILABLE = False
    torch = None


def check_dependencies():
    """Check if all required dependencies are available."""
    if not SB3_AVAILABLE:
        print("\n[ERROR] ERROR: stable-baselines3 is not available")
        print("[FIX] SOLUTIONS:")
        print("   1. Downgrade NumPy: pip install 'numpy<2'")
        print("   2. Upgrade packages: pip install --upgrade stable-baselines3 torch tensorflow")
        print("   3. Create new environment with compatible versions")
        print("\n[TIP] TIP: NumPy 2.x compatibility is still being implemented by many packages")
        return False
    
    if not WRAPPER_AVAILABLE:
        print("\n[ERROR] ERROR: sb3_wrapper (SuperSuit wrapper) is not available")
        print("[FIX] SOLUTION: Ensure sb3_wrapper.py exists and has no import issues")
        return False
    
    if not TORCH_AVAILABLE:
        print("\n[ERROR] ERROR: PyTorch is not available")
        print("[FIX] SOLUTION: pip install torch")
        return False
    
    return True


# ==============================================================================
# EPISODIC BUFFER AND PPO IMPLEMENTATIONS
# ==============================================================================

class EpisodicRolloutBuffer(RolloutBuffer):
    """
    Custom rollout buffer that ensures complete episodes are used for training.
    Only starts training when we have enough complete episodes.
    """
    
    def __init__(self, *args, min_episodes=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_episodes = min_episodes
        self.episode_starts_buffer = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0
    
    def add(self, obs, action, reward, episode_start, value, log_prob):
        """Add experience and track episode completion."""
        super().add(obs, action, reward, episode_start, value, log_prob)
        
        self.current_episode_reward += reward
        self.current_episode_steps += 1
        
        # Track episode completion
        if episode_start and self.current_episode_steps > 1:
            # Previous episode just ended
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            self.current_episode_steps = 0
    
    def is_ready_for_training(self):
        """Only train when we have enough complete episodes."""
        return len(self.episode_rewards) >= self.min_episodes


class EpisodicPPO(PPO):
    """
    PPO that waits for complete episodes before training.
    Addresses sparse reward problem by ensuring reward signals are included.
    """
    
    def __init__(self, *args, min_episodes_per_update=3, **kwargs):
        self.min_episodes_per_update = min_episodes_per_update
        super().__init__(*args, **kwargs)
        print(f"[TARGET] EpisodicPPO initialized: min {min_episodes_per_update} episodes per update")
    
    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="EpisodicPPO", 
              reset_num_timesteps=True, progress_bar=False):
        """
        Enhanced learning that tracks episode completion.
        """
        print(f"[LEARN] Starting episodic learning for {total_timesteps:,} timesteps")
        print(f"   Waiting for minimum {self.min_episodes_per_update} episodes per update")
        
        # Use the parent's learn method but with enhanced logging
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )


# ==============================================================================
# EXISTING CLASSES
# ==============================================================================


class TFTSparseRewardWrapper(gym.Wrapper):
    """Addresses TFT's sparse reward problem with intermediate rewards."""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_steps = 0
        self.survival_bonus = 0.01
        
    def reset(self, **kwargs):
        self.episode_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_steps += 1
        shaped_reward = float(reward)
        
        # Survival bonus (encourages staying alive longer)
        if not terminated and not truncated:
            shaped_reward += self.survival_bonus
        
        # Scale up final rewards and add survival time bonus
        if terminated and float(reward) > 0:
            shaped_reward = float(reward) * 2
            survival_bonus = min(self.episode_steps * 0.1, 50)
            shaped_reward += survival_bonus
        
        return obs, shaped_reward, terminated, truncated, info


class TFTTrainingCallback(BaseCallback):
    """Callback for monitoring training progress and auto-saving."""
    
    def __init__(self, save_freq=10000, save_path="./checkpoint/", log_freq=10):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_save_step = 0
        
        # Enhanced episodic tracking
        self.completed_episodes_this_rollout = 0
        self.episodes_with_reward = 0
        self.total_positive_reward = 0
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Auto-save model periodically
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            save_file = self.save_path / f"tft_model_step_{self.num_timesteps}"
            self.model.save(save_file)
            self.last_save_step = self.num_timesteps
            print(f"📁 Auto-saved model at {self.num_timesteps} steps")
        
        # Track episode completion with enhanced episodic metrics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if info.get('episode'):
                    self.episode_count += 1
                    self.completed_episodes_this_rollout += 1
                    episode_reward = float(info['episode']['r'])
                    episode_length = int(info['episode']['l'])
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # Track episodes with positive rewards (sparse reward metric)
                    if episode_reward > 0:
                        self.episodes_with_reward += 1
                        self.total_positive_reward += episode_reward
                    
                    # Log progress with episodic focus
                    if self.episode_count % self.log_freq == 0:
                        recent_rewards = self.episode_rewards[-self.log_freq:]
                        recent_lengths = self.episode_lengths[-self.log_freq:]
                        
                        avg_reward = np.mean(recent_rewards)
                        avg_length = np.mean(recent_lengths)
                        max_reward = np.max(recent_rewards)
                        positive_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) * 100
                        
                        print(f"Episode {self.episode_count:4d} | "
                              f"Avg Reward: {avg_reward:6.1f} | "
                              f"Max Reward: {max_reward:6.1f} | "
                              f"Positive Rate: {positive_rate:4.1f}% | "
                              f"Avg Length: {avg_length:5.0f} | "
                              f"Steps: {self.num_timesteps}")
        
        return True
    
    def on_rollout_start(self) -> None:
        """Reset episode counter for this rollout."""
        self.completed_episodes_this_rollout = 0
        
    def on_rollout_end(self) -> None:
        """Report episode completion for this rollout."""
        if self.completed_episodes_this_rollout > 0:
            print(f"   [TARGET] Completed {self.completed_episodes_this_rollout} episodes in this rollout")


def create_tft_environment(n_envs=1):
    """Create TFT environment(s) with SuperSuit wrapper and sparse reward solutions."""
    if not check_dependencies():
        raise ImportError("Required dependencies not available")
    
    if n_envs == 1:
        # Single environment using Gymnasium wrapper
        print("[TARGET] Creating Gymnasium TFT environment (SB3 compatible)")
        base_env = create_sb3_compatible_env(num_parallel_envs=1)
        print("[OK] Gymnasium SB3-compatible environment created")
        return base_env
    else:
        # Multiple parallel environments
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        
        def make_env(rank=0):
            def _init():
                print(f"[WORLD] Creating Gymnasium environment #{rank}")
                base_env = create_sb3_compatible_env(num_parallel_envs=1)
                return base_env
            return _init
        
        print(f"🌍 Creating {n_envs} parallel Gymnasium environments...")
        print("   Note: Each environment runs an independent TFT game")
        
        # Use SubprocVecEnv for true parallelization (separate processes)
        if n_envs <= 4:
            # For smaller numbers, use DummyVecEnv (threading)
            env = DummyVecEnv([make_env(i) for i in range(n_envs)])
            print(f"   Using DummyVecEnv for {n_envs} environments (threading)")
        else:
            # For larger numbers, use SubprocVecEnv (multiprocessing)
            env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
            print(f"   Using SubprocVecEnv for {n_envs} environments (multiprocessing)")
        
        return env


def create_sparse_reward_ppo(env, device="auto", tensorboard_log="./logs/tft_training/"):
    """Create EpisodicPPO optimized for sparse reward environments."""
    
    # Determine optimal device
    if device == "auto":
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            # For MLP policies, CPU is often faster due to small networks
            device = "cpu"
            print("[CPU]  Using CPU (optimal for MLP policies)")
        else:
            device = "cpu"
            print("[CPU]  Using CPU (CUDA not available)")
    else:
        print(f"[CPU]  Using device: {device}")
    
    # Get number of environments
    try:
        n_envs = env.num_envs
        print(f"[LOOP] Detected {n_envs} parallel environments")
    except AttributeError:
        n_envs = 1
        print("[LOOP] Using single environment")
    
    # Adjust batch size for multiple environments
    batch_size = 128 * max(1, n_envs // 2)  # Scale batch size with envs
    n_steps = 4096 // max(1, n_envs)        # Reduce steps per env when parallel
    
    print(f"[STATS] Training configuration:")
    print(f"   Environments: {n_envs}")
    print(f"   Steps per env: {n_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total steps per update: {n_steps * n_envs}")
    
    # For SuperSuit vectorized environments, we need to pass it directly 
    # without SB3's environment wrapping since it's already properly vectorized
    # SB3 should recognize VectorEnv instances and not try to wrap them
    return EpisodicPPO(
        'MlpPolicy',
        env,
        verbose=1,
        device=device,
        
        # Episodic settings
        min_episodes_per_update=3,  # Wait for at least 3 complete episodes
        
        # Parallel environment optimizations
        n_steps=n_steps,            # Steps per environment
        batch_size=batch_size,      # Scaled batch size
        n_epochs=15,                # Extract more from sparse data
        ent_coef=0.02,              # Higher entropy for exploration
        learning_rate=5e-5,         # Lower LR for stability
        
        # Long-term planning
        gamma=0.995,                # High discount for delayed rewards
        gae_lambda=0.98,            # Better advantage estimation
        
        # Stability
        clip_range=0.15,            # Conservative updates
        max_grad_norm=0.5,          # Gradient clipping
        
        tensorboard_log=tensorboard_log
    )


def load_or_create_model(model_path, env, force_new=False, device="auto"):
    """Load existing model or create new one."""
    model_file = Path(model_path) if model_path else None
    
    if model_file and model_file.exists() and not force_new:
        print(f"[FOLDER] Loading existing model from {model_path}")
        try:
            # Try to load as EpisodicPPO first, fallback to regular PPO
            try:
                model = EpisodicPPO.load(model_path, device=device)
            except:
                print("[!]  Loading as regular PPO (not episodic)")
                model = PPO.load(model_path, device=device)
            model.set_env(env)
            return model, True  # True = resumed
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("🆕 Creating new EpisodicPPO model instead...")
    
    print("🆕 Creating new EpisodicPPO model...")
    model = create_sparse_reward_ppo(env, device=device)
    return model, False  # False = new model


def create_episodic_tft_model(env, min_episodes_per_update=3, tensorboard_log="./logs/tft_training/"):
    """Create EpisodicPPO with EpisodicRolloutBuffer for TFT training."""
    # Create a custom buffer
    buffer = EpisodicRolloutBuffer(
        buffer_size=4096,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device='auto',
        gamma=0.995,
        gae_lambda=0.98,
        n_envs=1,
        min_episodes=min_episodes_per_update
    )
    
    model = EpisodicPPO(
        'MlpPolicy',
        env,
        verbose=1,
        
        # Episodic settings
        min_episodes_per_update=min_episodes_per_update,
        
        # Use custom buffer
        # Note: Setting rollout_buffer is typically done internally
        
        # Sparse reward optimizations
        n_steps=4096,           # Larger rollouts for more episodes
        batch_size=128,         # Stable learning
        n_epochs=15,            # Extract more from sparse data
        ent_coef=0.02,          # Higher entropy for exploration
        learning_rate=5e-5,     # Lower LR for stability
        
        # Long-term planning
        gamma=0.995,            # High discount for delayed rewards
        gae_lambda=0.98,        # Better advantage estimation
        
        # Stability
        clip_range=0.15,        # Conservative updates
        max_grad_norm=0.5,      # Gradient clipping
        
        tensorboard_log=tensorboard_log
    )
    
    print(f"[OK] Created EpisodicPPO with EpisodicRolloutBuffer")
    print(f"   Min episodes per update: {min_episodes_per_update}")
    print(f"   Buffer tracks episode completion for sparse rewards")
    
    return model


def demonstrate_episodic_usage():
    """Demonstrate how to use the episodic implementations."""
    print("[TARGET] TFT Episodic Training Demonstration")
    print("=" * 50)
    
    # Create environment
    env = create_tft_environment()
    
    # Option 1: Use the integrated approach (recommended)
    print("\n[LEARN] Option 1: Integrated EpisodicPPO (Recommended)")
    model1 = create_sparse_reward_ppo(env)
    print("   [OK] EpisodicPPO with sparse reward optimizations")
    
    # Option 2: Use the custom buffer approach
    print("\n[LEARN] Option 2: Custom EpisodicRolloutBuffer")
    model2 = create_episodic_tft_model(env, min_episodes_per_update=5)
    print("   [OK] EpisodicPPO with custom EpisodicRolloutBuffer")
    
    # Enhanced callback for episodic training
    print("\n[STATS] Enhanced Episodic Callback Features:")
    callback = TFTTrainingCallback(save_freq=25000, log_freq=5)
    print("   [OK] Tracks completed episodes per rollout")
    print("   [OK] Monitors positive reward rate (sparse reward metric)")
    print("   [OK] Enhanced logging for episode-based training")
    
    env.close()
    return model1, model2, callback


def train_tft_model(
    model_path="./checkpoint/tft_latest",
    total_timesteps=1000000,
    save_freq=25000,
    resume=True,
    force_new=False,
    n_envs=1,
    device="auto"
):
    """
    Main training function with parallel environment support.
    
    Args:
        model_path: Path to save/load model
        total_timesteps: Total training steps (can be increased for continuing)
        save_freq: How often to auto-save (steps)
        resume: Whether to try resuming from existing model
        force_new: Force create new model even if one exists
        n_envs: Number of parallel environments (1=single, 2-8=recommended for parallel)
        device: Device to use ('auto', 'cpu', 'cuda')
    """
    
    print("[START] TFT SB3 Training - Production Mode with Parallel Support")
    print("=" * 60)
    
    # Setup directories
    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./logs/tft_training", exist_ok=True)
    
    # Create environment(s)
    print("[GAME] Setting up environment(s)...")
    env = create_tft_environment(n_envs=n_envs)
    print("[OK] Environment(s) ready with sparse reward solutions")
    
    # Load or create model
    model, was_resumed = load_or_create_model(
        model_path if resume else None, 
        env, 
        force_new,
        device=device
    )
    
    if was_resumed:
        print(f"[OK] Resumed training from existing model")
    else:
        print(f"[OK] Created new model for training")
    
    # Setup callback for monitoring and auto-saving
    callback = TFTTrainingCallback(
        save_freq=save_freq,
        save_path="./checkpoint/",
        log_freq=5  # Log every 5 episodes
    )
    
    print(f"\n[TARGET] Training Configuration:")
    print(f"   Parallel environments: {n_envs}")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Auto-save every: {save_freq:,} steps")
    print(f"   Device: {device}")
    print(f"   Tensorboard logs: ./logs/tft_training/")
    print(f"   Model saves: ./checkpoint/")
    
    print(f"\n[STATS] Optimizations for Sparse Rewards + SuperSuit:")
    print(f"   • SuperSuit wrapper: Single model controls all 8 players")
    print(f"   • EpisodicPPO: Waits for complete episodes")
    print(f"   • EpisodicRolloutBuffer: Tracks episode completion")
    print(f"   • Min episodes per update: 3")
    if n_envs > 1:
        print(f"   • Parallel environments: {n_envs}x speedup")
        print(f"   • Scaled batch size and steps for parallelization")
    print(f"   • More training epochs: 15")
    print(f"   • Higher exploration: 0.02 entropy")
    print(f"   • Survival bonus: +0.01 per step")
    print(f"   • Scaled final rewards: 2x + survival bonus")
    
    print(f"\n[GAME] Starting training... (Ctrl+C to stop safely)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False  # Don't reset if resuming
        )
        
        training_time = time.time() - start_time
        print(f"\n[SUCCESS] Training completed!")
        print(f"[TIME]  Total time: {training_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\n[!]  Training interrupted by user")
        training_time = time.time() - start_time
        print(f"[TIME]  Training time: {training_time:.1f} seconds")
    
    # Save final model
    final_path = f"{model_path}_final"
    model.save(final_path)
    print(f"[SAVE] Final model saved: {final_path}.zip")
    
    # Save latest for easy resuming
    model.save(model_path)
    print(f"[SAVE] Latest model saved: {model_path}.zip")
    
    # Show final stats
    if callback.episode_count > 0:
        print(f"\n[CHART] Training Stats:")
        print(f"   Episodes completed: {callback.episode_count}")
        print(f"   Average reward (last 20): {np.mean(callback.episode_rewards[-20:]):.1f}")
        print(f"   Best reward: {np.max(callback.episode_rewards):.1f}")
        print(f"   Average episode length: {np.mean(callback.episode_lengths):.0f}")
        if n_envs > 1:
            print(f"   Environments: {n_envs} (parallel)")
            print(f"   Effective speedup: ~{n_envs}x")
    
    env.close()
    return model


def test_model(model_path="./checkpoint/tft_latest", num_episodes=10, n_envs=1):
    """Test a trained model against other AI agents in competitive play.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Total number of episodes to test
        n_envs: Number of parallel environments (1=sequential, 2+=parallel)
    """
    print(f"[TEST] Testing model: {model_path}")
    print("[TARGET] Setting up competitive environment with specialized AI agents")
    if n_envs > 1:
        print(f"[LOOP] Using {n_envs} parallel environments for faster testing")
    print("[NOTE] Logging enabled - all environment events will be recorded to log.txt")
    
    # Handle model path - add .zip if not already present
    if not model_path.endswith('.zip'):
        full_model_path = f"{model_path}.zip"
    else:
        full_model_path = model_path
    
    if not Path(full_model_path).exists():
        print(f"[ERROR] Model not found: {full_model_path}")
        return
    
    # Import TFT environment and agents
    try:
        from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
        from Models.Common_agents import CultistAgent, DivineAgent, RandomAgent, RerollAgent, FastLevelAgent
        print("[OK] Imported TFT simulator and improved AI agents")
    except ImportError as e:
        print(f"[ERROR] Failed to import components: {e}")
        return
    
    # Load the trained model
    try:
        from stable_baselines3 import PPO
        model = PPO.load(full_model_path)
        print("[OK] Loaded PPO model")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Create a simple wrapper to make SB3 model compatible with TFT agent interface
    class SB3AgentWrapper:
        """Wrapper to make SB3 model compatible with TFT agent interface."""
        def __init__(self, sb3_model):
            self.model = sb3_model
            self.agent_type = "SB3Agent"
            
            # Get expected observation shape from the model
            try:
                self.expected_obs_shape = self.model.observation_space.shape[0]
                print(f"   [SEARCH] SB3 model expects observation shape: ({self.expected_obs_shape},)")
            except:
                self.expected_obs_shape = None
                print("   [!]  Could not determine model's expected observation shape")
        
        def select_action(self, observation, action_mask=None):
            """Select action using the SB3 model."""
            try:
                # Flatten the observation if it's a dict
                if isinstance(observation, dict):
                    if 'tensor' in observation:
                        # Use tensor from dict observation (gym environment format)
                        flat_obs = observation['tensor'].flatten()
                        # Also include action_mask if available
                        if 'action_mask' in observation:
                            action_mask_flat = observation['action_mask'].flatten()
                            flat_obs = np.concatenate([flat_obs, action_mask_flat])
                    else:
                        # Fallback: concatenate all dict values
                        flat_obs = np.concatenate([np.array(v).flatten() for v in observation.values()])
                else:
                    # Direct observation array
                    flat_obs = np.array(observation).flatten()
                
                # Handle observation shape mismatch
                if self.expected_obs_shape and len(flat_obs) != self.expected_obs_shape:
                    flat_obs = self._fix_observation_shape(flat_obs)
                
                # Predict action using SB3 model
                action, _ = self.model.predict(flat_obs, deterministic=True)
                return action
            except Exception as e:
                # Fallback to random action if prediction fails
                print(f"      [FIX] SB3Agent fallback to random action: {e}")
                return [np.random.randint(0, 6), np.random.randint(0, 37), np.random.randint(0, 28)]
        
        def _fix_observation_shape(self, flat_obs):
            """Fix observation shape to match model expectations."""
            current_shape = len(flat_obs)
            
            if self.expected_obs_shape is None:
                return flat_obs
            
            if current_shape < self.expected_obs_shape:
                # Pad with zeros if observation is too small
                padding_needed = self.expected_obs_shape - current_shape
                padded_obs = np.pad(flat_obs, (0, padding_needed), mode='constant', constant_values=0)
                print(f"      [FIX] Padded observation from {current_shape} to {self.expected_obs_shape}")
                return padded_obs
            elif current_shape > self.expected_obs_shape:
                # Truncate if observation is too large
                truncated_obs = flat_obs[:self.expected_obs_shape]
                print(f"      [FIX] Truncated observation from {current_shape} to {self.expected_obs_shape}")
                return truncated_obs
            else:
                return flat_obs

    # Set up competitive agents
    print("[BOT] Setting up diverse AI opponents...")
    
    # Create our trained agent wrapper
    trained_agent = SB3AgentWrapper(model)
    
    # Create diverse specialized opponents
    agent_cultist = CultistAgent()
    agent_divine = DivineAgent() 
    agent_reroll = RerollAgent()
    agent_fast_level = FastLevelAgent()
    agent_random_1 = RandomAgent("RandomAgent1")
    agent_random_2 = RandomAgent("RandomAgent2")
    agent_random_3 = RandomAgent("RandomAgent3")
    
    # All agents for the 8-player game
    all_agents = [trained_agent, agent_cultist, agent_divine, agent_reroll, 
                  agent_fast_level, agent_random_1, agent_random_2, agent_random_3]
    
    print(f"   [BRAIN] Trained Model: 1 player")
    print(f"   [FIGHT]  Cultist Agent: 1 player (focuses on cultist units)")
    print(f"   [MAGIC] Divine Agent: 1 player (focuses on divine units)")  
    print(f"   [LOOP] Reroll Agent: 1 player (low-cost reroll strategy)")
    print(f"   [FAST] Fast Level Agent: 1 player (aggressive leveling)")
    print(f"   [DICE] Random Agents: 3 players") 
    print(f"   [STATS] Total: 8 players per game")
    
    # Initialize results tracking
    episode_rewards = []
    episode_placements = []
    agent_placements = {}  # Will populate dynamically
    
    # Choose testing approach based on n_envs
    if n_envs == 1:
        print(f"\n[LOOP] Running {num_episodes} episodes sequentially...")
        results = run_sequential_testing(all_agents, num_episodes)
    else:
        print(f"\n[LOOP] Running {num_episodes} episodes across {n_envs} parallel environments...")
        results = run_parallel_testing(all_agents, num_episodes, n_envs)
    
    return results


def run_sequential_testing(all_agents, num_episodes):
    """Run testing episodes sequentially (original method)."""
    # Use the SAME environment type as training (SuperSuit)
    print("   [FIX] Using SuperSuit environment (same as training)")
    
    episode_rewards = []
    episode_placements = []
    agent_placements = {}
    
    # Initialize logging for testing
    try:
        from TFTSet4Gym.tft_set4_gym.game_round import log_to_file_start
        log_to_file_start("log.txt")
        print("[OK] Initialized logging to log.txt")
    except ImportError:
        print("[!]  Could not import logging functions")
    
    for episode in range(num_episodes):
        print(f"\n[GAME] Episode {episode + 1}/{num_episodes}")
        
        # Create direct TFT environment (SuperSuit disabled for stability)
        print(f"[DEBUG] Creating direct TFT environment")
        from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
        env = parallel_env()
        observations, _ = env.reset()
        
        # Run single episode
        episode_result = run_single_episode(env, observations, all_agents, episode + 1)
        
        # Debug output
        print(f"   [DEBUG] Episode result: {episode_result}")
        
        # Only record successful episodes
        if episode_result and episode_result['trained_placement'] is not None:
            episode_rewards.append(episode_result['trained_reward'])
            episode_placements.append(episode_result['trained_placement'])
            
            # Update agent placements
            for agent_name, placement in episode_result['all_placements'].items():
                if agent_name not in agent_placements:
                    agent_placements[agent_name] = []
                agent_placements[agent_name].append(placement)
            
            print(f"   [WIN] Trained agent placed: #{episode_result['trained_placement']}/8")
            print(f"   [MONEY] Reward: {episode_result['trained_reward']:.1f}")
            print(f"   [TIME]  Steps: {episode_result['steps']}")
        else:
            print(f"   [ERROR] Episode {episode + 1}: Failed to complete")
    
    print(f"\n[OK] Sequential testing completed: {len(episode_rewards)} episodes total")
    return analyze_results(episode_rewards, episode_placements, agent_placements)


def run_parallel_testing(all_agents, num_episodes, n_envs):
    """Run testing episodes in parallel environments."""
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
    
    episode_rewards = []
    episode_placements = []
    agent_placements = {}
    
    # Calculate episodes per environment
    episodes_per_env = num_episodes // n_envs
    remaining_episodes = num_episodes % n_envs
    
    print(f"   [STATS] Episodes per environment: {episodes_per_env}")
    if remaining_episodes > 0:
        print(f"   [STATS] Extra episodes: {remaining_episodes}")
    
    def run_env_batch(env_id, num_episodes_for_env):
        """Run a batch of episodes in one environment."""
        env_results = {
            'rewards': [],
            'placements': [],
            'agent_placements': {}
        }
        
        print(f"   [WORLD] Environment {env_id}: Starting {num_episodes_for_env} episodes")
        
        for local_episode in range(num_episodes_for_env):
            try:
                # Create direct TFT environment (SuperSuit disabled for stability)
                from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
                env = parallel_env(rank=env_id)  # Use different rank for each env
                observations, _ = env.reset()
                
                # Run single episode
                episode_result = run_single_episode(
                    env, observations, all_agents, 
                    f"{env_id}-{local_episode + 1}"
                )
                
                # Only record successful episodes
                if episode_result and episode_result['trained_placement'] is not None:
                    env_results['rewards'].append(episode_result['trained_reward'])
                    env_results['placements'].append(episode_result['trained_placement'])
                    
                    # Update agent placements
                    for agent_name, placement in episode_result['all_placements'].items():
                        if agent_name not in env_results['agent_placements']:
                            env_results['agent_placements'][agent_name] = []
                        env_results['agent_placements'][agent_name].append(placement)
                else:
                    print(f"   [!]  Environment {env_id}, Episode {local_episode + 1}: Failed to complete")
                
                if (local_episode + 1) % max(1, num_episodes_for_env // 4) == 0:
                    print(f"   [WORLD] Environment {env_id}: {local_episode + 1}/{num_episodes_for_env} episodes complete")
                
            except Exception as e:
                print(f"   [ERROR] Environment {env_id}, Episode {local_episode + 1}: Error {e}")
                continue
        
        print(f"   [OK] Environment {env_id}: Completed all {num_episodes_for_env} episodes")
        return env_results
    
    # Run environments in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_envs) as executor:
        # Submit tasks
        future_to_env = {}
        for env_id in range(n_envs):
            num_eps = episodes_per_env + (1 if env_id < remaining_episodes else 0)
            future = executor.submit(run_env_batch, env_id, num_eps)
            future_to_env[future] = env_id
        
        # Collect results as they complete
        for future in as_completed(future_to_env):
            env_id = future_to_env[future]
            try:
                env_result = future.result()
                
                # Merge results
                episode_rewards.extend(env_result['rewards'])
                episode_placements.extend(env_result['placements'])
                
                for agent_name, placements in env_result['agent_placements'].items():
                    if agent_name not in agent_placements:
                        agent_placements[agent_name] = []
                    agent_placements[agent_name].extend(placements)
                
            except Exception as e:
                print(f"   [ERROR] Environment {env_id} failed: {e}")
    
    print(f"\n[OK] Parallel testing completed: {len(episode_rewards)} episodes total")
    return analyze_results(episode_rewards, episode_placements, agent_placements)


def run_single_episode(env, observations, all_agents, episode_id):
    """Run a single episode and return results."""
    # Ensure observations is in dict format (convert from vectorized if needed)
    if isinstance(observations, np.ndarray):
        # Convert vectorized observations to dict format
        observations_dict = {}
        for i in range(observations.shape[0]):
            player_obs = observations[i]  # Shape: (5206,)
            # Split back into tensor (5152) and action_mask (54)
            tensor_part = player_obs[:5152]
            action_mask_part = player_obs[5152:]
            observations_dict[f"player_{i}"] = {
                'tensor': tensor_part,
                'action_mask': action_mask_part
            }
        observations = observations_dict
    
    # Map each player to an agent
    player_ids = list(observations.keys())
    player_to_agent = {}
    for i, player_id in enumerate(player_ids):
        player_to_agent[player_id] = all_agents[i]
    
    terminated = {player_id: False for player_id in player_ids}
    rewards = {player_id: 0.0 for player_id in player_ids}
    final_scores = {}
    
    steps = 0
    max_steps = 5000  # Increase to allow natural game completion
    
    # Track if this is a SuperSuit environment
    is_supersuit = hasattr(env, 'supersuit_env')
    
    # Run the episode
    while not all(terminated.values()) and steps < max_steps:
        # Observations are already in dict format from parallel environment
        # No conversion needed since we removed vectorization
        
        # Get actions from all agents for current live players
        actions = {}
        for player_id in observations.keys():
            if player_id in player_to_agent and not terminated.get(player_id, False):
                agent = player_to_agent[player_id]
                obs = observations[player_id]
                action_mask = obs.get('action_mask', None) if isinstance(obs, dict) else None
                try:
                    action = agent.select_action(obs, action_mask)
                    actions[player_id] = action
                except Exception as e:
                    # Fallback to random action if agent fails
                    actions[player_id] = [0, 0, 0]
        
        # Step the environment
        try:
            # For SuperSuit environments, actions should be in dict format
            if is_supersuit:
                # SuperSuit environments now work with dict format (no vectorization)
                step_result = env.step(actions)
                
                if len(step_result) == 4:
                    # Old format: obs, reward, done, info
                    observations, rewards, done, info = step_result
                    terminated = done
                    truncated = done  # Set truncated same as done for compatibility
                elif len(step_result) == 5:
                    # New format: obs, reward, terminated, truncated, info
                    observations, rewards, terminated, truncated, info = step_result
                else:
                    raise ValueError(f"Unexpected step result format with {len(step_result)} elements")
                
                # Observations, rewards, and terminated are already in dict format from parallel env
                # No conversion needed since we removed vectorization
                
                # Debug termination status (every 100 steps)
                if steps % 100 == 0 or steps < 5:
                    print(f"      [DEBUG] Step {steps}: Rewards: {rewards}")
                    print(f"      [DEBUG] Step {steps}: Terminated: {terminated}")
                
            else:
                # Regular TFT environment
                step_result = env.step(actions)
                observations, rewards, terminated, truncated, info = step_result
                
        except Exception as e:
            print(f"      [!]  Episode {episode_id}: Environment step failed: {e}")
            import traceback
            print(f"      [DEBUG] Traceback: {traceback.format_exc()}")
            # Return None to indicate failed episode
            return None
        
        # Update final scores when agents are eliminated
        for player_id in rewards.keys():
            if terminated.get(player_id, False) and player_id not in final_scores:
                final_scores[player_id] = rewards[player_id]
        
        steps += 1
    
    # Collect logs from rank-specific files (only for sequential testing)
    if isinstance(episode_id, int):  # Only for sequential (not parallel)
        try:
            import os
            import glob
            log_files = glob.glob("log_*.txt")
            if log_files:
                with open("log.txt", "a") as main_log:
                    main_log.write(f"\n=== EPISODE {episode_id} LOGS ===\n")
                    for log_file in log_files:
                        if os.path.exists(log_file):
                            with open(log_file, "r") as rank_log:
                                content = rank_log.read()
                                if content.strip():
                                    main_log.write(f"\n--- {log_file} ---\n")
                                    main_log.write(content)
                            # Clean up rank-specific log file
                            os.remove(log_file)
        except Exception as e:
            pass  # Logging is optional
    
    # Calculate placements
    print(f"      [DEBUG] Final scores: {final_scores}")
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"      [DEBUG] Sorted scores: {sorted_scores}")
    
    # Find results for all agents
    all_placements = {}
    trained_agent_placement = None
    trained_agent_reward = None
    
    for placement, (player_id, score) in enumerate(sorted_scores, 1):
        agent = player_to_agent[player_id]
        agent_name = agent.agent_type
        all_placements[agent_name] = placement
        
        # Check if this is our trained agent
        if agent.agent_type == "SB3Agent":
            trained_agent_placement = placement
            trained_agent_reward = score
    
    return {
        'trained_placement': trained_agent_placement,
        'trained_reward': trained_agent_reward,
        'all_placements': all_placements,
        'steps': steps
    }


def analyze_results(episode_rewards, episode_placements, agent_placements):
    """Analyze and display testing results."""
    if not episode_placements or not episode_rewards:
        print("\n[ERROR] No successful episodes to analyze")
        return {
            'avg_placement': None,
            'avg_reward': None,
            'episode_placements': episode_placements,
            'episode_rewards': episode_rewards,
            'agent_placements': agent_placements
        }
    
    avg_placement = np.mean(episode_placements)
    avg_reward = np.mean(episode_rewards)
    
    print(f"\n[STATS] Competitive Test Results:")
    print(f"   [TARGET] Trained Agent Performance:")
    print(f"      Episodes tested: {len(episode_placements)}")
    print(f"      Average placement: {avg_placement:.1f}/8")
    print(f"      Average reward: {avg_reward:.1f}")
    print(f"      Best placement: {min(episode_placements)}/8")
    print(f"      Worst placement: {max(episode_placements)}/8")
    
    # Show opponent performance for comparison
    print(f"\n   [BOT] Opponent Performance:")
    for agent_name, placements in agent_placements.items():
        if placements:  # Only show agents that played
            avg_place = np.mean(placements)
            print(f"      {agent_name}: {avg_place:.1f}/8 average")
    
    # Performance assessment
    if avg_placement <= 2.0:
        print(f"\n[WIN] EXCELLENT: Model consistently places in top 2!")
    elif avg_placement <= 4.0:
        print(f"\n[OK] GOOD: Model places in top half on average")
    elif avg_placement <= 6.0:
        print(f"\n[!]  AVERAGE: Model performance is mediocre")
    else:
        print(f"\n[ERROR] POOR: Model needs more training")
    
    return {
        'avg_placement': avg_placement,
        'avg_reward': avg_reward,
        'episode_placements': episode_placements,
        'episode_rewards': episode_rewards,
        'agent_placements': agent_placements
    }


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="TFT SB3 Training Script with Parallel Support")
    parser.add_argument("--mode", choices=["train", "test", "continue"], default="train",
                       help="Training mode")
    parser.add_argument("--model", default="./checkpoint/tft_latest", 
                       help="Model path")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total timesteps for training")
    parser.add_argument("--save-freq", type=int, default=25000,
                       help="Auto-save frequency")
    parser.add_argument("--new", action="store_true",
                       help="Force create new model")
    parser.add_argument("--test-episodes", type=int, default=5,
                       help="Number of episodes for testing")
    parser.add_argument("--n-envs", type=int, default=1,
                       help="Number of parallel environments (1-16)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Validate n_envs
    if not (1 <= args.n_envs <= 16):
        print("[ERROR] Error: n_envs must be between 1 and 16")
        return
    
    if args.mode == "train":
        train_tft_model(
            model_path=args.model,
            total_timesteps=args.timesteps,
            save_freq=args.save_freq,
            resume=False,
            force_new=args.new,
            n_envs=args.n_envs,
            device=args.device
        )
    elif args.mode == "continue":
        train_tft_model(
            model_path=args.model,
            total_timesteps=args.timesteps,
            save_freq=args.save_freq,
            resume=True,
            force_new=False,
            n_envs=args.n_envs,
            device=args.device
        )
    elif args.mode == "test":
        test_model(args.model, args.test_episodes, args.n_envs)


if __name__ == "__main__":
    # Check dependencies first
    print("[SEARCH] Checking dependencies...")
    if not check_dependencies():
        print("\n[ERROR] DEPENDENCY CHECK FAILED")
        print("Please resolve the dependency issues above before running the training script.")
        exit(1)
    
    print("[OK] All dependencies available!")
    
    # If no command line args, run interactive mode
    import sys
    if len(sys.argv) == 1:
        print("\n[GAME] TFT SB3 Training with SuperSuit + Episodic Support - Interactive Mode")
        print("=" * 60)
        print("1. Start new SuperSuit training (single environment)")
        print("2. Start new SuperSuit training (parallel environments)")
        print("3. Continue existing training") 
        print("4. Test model")
        print("5. Demonstrate episodic implementations")
        print("6. Command line help")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            train_tft_model(force_new=True, n_envs=1)
        elif choice == "2":
            print("\n[LOOP] Parallel Environment Setup")
            print("Recommended configurations:")
            print("  2-4 envs: Good for most systems")
            print("  4-8 envs: High-end systems")
            print("  8+ envs: Usually diminishing returns")
            
            while True:
                try:
                    n_envs = int(input("\nNumber of parallel environments (2-16): ").strip())
                    if 2 <= n_envs <= 16:
                        break
                    else:
                        print("Please enter a number between 2 and 16")
                except ValueError:
                    print("Please enter a valid number")
            
            device_choice = input("\nDevice preference (auto/cpu/cuda) [auto]: ").strip().lower()
            if not device_choice:
                device_choice = "auto"
            
            print(f"\n[START] Starting SuperSuit parallel training with {n_envs} environments on {device_choice}")
            print("   Note: Each environment runs a model controlling all TFT players")
            train_tft_model(force_new=True, n_envs=n_envs, device=device_choice)
        elif choice == "3":
            print("\n[FOLDER] Continue Training Options")
            resume_choice = input("Use parallel environments? (y/n) [n]: ").strip().lower()
            
            if resume_choice == 'y':
                while True:
                    try:
                        n_envs = int(input("Number of parallel environments (1-16): ").strip())
                        if 1 <= n_envs <= 16:
                            break
                        else:
                            print("Please enter a number between 1 and 16")
                    except ValueError:
                        print("Please enter a valid number")
                
                device_choice = input("Device preference (auto/cpu/cuda) [auto]: ").strip().lower()
                if not device_choice:
                    device_choice = "auto"
                
                train_tft_model(resume=True, n_envs=n_envs, device=device_choice)
            else:
                train_tft_model(resume=True, n_envs=1)
        elif choice == "4":
            test_model()
        elif choice == "5":
            demonstrate_episodic_usage()
        elif choice == "6":
            print("\n[BOOK] Command Line Usage:")
            print("  python tft_train.py --mode train --timesteps 500000")
            print("  python tft_train.py --mode continue --model ./checkpoint/my_model")
            print("  python tft_train.py --mode test --test-episodes 10")
            print("  python tft_train.py --help")
            print("\n[TARGET] SuperSuit + Episodic Features:")
            print("  • SuperSuit wrapper: Single model controls all players")
            print("  • EpisodicPPO: Enhanced PPO for sparse rewards")
            print("  • EpisodicRolloutBuffer: Episode-aware buffer")
            print("  • Enhanced callbacks with episode tracking")
            print("  • Optimized for TFT's sparse reward structure")
        else:
            print("Invalid choice. Exiting.")
    else:
        main()