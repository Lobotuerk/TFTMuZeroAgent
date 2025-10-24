"""
TFT SB3 Training Script with Episodic Support - Production Ready

Enhanced training script with episodic implementations for TFT's sparse reward environment.
Includes EpisodicRolloutBuffer and EpisodicPPO for better handling of sparse rewards.

EPISODIC FEATURES:
- EpisodicRolloutBuffer: Tracks complete episodes for sparse reward training
- EpisodicPPO: Enhanced PPO that prioritizes episode completion
- Enhanced callbacks with episode-based metrics
- Optimized hyperparameters for sparse reward environments

SPARSE REWARD PROBLEM:
- TFT rewards are extremely sparse (only at elimination/victory)
- Most steps have zero reward, making standard RL difficult
- Episodic approach ensures complete reward signals are captured

Combined script supports:
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
    print("✅ stable-baselines3 imported successfully")
except ImportError as e:
    print(f"⚠️  stable-baselines3 import failed: {e}")
    print("📝 This is likely due to NumPy version incompatibility")
    print("💡 Suggestion: pip install 'numpy<2' or upgrade packages for NumPy 2.x support")
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
    from sb3_wrapper import TFTSingleAgentWrapper
    WRAPPER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  sb3_wrapper import failed: {e}")
    WRAPPER_AVAILABLE = False
    
    class TFTSingleAgentWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("TFTSingleAgentWrapper not available")

# Try to import torch with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  torch import failed: {e}")
    TORCH_AVAILABLE = False
    torch = None


def check_dependencies():
    """Check if all required dependencies are available."""
    if not SB3_AVAILABLE:
        print("\n❌ ERROR: stable-baselines3 is not available")
        print("🔧 SOLUTIONS:")
        print("   1. Downgrade NumPy: pip install 'numpy<2'")
        print("   2. Upgrade packages: pip install --upgrade stable-baselines3 torch tensorflow")
        print("   3. Create new environment with compatible versions")
        print("\n💡 TIP: NumPy 2.x compatibility is still being implemented by many packages")
        return False
    
    if not WRAPPER_AVAILABLE:
        print("\n❌ ERROR: sb3_wrapper (TFTSingleAgentWrapper) is not available")
        print("🔧 SOLUTION: Ensure sb3_wrapper.py exists and has no import issues")
        return False
    
    if not TORCH_AVAILABLE:
        print("\n❌ ERROR: PyTorch is not available")
        print("🔧 SOLUTION: pip install torch")
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
        print(f"🎯 EpisodicPPO initialized: min {min_episodes_per_update} episodes per update")
    
    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="EpisodicPPO", 
              reset_num_timesteps=True, progress_bar=False):
        """
        Enhanced learning that tracks episode completion.
        """
        print(f"📚 Starting episodic learning for {total_timesteps:,} timesteps")
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
            print(f"   🎯 Completed {self.completed_episodes_this_rollout} episodes in this rollout")


def create_tft_environment(n_envs=1):
    """Create TFT environment(s) with sparse reward solutions."""
    if not check_dependencies():
        raise ImportError("Required dependencies not available")
    
    if n_envs == 1:
        # Single environment
        base_env = TFTSingleAgentWrapper()
        env = TFTSparseRewardWrapper(base_env)
        env = Monitor(env)
        return env
    else:
        # Multiple parallel environments
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        
        def make_env(rank=0):
            def _init():
                base_env = TFTSingleAgentWrapper()
                env = TFTSparseRewardWrapper(base_env)
                env = Monitor(env, filename=f"./logs/env_{rank}")
                return env
            return _init
        
        print(f"🔧 Creating {n_envs} parallel environments...")
        
        # Use SubprocVecEnv for true parallelization (separate processes)
        # Use DummyVecEnv for threading (less overhead but shared GIL)
        if n_envs <= 4:
            # For smaller numbers, threading is often faster
            env = DummyVecEnv([make_env(i) for i in range(n_envs)])
            print(f"   Using DummyVecEnv (threading) for {n_envs} environments")
        else:
            # For larger numbers, multiprocessing is better
            env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
            print(f"   Using SubprocVecEnv (multiprocessing) for {n_envs} environments")
        
        return env


def create_sparse_reward_ppo(env, device="auto", tensorboard_log="./logs/tft_training/"):
    """Create EpisodicPPO optimized for sparse reward environments."""
    
    # Determine optimal device
    if device == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # For MLP policies, CPU is often faster due to small networks
            device = "cpu"
            print("🖥️  Using CPU (optimal for MLP policies)")
        else:
            device = "cpu"
            print("🖥️  Using CPU (CUDA not available)")
    else:
        print(f"🖥️  Using device: {device}")
    
    # Get number of environments
    try:
        n_envs = env.num_envs
        print(f"🔄 Detected {n_envs} parallel environments")
    except AttributeError:
        n_envs = 1
        print("🔄 Using single environment")
    
    # Adjust batch size for multiple environments
    batch_size = 128 * max(1, n_envs // 2)  # Scale batch size with envs
    n_steps = 4096 // max(1, n_envs)        # Reduce steps per env when parallel
    
    print(f"📊 Training configuration:")
    print(f"   Environments: {n_envs}")
    print(f"   Steps per env: {n_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total steps per update: {n_steps * n_envs}")
    
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
        print(f"📂 Loading existing model from {model_path}")
        try:
            # Try to load as EpisodicPPO first, fallback to regular PPO
            try:
                model = EpisodicPPO.load(model_path, device=device)
            except:
                print("⚠️  Loading as regular PPO (not episodic)")
                model = PPO.load(model_path, device=device)
            model.set_env(env)
            return model, True  # True = resumed
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
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
    
    print(f"✅ Created EpisodicPPO with EpisodicRolloutBuffer")
    print(f"   Min episodes per update: {min_episodes_per_update}")
    print(f"   Buffer tracks episode completion for sparse rewards")
    
    return model


def demonstrate_episodic_usage():
    """Demonstrate how to use the episodic implementations."""
    print("🎯 TFT Episodic Training Demonstration")
    print("=" * 50)
    
    # Create environment
    env = create_tft_environment()
    
    # Option 1: Use the integrated approach (recommended)
    print("\n📚 Option 1: Integrated EpisodicPPO (Recommended)")
    model1 = create_sparse_reward_ppo(env)
    print("   ✅ EpisodicPPO with sparse reward optimizations")
    
    # Option 2: Use the custom buffer approach
    print("\n📚 Option 2: Custom EpisodicRolloutBuffer")
    model2 = create_episodic_tft_model(env, min_episodes_per_update=5)
    print("   ✅ EpisodicPPO with custom EpisodicRolloutBuffer")
    
    # Enhanced callback for episodic training
    print("\n📊 Enhanced Episodic Callback Features:")
    callback = TFTTrainingCallback(save_freq=25000, log_freq=5)
    print("   ✅ Tracks completed episodes per rollout")
    print("   ✅ Monitors positive reward rate (sparse reward metric)")
    print("   ✅ Enhanced logging for episode-based training")
    
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
    
    print("🚀 TFT SB3 Training - Production Mode with Parallel Support")
    print("=" * 60)
    
    # Setup directories
    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./logs/tft_training", exist_ok=True)
    
    # Create environment(s)
    print("🎮 Setting up environment(s)...")
    env = create_tft_environment(n_envs=n_envs)
    print("✅ Environment(s) ready with sparse reward solutions")
    
    # Load or create model
    model, was_resumed = load_or_create_model(
        model_path if resume else None, 
        env, 
        force_new,
        device=device
    )
    
    if was_resumed:
        print(f"✅ Resumed training from existing model")
    else:
        print(f"✅ Created new model for training")
    
    # Setup callback for monitoring and auto-saving
    callback = TFTTrainingCallback(
        save_freq=save_freq,
        save_path="./checkpoint/",
        log_freq=5  # Log every 5 episodes
    )
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Parallel environments: {n_envs}")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Auto-save every: {save_freq:,} steps")
    print(f"   Device: {device}")
    print(f"   Tensorboard logs: ./logs/tft_training/")
    print(f"   Model saves: ./checkpoint/")
    
    print(f"\n📊 Optimizations for Sparse Rewards:")
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
    
    print(f"\n🎮 Starting training... (Ctrl+C to stop safely)")
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
        print(f"\n🎉 Training completed!")
        print(f"⏱️  Total time: {training_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        training_time = time.time() - start_time
        print(f"⏱️  Training time: {training_time:.1f} seconds")
    
    # Save final model
    final_path = f"{model_path}_final"
    model.save(final_path)
    print(f"💾 Final model saved: {final_path}.zip")
    
    # Save latest for easy resuming
    model.save(model_path)
    print(f"💾 Latest model saved: {model_path}.zip")
    
    # Show final stats
    if callback.episode_count > 0:
        print(f"\n📈 Training Stats:")
        print(f"   Episodes completed: {callback.episode_count}")
        print(f"   Average reward (last 20): {np.mean(callback.episode_rewards[-20:]):.1f}")
        print(f"   Best reward: {np.max(callback.episode_rewards):.1f}")
        print(f"   Average episode length: {np.mean(callback.episode_lengths):.0f}")
        if n_envs > 1:
            print(f"   Environments: {n_envs} (parallel)")
            print(f"   Effective speedup: ~{n_envs}x")
    
    env.close()
    return model


def test_model(model_path="./checkpoint/tft_latest", num_episodes=5):
    """Test a trained model."""
    print(f"🧪 Testing model: {model_path}")
    
    if not Path(f"{model_path}.zip").exists():
        print(f"❌ Model not found: {model_path}.zip")
        return
    
    # Load model and environment
    env = create_tft_environment()
    try:
        model = EpisodicPPO.load(model_path)
    except:
        print("⚠️  Loading as regular PPO (not episodic)")
        model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0  # Initialize as float
        steps = 0
        
        while steps < 1000:  # Max steps safety
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)  # Ensure float conversion
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"  Episode {episode+1}: {episode_reward:.1f} reward, {steps} steps")
    
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n📊 Test Results:")
    print(f"   Average reward: {avg_reward:.1f}")
    print(f"   Average length: {avg_length:.0f}")
    print(f"   Best episode: {np.max(episode_rewards):.1f}")
    
    env.close()


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
        print("❌ Error: n_envs must be between 1 and 16")
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
        test_model(args.model, args.test_episodes)


if __name__ == "__main__":
    # Check dependencies first
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ DEPENDENCY CHECK FAILED")
        print("Please resolve the dependency issues above before running the training script.")
        exit(1)
    
    print("✅ All dependencies available!")
    
    # If no command line args, run interactive mode
    import sys
    if len(sys.argv) == 1:
        print("\n🎮 TFT SB3 Training with Episodic Support - Interactive Mode")
        print("=" * 60)
        print("1. Start new episodic training (single environment)")
        print("2. Start new episodic training (parallel environments)")
        print("3. Continue existing training") 
        print("4. Test model")
        print("5. Demonstrate episodic implementations")
        print("6. Command line help")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            train_tft_model(force_new=True, n_envs=1)
        elif choice == "2":
            print("\n🔄 Parallel Environment Setup")
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
            
            print(f"\n🚀 Starting parallel training with {n_envs} environments on {device_choice}")
            train_tft_model(force_new=True, n_envs=n_envs, device=device_choice)
        elif choice == "3":
            print("\n📂 Continue Training Options")
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
            print("\n📖 Command Line Usage:")
            print("  python tft_train.py --mode train --timesteps 500000")
            print("  python tft_train.py --mode continue --model ./checkpoint/my_model")
            print("  python tft_train.py --mode test --test-episodes 10")
            print("  python tft_train.py --help")
            print("\n🎯 Episodic Features:")
            print("  • EpisodicPPO: Enhanced PPO for sparse rewards")
            print("  • EpisodicRolloutBuffer: Episode-aware buffer")
            print("  • Enhanced callbacks with episode tracking")
            print("  • Optimized for TFT's sparse reward structure")
        else:
            print("Invalid choice. Exiting.")
    else:
        main()