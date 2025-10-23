"""
TFT SB3 Training Script - Production Ready

Combined script with sparse reward solutions, resumable training, and clean output.
Supports both new training and continuing from checkpoints.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from sb3_wrapper import TFTSingleAgentWrapper
import os
import time
import argparse
from pathlib import Path


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
    
    def __init__(self, save_freq=10000, save_path="./models/", log_freq=10):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_save_step = 0
        
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
        
        # Track episode completion
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if info.get('episode'):
                    self.episode_count += 1
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # Log progress
                    if self.episode_count % self.log_freq == 0:
                        recent_rewards = self.episode_rewards[-self.log_freq:]
                        recent_lengths = self.episode_lengths[-self.log_freq:]
                        
                        avg_reward = np.mean(recent_rewards)
                        avg_length = np.mean(recent_lengths)
                        max_reward = np.max(recent_rewards)
                        
                        print(f"Episode {self.episode_count:4d} | "
                              f"Avg Reward: {avg_reward:6.1f} | "
                              f"Max Reward: {max_reward:6.1f} | "
                              f"Avg Length: {avg_length:5.0f} | "
                              f"Steps: {self.num_timesteps}")
        
        return True


def create_tft_environment():
    """Create TFT environment with sparse reward solutions."""
    base_env = TFTSingleAgentWrapper()
    env = TFTSparseRewardWrapper(base_env)
    env = Monitor(env)
    return env


def create_sparse_reward_ppo(env, tensorboard_log="./logs/tft_training/"):
    """Create PPO optimized for sparse reward environments."""
    return PPO(
        'MlpPolicy',
        env,
        verbose=1,
        
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


def load_or_create_model(model_path, env, force_new=False):
    """Load existing model or create new one."""
    model_file = Path(model_path)
    
    if model_file.exists() and not force_new:
        print(f"📂 Loading existing model from {model_path}")
        try:
            model = PPO.load(model_path)
            model.set_env(env)
            return model, True  # True = resumed
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("🆕 Creating new model instead...")
    
    print("🆕 Creating new PPO model...")
    model = create_sparse_reward_ppo(env)
    return model, False  # False = new model


def train_tft_model(
    model_path="./models/tft_latest",
    total_timesteps=1000000,
    save_freq=25000,
    resume=True,
    force_new=False
):
    """
    Main training function.
    
    Args:
        model_path: Path to save/load model
        total_timesteps: Total training steps (can be increased for continuing)
        save_freq: How often to auto-save (steps)
        resume: Whether to try resuming from existing model
        force_new: Force create new model even if one exists
    """
    
    print("🚀 TFT SB3 Training - Production Mode")
    print("=" * 50)
    
    # Setup directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs/tft_training", exist_ok=True)
    
    # Create environment
    print("🎮 Setting up environment...")
    env = create_tft_environment()
    print("✅ Environment ready with sparse reward solutions")
    
    # Load or create model
    model, was_resumed = load_or_create_model(
        model_path if resume else None, 
        env, 
        force_new
    )
    
    if was_resumed:
        print(f"✅ Resumed training from existing model")
    else:
        print(f"✅ Created new model for training")
    
    # Setup callback for monitoring and auto-saving
    callback = TFTTrainingCallback(
        save_freq=save_freq,
        save_path="./models/",
        log_freq=5  # Log every 5 episodes
    )
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Auto-save every: {save_freq:,} steps")
    print(f"   Tensorboard logs: ./logs/tft_training/")
    print(f"   Model saves: ./models/")
    
    print(f"\n📊 Optimizations for Sparse Rewards:")
    print(f"   • Larger rollouts: 4096 steps")
    print(f"   • More training epochs: 15")
    print(f"   • Higher exploration: 0.02 entropy")
    print(f"   • Survival bonus: +0.01 per step")
    print(f"   • Scaled final rewards: 2x + survival bonus")
    
    print(f"\n🎮 Starting training... (Ctrl+C to stop safely)")
    print("=" * 50)
    
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
    
    env.close()
    return model


def test_model(model_path="./models/tft_latest", num_episodes=5):
    """Test a trained model."""
    print(f"🧪 Testing model: {model_path}")
    
    if not Path(f"{model_path}.zip").exists():
        print(f"❌ Model not found: {model_path}.zip")
        return
    
    # Load model and environment
    env = create_tft_environment()
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 1000:  # Max steps safety
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
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
    parser = argparse.ArgumentParser(description="TFT SB3 Training Script")
    parser.add_argument("--mode", choices=["train", "test", "continue"], default="train",
                       help="Training mode")
    parser.add_argument("--model", default="./models/tft_latest", 
                       help="Model path")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total timesteps for training")
    parser.add_argument("--save-freq", type=int, default=25000,
                       help="Auto-save frequency")
    parser.add_argument("--new", action="store_true",
                       help="Force create new model")
    parser.add_argument("--test-episodes", type=int, default=5,
                       help="Number of episodes for testing")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_tft_model(
            model_path=args.model,
            total_timesteps=args.timesteps,
            save_freq=args.save_freq,
            resume=False,
            force_new=args.new
        )
    elif args.mode == "continue":
        train_tft_model(
            model_path=args.model,
            total_timesteps=args.timesteps,
            save_freq=args.save_freq,
            resume=True,
            force_new=False
        )
    elif args.mode == "test":
        test_model(args.model, args.test_episodes)


if __name__ == "__main__":
    # If no command line args, run interactive mode
    import sys
    if len(sys.argv) == 1:
        print("🎮 TFT SB3 Training - Interactive Mode")
        print("=" * 40)
        print("1. Start new training")
        print("2. Continue existing training") 
        print("3. Test model")
        print("4. Command line help")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            train_tft_model(force_new=True)
        elif choice == "2":
            train_tft_model(resume=True)
        elif choice == "3":
            test_model()
        elif choice == "4":
            print("\n📖 Command Line Usage:")
            print("  python tft_train.py --mode train --timesteps 500000")
            print("  python tft_train.py --mode continue --model ./models/my_model")
            print("  python tft_train.py --mode test --test-episodes 10")
            print("  python tft_train.py --help")
        else:
            print("Invalid choice. Exiting.")
    else:
        main()