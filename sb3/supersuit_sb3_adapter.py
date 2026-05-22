"""
Compatibility wrapper to make SuperSuit environments work with SB3.
SB3 doesn't recognize SuperSuit's MarkovVectorEnv, so we need to adapt it.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from sb3_wrapper import create_sb3_env


class SuperSuitSB3Adapter(VecEnv):
    """
    Adapter that makes SuperSuit's MarkovVectorEnv compatible with SB3.
    
    This wrapper inherits from SB3's VecEnv base class so SB3 recognizes it
    as a proper vectorized environment without trying to wrap it further.
    """
    
    def __init__(self, supersuit_env):
        """
        Initialize the adapter.
        
        Args:
            supersuit_env: SuperSuit MarkovVectorEnv instance
        """
        self.supersuit_env = supersuit_env
        
        # Initialize VecEnv with required parameters
        super().__init__(
            num_envs=supersuit_env.num_envs,
            observation_space=supersuit_env.observation_space,
            action_space=supersuit_env.action_space
        )
        
        # Copy relevant attributes
        self.metadata = getattr(supersuit_env, 'metadata', {})
        self.render_mode = getattr(supersuit_env, 'render_mode', None)
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environments are wrapped by a specific wrapper class."""
        # SuperSuit environments are not wrapped by gymnasium wrappers
        if indices is None:
            return [False] * self.num_envs
        elif isinstance(indices, int):
            return [False]
        else:
            return [False] * len(list(indices))
    
    def reset(self):
        """Reset all environments."""
        result = self.supersuit_env.reset()
        if isinstance(result, tuple):
            obs, infos = result
            return obs
        return result
    
    def step_async(self, actions):
        """Asynchronous step - store actions for later execution."""
        self._actions = actions
    
    def step_wait(self):
        """Execute stored actions and return results."""
        # Handle actions in dict format (from parallel environment)
        actions = self._actions
        
        # Pass actions directly to the parallel environment
        # No conversion needed since we removed vectorization
        result = self.supersuit_env.step(actions)
        
        if len(result) == 5:
            # New gymnasium format: obs, reward, terminated, truncated, info
            obs, rewards, terminated, truncated, infos = result
            # Convert to old format for SB3 compatibility
            dones = terminated  # Keep dict format instead of numpy array
            return obs, rewards, dones, infos
        else:
            # Old format: obs, reward, done, info
            return result
    
    def close(self):
        """Close the environment."""
        return self.supersuit_env.close()
    
    def get_attr(self, attr_name, indices=None):
        """Get attribute from the environment."""
        if hasattr(self.supersuit_env, attr_name):
            attr = getattr(self.supersuit_env, attr_name)
            if indices is None:
                return [attr] * self.num_envs
            else:
                # Handle different types of indices
                if isinstance(indices, int):
                    return [attr]
                else:
                    return [attr] * len(list(indices))
        else:
            raise AttributeError(f"SuperSuit environment does not have attribute '{attr_name}'")
    
    def set_attr(self, attr_name, value, indices=None):
        """Set attribute on the environment."""
        if hasattr(self.supersuit_env, attr_name):
            setattr(self.supersuit_env, attr_name, value)
        else:
            raise AttributeError(f"SuperSuit environment does not have attribute '{attr_name}'")
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call method on the environment."""
        if hasattr(self.supersuit_env, method_name):
            method = getattr(self.supersuit_env, method_name)
            result = method(*method_args, **method_kwargs)
            if indices is None:
                return [result] * self.num_envs
            else:
                # Handle different types of indices
                if isinstance(indices, int):
                    return [result]
                else:
                    return [result] * len(list(indices))
        else:
            raise AttributeError(f"SuperSuit environment does not have method '{method_name}'")
    
    def seed(self, seed=None):
        """Seed the environment."""
        if hasattr(self.supersuit_env, 'seed'):
            return self.supersuit_env.seed(seed)
        else:
            # Fallback for environments without seed method
            return [None] * self.num_envs
    
    def render(self, mode='human'):
        """Render the environment."""
        if hasattr(self.supersuit_env, 'render'):
            return self.supersuit_env.render()
        else:
            return None


def create_sb3_compatible_env(num_parallel_envs: int = 1):
    """
    Create SB3-compatible environment using SuperSuit.
    
    Args:
        num_parallel_envs: Number of parallel environments (limited to 1 due to serialization issues)
    
    Returns:
        SB3-compatible VectorEnv wrapping SuperSuit environment
    """
    # Create SuperSuit environment
    supersuit_env = create_sb3_env(num_parallel_envs=num_parallel_envs)
    
    # Wrap it in our compatibility adapter
    sb3_env = SuperSuitSB3Adapter(supersuit_env)
    
    return sb3_env


if __name__ == "__main__":
    print("Testing SuperSuit SB3 Compatibility Adapter...")
    
    try:
        # Test environment creation
        print("\n1. Creating adapted environment:")
        env = create_sb3_compatible_env(num_parallel_envs=1)
        
        print(f"   ✅ Environment created successfully")
        print(f"   Environment type: {type(env)}")
        print(f"   Is SB3 VecEnv: {isinstance(env, VecEnv)}")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        print(f"   Number of environments: {env.num_envs}")
        
        # Test SB3 integration
        print("\n2. Testing SB3 PPO integration:")
        from stable_baselines3 import PPO
        
        try:
            model = PPO('MlpPolicy', env, verbose=1, n_steps=512)
            print("   ✅ PPO creation successful!")
            
            # Test learning for a few steps
            print("\n3. Testing short training run:")
            model.learn(total_timesteps=1000)
            print("   ✅ Short training successful!")
            
        except Exception as e:
            print(f"   ❌ SB3 integration failed: {e}")
            import traceback
            traceback.print_exc()
        
        env.close()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Compatibility testing completed!")