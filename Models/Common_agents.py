import numpy as np
import config
from tft_set4_gym import utils

def preprocess_observation(observation):
    """
    Preprocess observation to the format expected by utils functions.
    
    Args:
        observation: Either a dict with 'tensor' key (from gym env) or raw array
        
    Returns:
        numpy array with shape (184, 4, 7) suitable for utils functions
    """
    try:
        # Handle dict observation format from gym environment
        if isinstance(observation, dict):
            if 'tensor' in observation:
                obs_tensor = observation['tensor']
            else:
                # Fallback to observation itself if no 'tensor' key
                obs_tensor = observation
        else:
            obs_tensor = observation
        
        # Convert to numpy array if needed
        if not isinstance(obs_tensor, np.ndarray):
            obs_tensor = np.array(obs_tensor)
        
        # Check if reshaping is needed
        if obs_tensor.shape == (5152,):
            # Reshape from flat gym observation to utils format
            reshaped_obs = obs_tensor.reshape(184, 4, 7)
            return reshaped_obs
        elif obs_tensor.shape == (184, 4, 7):
            # Already in correct format
            return obs_tensor
        else:
            # Try to determine correct shape based on total elements
            total_elements = obs_tensor.size
            if total_elements == 5152:  # 184 * 4 * 7
                return obs_tensor.reshape(184, 4, 7)
            else:
                # Unknown format, return as-is and hope for the best
                return obs_tensor
                
    except Exception as e:
        # If all else fails, return the observation as-is
        return observation

class RandomAgent:
    def __init__(self, agent_name="RandomAgent"):
        self.agent_type = agent_name

    def select_action(self, observation, action_mask):
        """Select a random valid action."""
        return [np.random.randint(0, 6), np.random.randint(0, 37), np.random.randint(0, 28)]

class BuyingAgent:
    def __init__(self, units_to_buy, agent_name="BuyingAgent"):
        self.units_to_buy = units_to_buy
        self.agent_type = agent_name

    def select_action(self, observation, action_mask):
        """Select action based on buying strategy."""
        try:
            # Preprocess observation to correct format for utils functions
            processed_obs = preprocess_observation(observation)
            return self.decide_action(processed_obs)
        except Exception as e:
            # Fallback to random action if observation parsing fails
            return [np.random.randint(0, 6), np.random.randint(0, 37), np.random.randint(0, 28)]
        
    def decide_action(self, observation):
        """Core decision logic for buying agents."""
        try:
            gold = utils.gold_from_obs(observation)
            units_in_shop, chosen = utils.units_in_shop_from_obs(observation)
            
            # Try to buy desired units from shop
            for champ in units_in_shop:
                if champ in self.units_to_buy and gold >= 10:  # Assume min cost
                    action = [2, utils.champ_id_from_name(champ), 0]
                    return action
            
            # Sell units not in our buying list
            board = utils.board_from_obs(observation)
            for champ in board:
                if champ["name"] not in self.units_to_buy:
                    action = [3, utils.x_y_to_1d_coord(champ["pos_x"], champ["pos_y"]), 0]
                    return action
            
            # Level up or refresh logic
            level = utils.level_from_obs(observation)
            if (level >= 8.0 or len(board) < level) and gold > 52.0:
                return [4, 0, 0]  # Refresh
            
            if gold > 54.0 and level < 8.0:
                return [5, 0, 0]  # Level up
            
            # Sell units on bench not in directive
            bench = utils.bench_from_obs(observation)
            for i, champ in enumerate(bench):
                if champ not in self.units_to_buy:
                    return [3, 28+i, 0]
                    
        except Exception as e:
            # If any parsing fails, default action
            pass
            
        return [0, 0, 0]  # Default: do nothing

class CultistAgent(BuyingAgent):
    def __init__(self):
        cultist_units = ["elise", "twistedfate", "pyke", "evelynn", "aatrox", "zilean", "kalista", "jhin"]
        super().__init__(cultist_units, "CultistAgent")

class DivineAgent(BuyingAgent):
    def __init__(self):
        divine_units = ["wukong", "jax", "irelia", "lux", "warwick", "leesin", "ashe", "kindred", "teemo"]
        super().__init__(divine_units, "DivineAgent")

class RerollAgent(BuyingAgent):
    def __init__(self):
        reroll_units = ["yasuo", "fiora", "vayne", "nidalee", "garen"]  # Low cost reroll units
        super().__init__(reroll_units, "RerollAgent")
        
    def decide_action(self, observation):
        """Reroll strategy focuses on low-cost units and frequent refreshing."""
        try:
            gold = utils.gold_from_obs(observation)
            units_in_shop, chosen = utils.units_in_shop_from_obs(observation)
            
            # Prioritize buying our target units
            for champ in units_in_shop:
                if champ in self.units_to_buy and gold >= 5:  # Lower gold threshold for reroll
                    action = [2, utils.champ_id_from_name(champ), 0]
                    return action
            
            # More aggressive refreshing for reroll strategy
            level = utils.level_from_obs(observation)
            if gold > 30.0 and level <= 6:  # Refresh more often at lower levels
                return [4, 0, 0]
                
            # Don't level up as much - stay low level for better reroll odds
            if gold > 60.0 and level < 6:
                return [5, 0, 0]
                
        except Exception as e:
            pass
            
        return super().decide_action(observation)

class FastLevelAgent:
    def __init__(self):
        self.agent_type = "FastLevelAgent"
        
    def select_action(self, observation, action_mask):
        """Strategy focused on fast leveling and strongest board."""
        try:
            # Preprocess observation to correct format for utils functions
            processed_obs = preprocess_observation(observation)
                
            gold = utils.gold_from_obs(processed_obs)
            level = utils.level_from_obs(processed_obs)
            
            # Prioritize leveling up quickly
            if gold > 40.0 and level < 8:
                return [5, 0, 0]  # Level up
                
            # Buy any decent units when we have excess gold
            units_in_shop, chosen = utils.units_in_shop_from_obs(processed_obs)
            if gold > 60.0 and len(units_in_shop) > 0:
                # Buy first available unit
                champ = units_in_shop[0]
                return [2, utils.champ_id_from_name(champ), 0]
                
            # Refresh when we have lots of gold and high level
            if gold > 70.0 and level >= 7:
                return [4, 0, 0]
                
        except Exception as e:
            pass
            
        return [0, 0, 0]  # Default: do nothing
