import numpy as np
import config
from tft_set4_gym import utils
from tft_set4_gym.stats import COST

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
    
    def count_units_needed_for_three_star(self, unit_counts):
        """
        Calculate how many units are needed to reach 3-star (9 total units).
        
        Args:
            unit_counts: dict with unit names as keys and counts as values
            
        Returns:
            dict with unit names as keys and units needed for 3-star as values
        """
        units_needed = {}
        for unit_name, count in unit_counts.items():
            # Need 9 total units for 3-star (1 + 3 + 9 pattern for 1*, 2*, 3*)
            units_needed[unit_name] = max(0, 9 - count)
        return units_needed
    
    def get_unit_counts(self, observation):
        """
        Count all units of each type on board and bench.
        
        Returns:
            dict with unit names as keys and total counts as values
        """
        unit_counts = {}
        
        # Count board units
        board = utils.board_from_obs(observation)
        for unit in board:
            name = unit["name"]
            # Count by star level - each star level represents cumulative units
            # 1-star = 1 unit, 2-star = 3 units (1 + 2), 3-star = 9 units (1 + 2 + 6)
            stars = int(unit["stars"])
            if stars == 1:
                unit_count = 1
            elif stars == 2:
                unit_count = 3
            elif stars == 3:
                unit_count = 9
            else:
                unit_count = 1  # fallback
                
            unit_counts[name] = unit_counts.get(name, 0) + unit_count
        
        # Count bench units (these are always 1-star)
        bench = utils.bench_from_obs(observation)
        for unit_name in bench:
            if unit_name:  # bench can have None entries
                unit_counts[unit_name] = unit_counts.get(unit_name, 0) + 1
                
        return unit_counts
    
    def is_board_and_bench_full(self, observation):
        """Check if both board and bench are full."""
        try:
            board = utils.board_from_obs(observation)
            bench = utils.bench_from_obs(observation)
            level = utils.level_from_obs(observation)
            
            # Board is full if we have max units for our level
            board_full = len(board) >= int(level)
            
            # Bench is full if all 9 slots are occupied
            bench_full = len([unit for unit in bench if unit]) >= 9
            
            return board_full and bench_full
        except:
            return False
    
    def find_lowest_priority_unit_to_sell(self, observation):
        """
        Find the lowest priority unit to sell when board and bench are full.
        Priority based on how close to 3-star (fewer units needed = higher priority).
        Won't sell if we only have 1 copy of a unit.
        
        Returns:
            tuple: (action_type, position) or None if no unit should be sold
        """
        try:
            unit_counts = self.get_unit_counts(observation)
            units_needed = self.count_units_needed_for_three_star(unit_counts)
            
            # Get board and bench units with their positions
            board = utils.board_from_obs(observation)
            bench = utils.bench_from_obs(observation)
            
            candidates = []
            
            # Check board units
            for unit in board:
                name = unit["name"]
                if name in self.units_to_buy and unit_counts.get(name, 0) > 1:
                    pos = utils.x_y_to_1d_coord(unit["pos_x"], unit["pos_y"])
                    priority = units_needed.get(name, 9)  # Lower number = higher priority
                    candidates.append((priority, 3, pos))  # action_type=3 for sell
            
            # Check bench units
            for i, unit_name in enumerate(bench):
                if unit_name and unit_name in self.units_to_buy and unit_counts.get(unit_name, 0) > 1:
                    pos = 28 + i  # bench positions start at 28
                    priority = units_needed.get(unit_name, 9)
                    candidates.append((priority, 3, pos))
            
            # Sort by priority (highest priority value = lowest priority to keep)
            # We want to sell the unit that needs the MOST additional units (highest priority value)
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, action_type, position = candidates[0]
                return [action_type, position, 0]
                
        except Exception as e:
            pass
            
        return None
        
    def decide_action(self, observation):
        """Core decision logic for buying agents with improved unit management."""
        try:
            gold = utils.gold_from_obs(observation)
            units_in_shop, chosen = utils.units_in_shop_from_obs(observation)
            level = utils.level_from_obs(observation)
            
            # Check if board and bench are full before trying to buy
            board_and_bench_full = self.is_board_and_bench_full(observation)
            
            # If board and bench are full, sell lowest priority unit first
            if board_and_bench_full:
                sell_action = self.find_lowest_priority_unit_to_sell(observation)
                if sell_action:
                    return sell_action
            
            # Try to buy desired units from shop (prioritize by cost for now)
            shop_priorities = []
            for i, champ in enumerate(units_in_shop):
                if champ in self.units_to_buy:
                    try:
                        cost = COST.get(champ, 5)  # fallback to cost 5
                        if gold >= cost:
                            shop_priorities.append((cost, i, champ))
                    except:
                        # Fallback if cost lookup fails
                        if gold >= 5:  # assume reasonable minimum cost
                            shop_priorities.append((5, i, champ))
            
            # Buy cheapest desired unit first (or most expensive if prioritizing high-cost)
            if shop_priorities:
                shop_priorities.sort(key=lambda x: x[0])  # Sort by cost, cheapest first
                cost, shop_index, champ_name = shop_priorities[0]
                champ_id = utils.champ_id_from_name(champ_name)
                return [2, champ_id, 0]  # action_type=2 for buy
            
            # Sell units not in our buying list
            board = utils.board_from_obs(observation)
            for unit in board:
                if unit["name"] not in self.units_to_buy:
                    pos = utils.x_y_to_1d_coord(unit["pos_x"], unit["pos_y"])
                    return [3, pos, 0]
            
            # Level up logic
            if gold > 54.0 and level < 8.0:
                return [5, 0, 0]  # action_type=5 for level up
            
            # Refresh logic  
            if (level >= 6.0 or len(board) >= level) and gold > 52.0:
                return [4, 0, 0]  # action_type=4 for refresh
            
            # Sell units on bench not in directive
            bench = utils.bench_from_obs(observation)
            for i, unit_name in enumerate(bench):
                if unit_name and unit_name not in self.units_to_buy:
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
