import numpy as np
import sys
import os

# Add parent directory to access config and TFTSet4Gym
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config

from TFTSet4Gym.tft_set4_gym.stats import COST
from TFTSet4Gym.tft_set4_gym.observation_builder import get_field_value_from_obs
from TFTSet4Gym.tft_set4_gym.observation_schema import get_observation_schema
from Models.replay_buffer import ReplayBuffer


def extract_field_from_observation(observation, field_name):
    """
    Extract a specific field from an observation using the new schema system.
    
    Args:
        observation: The observation tensor (numpy array) or dictionary from parallel_env
        field_name: Name of the field to extract
        
    Returns:
        The extracted field value
    """
    
    return get_field_value_from_obs(observation, field_name)


def get_board_units_from_observation(observation):
    """Extract board units using new schema system."""
    # Handle dictionary observations from parallel_env
    if isinstance(observation, dict):
        observation = observation.get('tensor', observation)
    
    # Use new schema-based extraction
    board_champions = extract_field_from_observation(observation, 'board_champions')
    board_stars = extract_field_from_observation(observation, 'board_stars')
    board_chosen = extract_field_from_observation(observation, 'board_chosen')
    
    if board_champions is not None and board_stars is not None:
        return _parse_board_from_fields(board_champions, board_stars, board_chosen)
    
    return []


def get_bench_units_from_observation(observation):
    """Extract bench units using new schema system."""
    # Handle dictionary observations from parallel_env
    if isinstance(observation, dict):
        # Extract the actual observation tensor from the dictionary
        # parallel_env returns observations as {'tensor': array, 'action_mask': array}
        observation = observation.get('tensor', observation)
    
    bench_champions = extract_field_from_observation(observation, 'bench_champions')
    if bench_champions is not None:
        return _parse_bench_from_field(bench_champions)
    
    return []


def get_shop_units_from_observation(observation):
    """Extract shop units using new schema system."""
    # Handle dictionary observations from parallel_env
    if isinstance(observation, dict):
        # Extract the actual observation tensor from the dictionary
        # parallel_env returns observations as {'tensor': array, 'action_mask': array}
        observation = observation.get('tensor', observation)
    
    shop_champions = extract_field_from_observation(observation, 'shop_champions')
    shop_chosen = extract_field_from_observation(observation, 'shop_chosen')
    if shop_champions is not None:
        return _parse_shop_from_fields(shop_champions, shop_chosen)
    
    return [" "] * 5


def _parse_board_from_fields(board_champions, board_stars, board_chosen):
    """Parse board units from schema fields."""
    champs = []
    if board_champions.ndim == 3:  # (58, 4, 7) format
        for i, unit_board in enumerate(board_champions):
            indexes = np.where(unit_board == 1.0)
            if len(indexes[0]) > 0 and len(indexes[1]) > 0:
                champion_name = list(COST.keys())[i + 1] if i + 1 < len(COST.keys()) else f"unknown_{i}"
                
                stars = 1
                chosen = False
                if board_stars is not None and len(indexes[0]) > 0:
                    stars = board_stars[indexes[0][0], indexes[1][0]] if board_stars.ndim >= 2 else 1
                if board_chosen is not None and len(indexes[0]) > 0:
                    chosen = board_chosen[indexes[0][0], indexes[1][0]] > 0.5 if board_chosen.ndim >= 2 else False
                
                champ = {
                    "name": champion_name,
                    "id": i,
                    "pos_y": int(indexes[0][0]),
                    "pos_x": int(indexes[1][0]),
                    "stars": int(stars),
                    "chosen": bool(chosen)
                }
                champs.append(champ)
    return champs


def _parse_bench_from_field(bench_champions):
    """Parse bench units from schema field."""
    bench_list = []
    if bench_champions.ndim == 3:  # (58, 4, 7) format - use first position for count
        bench = bench_champions[:, 0, 0]
        for i, count in enumerate(bench):
            if count > 0:
                champion_name = list(COST.keys())[i + 1] if i + 1 < len(COST.keys()) else f"unknown_{i}"
                for _ in range(int(count)):
                    bench_list.append(champion_name)
    return bench_list


def _parse_shop_from_fields(shop_champions, shop_chosen):
    """Parse shop units from schema fields."""
    shop_units = [" "] * 5
    if shop_champions.ndim == 3:  # (58, 4, 7) format
        for slot in range(min(5, shop_champions.shape[2])):
            for i, unit_data in enumerate(shop_champions):
                if unit_data[0, slot] > 0:  # Check if unit is in this shop slot
                    champion_name = list(COST.keys())[i + 1] if i + 1 < len(COST.keys()) else f"unknown_{i}"
                    
                    # Check for chosen status with array-safe comparison and bounds checking
                    if shop_chosen is not None:
                        try:
                            # Make sure we don't go out of bounds
                            if slot < shop_chosen.shape[1]:
                                chosen_value = shop_chosen[0, slot]
                                # Handle scalar or array values safely
                                if hasattr(chosen_value, 'item'):
                                    if chosen_value.size == 1:
                                        chosen_value = chosen_value.item()
                                    else:
                                        chosen_value = chosen_value.flat[0] if chosen_value.size > 0 else 0
                                if chosen_value > 0.5:
                                    champion_name += "_c"
                        except (ValueError, TypeError, IndexError):
                            # If comparison fails, skip chosen status
                            pass
                    
                    shop_units[slot] = champion_name
                    break
    return shop_units

class BaseAgent:
    """Base class for all TFT agents providing common interface and utilities."""
    
    def __init__(self, agent_name="BaseAgent", global_buffer=None):
        self.agent_type = agent_name
        # Create local replay buffer that points to the global buffer
        if global_buffer is not None:
            self.replay_buffer = ReplayBuffer(global_buffer)
        else:
            self.replay_buffer = None
    
    def select_action(self, observation, action_mask=None, reward=None, terminated=None):
        """
        Select an action based on the current observation and action mask.
        
        Args:
            observation: The current game state observation
            action_mask: Valid actions that can be taken
            reward: Reward signal (optional)
            terminated: Termination flags (optional)
            
        Returns:
            list: Action in format [action_type, target, value]
        """
        # If observation is a dict (standard TFTSet4Gym format), extract tensor and mask
        if isinstance(observation, dict) and 'tensor' in observation:
            obs = observation['tensor']
            mask = observation.get('action_mask', action_mask)
        else:
            obs = observation
            mask = action_mask
        
        # Flatten to 1D array if needed (schema expects flat observation)
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs = obs.flatten()
            
        try:
            action = self._select_action_impl(obs, mask, reward, terminated)
        except Exception:
            action = [0, 0, 0]

        return action
    
    def batch_select_action(self, observations, masks, precomputed_results=None):
        """
        Select actions for a batch of observations.
        
        Default implementation iteratively calls self.select_action for each item.
        Subclasses with native batched inference (e.g. MuZeroAgent) may override.
        
        Args:
            observations: List of observations
            masks: List of action masks
            precomputed_results: Optional list of precomputed results per item
            
        Returns:
            List of actions
        """
        actions = []
        for i, obs in enumerate(observations):
            mask = masks[i] if i < len(masks) else None
            action = self.select_action(obs, mask)
            actions.append(action)
        return actions
    
    def _store_experience(self, observation=None, policy=None, value=0, reward=0, terminated=False, action=None):
        if self.replay_buffer is not None:
            self.replay_buffer.store_step(observation=observation, policy=policy, value=value, reward=reward, action=action)

    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None):
        """
        Implementation method that subclasses should override.
        This separates the core action selection from replay buffer management.
        """
        raise NotImplementedError("Subclasses must implement _select_action_impl")
    
    def terminate(self, final_value):
        """
        Handle episode termination: add final value and send data to global storage.
        
        Args:
            final_value: The final value for the episode
        """
        if self.replay_buffer is not None:
            self.replay_buffer.move_buffer_to_global(final_value=final_value)
            self.replay_buffer.reset()
    
    def _get_champion_id(self, champ_name):
        """Get champion ID from name using COST dictionary."""
        champ_names = list(COST.keys())
        if champ_name in champ_names:
            return champ_names.index(champ_name) - 1
        return 0  # Default fallback

class RandomAgent(BaseAgent):
    def __init__(self, agent_name="RandomAgent", global_buffer=None):
        super().__init__(agent_name, global_buffer)

    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None):
        """Select a random valid action."""
        return [np.random.randint(0, 6), np.random.randint(0, 37), np.random.randint(0, 28)]

class BuyingAgent(BaseAgent):
    def __init__(self, units_to_buy, agent_name="BuyingAgent", global_buffer=None):
        super().__init__(agent_name, global_buffer)
        self.units_to_buy = units_to_buy

    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None):
        """Select action based on buying strategy."""
        gold = extract_field_from_observation(obs, 'gold')
        units_in_shop = get_shop_units_from_observation(obs)
        level = extract_field_from_observation(obs, 'level')

        # Check if board and bench are full before trying to buy
        board_and_bench_full = self.is_board_and_bench_full(obs)

        # If board and bench are full, sell lowest priority unit first
        if board_and_bench_full:
            sell_action = self.find_lowest_priority_unit_to_sell(obs)
            if sell_action:
                return sell_action
        
        # Try to buy desired units from shop (prioritize by cost for now)
        shop_priorities = []
        for i, champ in enumerate(units_in_shop):
            if champ in self.units_to_buy:
                cost = COST.get(champ, 5)  # fallback to cost 5
                if gold >= cost:
                    shop_priorities.append((cost, i, champ))
        
        # Buy cheapest desired unit first (or most expensive if prioritizing high-cost)
        if shop_priorities:
            shop_priorities.sort(key=lambda x: x[0])  # Sort by cost, cheapest first
            cost, shop_index, champ_name = shop_priorities[0]
            champ_id = self._get_champion_id(champ_name)
            return [2, champ_id, 0]  # action_type=2 for buy
        
        # Sell units not in our buying list
        board = get_board_units_from_observation(obs)
        for unit in board:
            unit_name = unit.get("name", "")
            if unit_name not in self.units_to_buy:
                pos = unit["pos_y"] * 7 + unit["pos_x"]  # Convert x,y to position
                return [3, pos, 0]
        
        # Level up logic
        if gold > 54.0 and level < 8.0:
            return [5, 0, 0]  # action_type=5 for level up
        
        
        # Refresh logic  
        if (level >= 6.0 or len(board) >= level) and gold > 52.0:
            return [4, 0, 0]  # action_type=4 for refresh
        
        # Sell units on bench not in directive
        bench = get_bench_units_from_observation(obs)
        for i, unit_name in enumerate(bench):
            if unit_name and unit_name != " " and unit_name not in self.units_to_buy:
                return [3, 28+i, 0]
                
        return [0, 0, 0]  # Default: do nothing
    
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
        board = get_board_units_from_observation(observation)
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
        bench = get_bench_units_from_observation(observation)
        for unit_name in bench:
            if unit_name and unit_name != " ":  # bench can have None entries
                unit_counts[unit_name] = unit_counts.get(unit_name, 0) + 1
                
        return unit_counts
    
    def is_board_and_bench_full(self, observation):
        """Check if both board and bench are full."""
        board = get_board_units_from_observation(observation)
        bench = get_bench_units_from_observation(observation)
        level = extract_field_from_observation(observation, 'level')
        
        # Board is full if we have max units for our level
        board_full = len(board) >= int(level)
        
        # Bench is full if all 9 slots are occupied
        bench_full = len([unit for unit in bench if unit and unit != " "]) >= 9
        
        return board_full and bench_full
    
    def find_lowest_priority_unit_to_sell(self, observation):
        """
        Find the lowest priority unit to sell when board and bench are full.
        Priority based on how close to 3-star (fewer units needed = higher priority).
        Won't sell if we only have 1 copy of a unit.
        
        Returns:
            tuple: (action_type, position) or None if no unit should be sold
        """
        unit_counts = self.get_unit_counts(observation)
        units_needed = self.count_units_needed_for_three_star(unit_counts)
        
        # Get board and bench units with their positions
        board = get_board_units_from_observation(observation)
        bench = get_bench_units_from_observation(observation)
        
        candidates = []
        
        # Check board units
        for unit in board:
            name = unit["name"]
            if name in self.units_to_buy and unit_counts.get(name, 0) > 1:
                # Convert x,y to simple position (row * 7 + col)
                pos = unit["pos_y"] * 7 + unit["pos_x"]
                priority = units_needed.get(name, 9)  # Lower number = higher priority
                candidates.append((priority, 3, pos))  # action_type=3 for sell
        
        # Check bench units
        for i, unit_name in enumerate(bench):
            if unit_name and unit_name != " " and unit_name in self.units_to_buy and unit_counts.get(unit_name, 0) > 1:
                pos = 28 + i  # bench positions start at 28
                priority = units_needed.get(unit_name, 9)
                candidates.append((priority, 3, pos))
        
        # Sort by priority (highest priority value = lowest priority to keep)
        # We want to sell the unit that needs the MOST additional units (highest priority value)
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, action_type, position = candidates[0]
            return [action_type, position, 0]
            
        return None

class CultistAgent(BuyingAgent):
    def __init__(self, global_buffer=None):
        cultist_units = ["elise", "twistedfate", "pyke", "evelynn", "aatrox", "zilean", "kalista", "jhin"]
        super().__init__(cultist_units, "CultistAgent", global_buffer)

class DivineAgent(BuyingAgent):
    def __init__(self, global_buffer=None):
        divine_units = ["wukong", "jax", "irelia", "lux", "warwick", "leesin", "ashe", "kindred", "teemo"]
        super().__init__(divine_units, "DivineAgent", global_buffer)

class RerollAgent(BuyingAgent):
    def __init__(self, global_buffer=None):
        reroll_units = ["yasuo", "fiora", "vayne", "nidalee", "garen"]  # Low cost reroll units
        super().__init__(reroll_units, "RerollAgent", global_buffer)
        
    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None):
        """Reroll strategy focuses on low-cost units and frequent refreshing."""
        gold = extract_field_from_observation(obs, 'gold')
        units_in_shop = get_shop_units_from_observation(obs)
        
        # Prioritize buying our target units
        for champ in units_in_shop:
            if champ in self.units_to_buy and gold >= 5:  # Lower gold threshold for reroll
                action = [2, self._get_champion_id(champ), 0]
                return action
        
        # More aggressive refreshing for reroll strategy
        level = extract_field_from_observation(obs, 'level')
            
        if gold > 30.0 and level <= 6:  # Refresh more often at lower levels
            return [4, 0, 0]
            
        # Don't level up as much - stay low level for better reroll odds
        if gold > 60.0 and level < 6:
            return [5, 0, 0]
            
        return super()._select_action_impl(obs, action_mask)

class FastLevelAgent(BaseAgent):
    def __init__(self, global_buffer=None):
        super().__init__("FastLevelAgent", global_buffer)
        
    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None):
        """Strategy focused on fast leveling and strongest board."""
        # Observation is already in the correct schema format
            
        gold = extract_field_from_observation(obs, 'gold')
        level = extract_field_from_observation(obs, 'level')
        
        # Prioritize leveling up quickly
        if gold > 40.0 and level < 8:
            return [5, 0, 0]  # Level up
            
        # Buy any decent units when we have excess gold
        units_in_shop = get_shop_units_from_observation(obs)
        if gold > 60.0 and len(units_in_shop) > 0:
            # Buy first available unit
            champ = units_in_shop[0]
            champ_id = self._get_champion_id(champ)
            return [2, champ_id, 0]
            
        # Refresh when we have lots of gold and high level
        if gold > 70.0 and level >= 7:
            return [4, 0, 0]
            
        return [0, 0, 0]  # Default: do nothing
