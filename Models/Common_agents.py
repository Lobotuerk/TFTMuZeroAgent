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
    """
    if isinstance(observation, dict) and field_name in observation:
        val = observation[field_name]
    else:
        val = get_field_value_from_obs(observation, field_name)
    
    # Normalize: squeeze any leading dimensions of size 1 (batch dimension)
    if isinstance(val, np.ndarray) and val.ndim > 0 and val.shape[0] == 1:
        val = np.squeeze(val, axis=0)
    
    # Handle tiled scalar fields by returning proper scalar values
    if field_name in ['gold', 'level', 'health', 'turns_for_combat']:
        if isinstance(val, np.ndarray):
            if val.size > 0:
                return val.flat[0]
            else:
                # If field is missing or empty, it should probably be an error now
                raise ValueError(f"Field {field_name} is empty in observation")
    
    return val


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
        observation = observation.get('tensor', observation)
    
    bench_champions = extract_field_from_observation(observation, 'bench_champions')
    if bench_champions is not None:
        return _parse_bench_from_field(bench_champions)
    
    return []


def get_shop_units_from_observation(observation):
    """Extract shop units using new schema system."""
    # Handle dictionary observations from parallel_env
    if isinstance(observation, dict):
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
                if i + 1 >= len(COST.keys()):
                    raise ValueError(f"Unknown champion index on board: {i+1}")
                champion_name = list(COST.keys())[i + 1]
                
                stars = 1
                chosen = False
                if board_stars is not None and len(indexes[0]) > 0:
                    val = board_stars[indexes[0][0], indexes[1][0]] if board_stars.ndim >= 2 else 1
                    stars = np.ravel(val)[0] if isinstance(val, np.ndarray) else val
                if board_chosen is not None and len(indexes[0]) > 0:
                    val = board_chosen[indexes[0][0], indexes[1][0]] if board_chosen.ndim >= 2 else False
                    chosen = (np.ravel(val)[0] if isinstance(val, np.ndarray) else val) > 0.5
                
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
                if i + 1 >= len(COST.keys()):
                    raise ValueError(f"Unknown champion index on bench: {i+1}")
                champion_name = list(COST.keys())[i + 1]
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
                    if i + 1 >= len(COST.keys()):
                        raise ValueError(f"Unknown champion index in shop: {i+1}")
                    champion_name = list(COST.keys())[i + 1]
                    
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
                            # Fail explicitly on unexpected data format
                            raise
                    
                    shop_units[slot] = champion_name
                    break
    return shop_units

class BaseAgent:
    """Base class for all TFT agents providing common interface and utilities."""
    
    def __init__(self, agent_name="BaseAgent", global_buffer=None, save_data=False):
        self.agent_type = agent_name
        # Always save data if a global buffer is assigned
        self.save_data = save_data or (global_buffer is not None)
        self.global_buffer = global_buffer

        # State tracking for combat detection (per player)
        self.prev_turns_for_combat = {} # player_id -> value
        self.prev_health = {}           # player_id -> value
        self.prev_observation = {}      # player_id -> value

        # Create local replay buffers that point to the global buffer
        # player_id -> ReplayBuffer
        self.replay_buffers = {}

    def _get_buffer(self, player_id):
        """Get or create a replay buffer for a specific player."""
        if player_id not in self.replay_buffers:
            if self.global_buffer is not None:
                from Models.replay_buffer import ReplayBuffer
                self.replay_buffers[player_id] = ReplayBuffer(self.global_buffer)
            else:
                self.replay_buffers[player_id] = None
        return self.replay_buffers[player_id]

    def _preprocess_observation(self, observation, action_mask=None, player_id="default"):
        """
        Validate, extract and preprocess observation, and handle combat tracking.
        Returns (processed_obs, processed_mask)
        """
        if observation is None:
            raise ValueError("Observation cannot be None")

        if not isinstance(observation, (dict, np.ndarray)):
            raise ValueError(f"Invalid observation type: {type(observation)}")

        # If observation is a dict (standard TFTSet4Gym format), extract tensor and mask
        if isinstance(observation, dict) and 'tensor' in observation:
            obs = observation['tensor']
            mask = observation.get('action_mask', action_mask)
        else:
            obs = observation
            mask = action_mask

        if obs is None or (isinstance(obs, np.ndarray) and obs.size == 0):
            raise ValueError("Observation tensor is empty or None")

        # Flatten to 1D array if needed (schema expects flat observation)
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs = obs.flatten()

        # Combat outcome tracking
        try:
            current_turns = extract_field_from_observation(observation, 'turns_for_combat')
            current_health = extract_field_from_observation(observation, 'health')

            p_turns = self.prev_turns_for_combat.get(player_id)
            p_health = self.prev_health.get(player_id)
            p_obs = self.prev_observation.get(player_id)

            if p_turns == 0 and current_turns > 0:
                # Combat just finished (turns_for_combat reset to max)
                if p_health is not None and p_obs is not None:
                    # Win if health stayed the same, loss if it decreased
                    result = 1.0 if current_health >= p_health else 0.0
                    self._store_combat(p_obs, result)

            self.prev_turns_for_combat[player_id] = current_turns
            self.prev_health[player_id] = current_health
            # Use copy to avoid reference issues if observation is mutated
            self.prev_observation[player_id] = obs.copy() if isinstance(obs, np.ndarray) else obs
        except Exception:
            # If schema extraction fails, we just don't track combat for this step
            pass

        return obs, mask

    def _postprocess_result(self, obs, result, reward=None, terminated=None, player_id="default"):
        """
        Parse result and handle experience storage.
        Returns action
        """
        # Parse result: could be just action, or (action, policy), or (action, policy, value)
        policy = None
        value = 0
        if isinstance(result, tuple):
            action = result[0]
            if len(result) > 1:
                policy = result[1]
            if len(result) > 2:
                value = result[2]
        else:
            action = result

        if self.save_data:
            self._store_experience(
                observation=obs, 
                action=action, 
                policy=policy, 
                value=value, 
                reward=reward or 0,
                terminated=terminated or False,
                player_id=player_id
            )

        return action

    def select_action(self, observation, action_mask=None, reward=None, terminated=None, precomputed_results=None, player_id="default"):
        """
        Select an action based on the current observation and action mask.
        """
        obs, mask = self._preprocess_observation(observation, action_mask, player_id)

        # Select action using implementation - handle complex returns and precomputed results
        result = self._select_action_impl(obs, mask, reward, terminated, precomputed_results=precomputed_results)

        return self._postprocess_result(obs, result, reward, terminated, player_id)

    def batch_select_action(self, observations, masks, rewards=None, terminated=None, precomputed_results=None, player_ids=None, **kwargs):
        """
        Select actions for a batch of observations.
        """
        processed_obs = []
        processed_masks = []
        pids = []
        
        # Phase 1: Preprocessing and Combat Tracking
        for i, observation in enumerate(observations):
            mask = masks[i] if i < len(masks) else None
            pid = player_ids[i] if player_ids and i < len(player_ids) else "default"
            
            obs, m = self._preprocess_observation(observation, mask, pid)
            processed_obs.append(obs)
            processed_masks.append(m)
            pids.append(pid)

        # Phase 2: Batched Inference
        results = self._batch_select_action_impl(
            processed_obs, 
            processed_masks, 
            rewards=rewards, 
            terminated=terminated, 
            precomputed_results=precomputed_results,
            **kwargs
        )

        # Phase 3: Postprocessing and Experience Storage
        actions = []
        for i, result in enumerate(results):
            obs = processed_obs[i]
            reward = rewards[i] if rewards and i < len(rewards) else None
            term = terminated[i] if terminated and i < len(terminated) else None
            pid = pids[i]
            
            action = self._postprocess_result(obs, result, reward, term, pid)
            actions.append(action)
            
        return actions

    def _batch_select_action_impl(self, observations, masks, rewards=None, terminated=None, precomputed_results=None, **kwargs):
        """
        Default batched implementation that falls back to _select_action_impl.
        Subclasses can override this for performance.
        """
        results = []
        for i, obs in enumerate(observations):
            mask = masks[i] if i < len(masks) else None
            reward = rewards[i] if rewards and i < len(rewards) else None
            term = terminated[i] if terminated and i < len(terminated) else None
            pc = precomputed_results[i] if precomputed_results and i < len(precomputed_results) else None
            
            result = self._select_action_impl(obs, mask, reward, term, precomputed_results=pc)
            results.append(result)
        return results

    def _store_experience(self, observation=None, policy=None, value=0, reward=0, terminated=False, action=None, player_id="default"):
        buffer = self._get_buffer(player_id)
        if buffer is not None:
            buffer.store_step(observation=observation, policy=policy, value=value, reward=reward, action=action)

    def _store_combat(self, observation, result):
        """Store combat experience (observation, result)."""
        if self.global_buffer is not None:
            if hasattr(self.global_buffer, 'store_combat'):
                self.global_buffer.store_combat((observation, result))

    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None, precomputed_results=None):
        """
        Implementation method that subclasses should override.
        """
        raise NotImplementedError("Subclasses must implement _select_action_impl")

    def terminate(self, final_value, player_id=None):
        """
        Handle episode termination.
        """
        if player_id is not None:
            buffer = self.replay_buffers.get(player_id)
            if buffer is not None:
                buffer.move_buffer_to_global(final_value=final_value)
                # We don't necessarily want to delete it from the dict, but reset it
                buffer.reset()
        else:
            # Terminate only the buffers that have scores in final_value if it's a dict
            # This is critical for shared agents across multiple concurrent games
            for pid, buffer in self.replay_buffers.items():
                if isinstance(final_value, dict):
                    if pid in final_value:
                        buffer.move_buffer_to_global(final_value=final_value[pid])
                        buffer.reset()
                else:
                    buffer.move_buffer_to_global(final_value=final_value)
                    buffer.reset()
    
    def _get_champion_id(self, champ_name):
        """Get champion ID from name using COST dictionary."""
        champ_names = list(COST.keys())
        if champ_name in champ_names:
            return champ_names.index(champ_name) - 1
        raise ValueError(f"Unknown champion: {champ_name}")

class RandomAgent(BaseAgent):
    def __init__(self, agent_name="RandomAgent", global_buffer=None, save_data=False):
        super().__init__(agent_name, global_buffer, save_data=save_data)

    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None, precomputed_results=None):
        """Select a random valid action."""
        return [np.random.randint(0, 6), np.random.randint(0, 37), np.random.randint(0, 28)]

class BuyingAgent(BaseAgent):
    def __init__(self, units_to_buy, agent_name="BuyingAgent", global_buffer=None, save_data=False):
        super().__init__(agent_name, global_buffer, save_data=save_data)
        self.units_to_buy = units_to_buy

    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None, precomputed_results=None):
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
        
        # Try to buy desired units from shop
        shop_priorities = []
        for i, champ in enumerate(units_in_shop):
            if champ and champ != " " and champ in self.units_to_buy:
                # Remove chosen suffix for cost lookup
                base_name = champ.replace("_c", "")
                if base_name not in COST:
                    raise ValueError(f"Unknown champion in shop: {champ}")
                cost = COST[base_name]
                if gold >= cost:
                    shop_priorities.append((cost, i, champ))
        
        # Buy cheapest desired unit first
        if shop_priorities:
            shop_priorities.sort(key=lambda x: x[0])  # Sort by cost, cheapest first
            cost, shop_index, champ_name = shop_priorities[0]
            champ_id = self._get_champion_id(champ_name.replace("_c", ""))
            return [2, champ_id, 0]  # action_type=2 for buy
        
        # Sell units not in our buying list
        board = get_board_units_from_observation(obs)
        for unit in board:
            unit_name = unit.get("name", "")
            if unit_name not in self.units_to_buy:
                pos = unit["pos_y"] * 7 + unit["pos_x"]
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
        """Calculate how many units are needed to reach 3-star (9 total units)."""
        units_needed = {}
        for unit_name, count in unit_counts.items():
            units_needed[unit_name] = max(0, 9 - count)
        return units_needed
    
    def get_unit_counts(self, observation):
        """Count all units of each type on board and bench."""
        unit_counts = {}
        
        # Count board units
        board = get_board_units_from_observation(observation)
        for unit in board:
            name = unit["name"]
            stars = int(unit["stars"])
            if stars == 1:
                unit_count = 1
            elif stars == 2:
                unit_count = 3
            elif stars == 3:
                unit_count = 9
            else:
                unit_count = 1  # default
                
            unit_counts[name] = unit_counts.get(name, 0) + unit_count
        
        # Count bench units
        bench = get_bench_units_from_observation(observation)
        for unit_name in bench:
            if unit_name and unit_name != " ":
                unit_counts[unit_name] = unit_counts.get(unit_name, 0) + 1
                
        return unit_counts
    
    def is_board_and_bench_full(self, observation):
        """Check if both board and bench are full."""
        board = get_board_units_from_observation(observation)
        bench = get_bench_units_from_observation(observation)
        level = extract_field_from_observation(observation, 'level')
        
        board_full = len(board) >= int(level)
        bench_full = len([unit for unit in bench if unit and unit != " "]) >= 9
        
        return board_full and bench_full
    
    def find_lowest_priority_unit_to_sell(self, observation):
        """Find the lowest priority unit to sell when board and bench are full."""
        unit_counts = self.get_unit_counts(observation)
        units_needed = self.count_units_needed_for_three_star(unit_counts)
        
        board = get_board_units_from_observation(observation)
        bench = get_bench_units_from_observation(observation)
        
        candidates = []
        for unit in board:
            name = unit["name"]
            if name in self.units_to_buy and unit_counts.get(name, 0) > 1:
                pos = unit["pos_y"] * 7 + unit["pos_x"]
                priority = units_needed.get(name, 9)
                candidates.append((priority, 3, pos))
        
        for i, unit_name in enumerate(bench):
            if unit_name and unit_name != " " and unit_name in self.units_to_buy and unit_counts.get(unit_name, 0) > 1:
                pos = 28 + i
                priority = units_needed.get(unit_name, 9)
                candidates.append((priority, 3, pos))
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, action_type, position = candidates[0]
            return [action_type, position, 0]
            
        return None

class CultistAgent(BuyingAgent):
    def __init__(self, global_buffer=None, save_data=False):
        cultist_units = ["elise", "twistedfate", "pyke", "evelynn", "aatrox", "zilean", "kalista", "jhin"]
        super().__init__(cultist_units, "CultistAgent", global_buffer, save_data=save_data)

class DivineAgent(BuyingAgent):
    def __init__(self, global_buffer=None, save_data=False):
        divine_units = ["wukong", "jax", "irelia", "lux", "warwick", "leesin", "ashe", "kindred", "teemo"]
        super().__init__(divine_units, "DivineAgent", global_buffer, save_data=save_data)

class RerollAgent(BuyingAgent):
    def __init__(self, global_buffer=None, save_data=False):
        reroll_units = ["yasuo", "fiora", "vayne", "nidalee", "garen"]
        super().__init__(reroll_units, "RerollAgent", global_buffer, save_data=save_data)
        
    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None, precomputed_results=None):
        """Reroll strategy implementation."""
        gold = extract_field_from_observation(obs, 'gold')
        units_in_shop = get_shop_units_from_observation(obs)
        
        for champ in units_in_shop:
            if champ in self.units_to_buy and gold >= 5:
                action = [2, self._get_champion_id(champ.replace("_c", "")), 0]
                return action
        
        level = extract_field_from_observation(obs, 'level')
        if gold > 30.0 and level <= 6:
            return [4, 0, 0]
        if gold > 60.0 and level < 6:
            return [5, 0, 0]
            
        return super()._select_action_impl(obs, action_mask)

class FastLevelAgent(BaseAgent):
    def __init__(self, global_buffer=None, save_data=False):
        super().__init__("FastLevelAgent", global_buffer, save_data=save_data)
        
    def _select_action_impl(self, obs, action_mask, reward=None, terminated=None, precomputed_results=None):
        """Fast level strategy implementation."""
        gold = extract_field_from_observation(obs, 'gold')
        level = extract_field_from_observation(obs, 'level')
        
        if gold > 40.0 and level < 8:
            return [5, 0, 0]
            
        units_in_shop = get_shop_units_from_observation(obs)
        if gold > 60.0 and len(units_in_shop) > 0 and units_in_shop[0] != " ":
            champ = units_in_shop[0]
            champ_id = self._get_champion_id(champ.replace("_c", ""))
            return [2, champ_id, 0]
            
        if gold > 70.0 and level >= 7:
            return [4, 0, 0]
            
        return [0, 0, 0]
