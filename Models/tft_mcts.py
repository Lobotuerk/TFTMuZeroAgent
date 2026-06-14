#!/usr/bin/env python3
"""
TFT MCTS Implementation

This module provides MCTS-compatible classes for Teamfight Tactics (TFT)
that integrate with both the TFTSet4Gym environment and the PyMCTS library.
It unifies legacy string-based actions with modern integer-based 3D actions.
"""

import numpy as np
import random
import torch
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
from Models.action_conversion import action_3d_to_policy
import sys
import os

# Add MonteCarloTreeSearch build directory to path
mcts_build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'MonteCarloTreeSearch')
if mcts_build_path not in sys.path:
    sys.path.insert(0, mcts_build_path)

try:
    import pymcts
    PYMCTS_AVAILABLE = True
    MCTS_MoveBase = pymcts.MCTS_move
    MCTS_StateBase = pymcts.MCTS_state
except ImportError:
    PYMCTS_AVAILABLE = False
    MCTS_MoveBase = object
    MCTS_StateBase = object

# Add parent directory to access config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
from Models.Common_agents import extract_field_from_observation


class TFTMove(MCTS_MoveBase):
    """
    Represents a single move/action in TFT.
    Supports both legacy string-based and modern 3D integer-based actions.
    """
    __slots__ = ['action_type', 'target_1', 'target_2', 'index', 'player_id', 
                 'shop_index', 'board_index', 'from_index', 'to_index']
    
    def __init__(self, action_type: Union[int, str], target_1: int = 0, target_2: int = 0, 
                 index: int = 0, player_id: str = "player_0", **kwargs):
        if PYMCTS_AVAILABLE:
            super().__init__()
        
        # Action data
        self.action_type = action_type
        self.target_1 = target_1
        self.target_2 = target_2
        self.index = index
        self.player_id = player_id
        
        # Legacy/Extra parameters support
        self.shop_index = kwargs.get('shop_index')
        self.board_index = kwargs.get('board_index')
        self.from_index = kwargs.get('from_index')
        self.to_index = kwargs.get('to_index')
        
        # Map parameters if they weren't provided but action_type suggests them
        if self.shop_index is None and action_type in [2, "buy"]:
            self.shop_index = target_1
        if self.board_index is None and action_type in [3, "sell"]:
            self.board_index = target_1
        if self.from_index is None and action_type in [1, "move"]:
            self.from_index = target_1
        if self.to_index is None and action_type in [1, "move"]:
            self.to_index = target_2

    def __eq__(self, other) -> bool:
        if not isinstance(other, TFTMove):
            return False
        
        return (self.action_type == other.action_type and
                self.target_1 == other.target_1 and
                self.target_2 == other.target_2 and
                self.index == other.index and
                self.player_id == other.player_id and
                self.shop_index == other.shop_index and
                self.board_index == other.board_index and
                self.from_index == other.from_index and
                self.to_index == other.to_index)
    
    def sprint(self) -> str:
        """String representation of the move for PyMCTS."""
        return f"TFTMove(type={self.action_type}, t1={self.target_1}, t2={self.target_2}, idx={self.index})"
    
    def __str__(self) -> str:
        return self.sprint()
    
    def __repr__(self):
        return self.sprint()

    def to_env_action(self) -> List[int]:
        """Convert to environment action format [action_type, target_1, target_2]."""
        # Map string types to integers if necessary
        type_map = {"pass": 0, "move": 1, "buy": 2, "sell": 3, "reroll": 4, "level": 5, "item": 6}
        a_type = type_map.get(self.action_type, self.action_type)
        if isinstance(a_type, str):
            a_type = 0 # Fallback
            
        return [int(a_type), int(self.target_1), int(self.target_2)]

    def to_numpy(self) -> List[float]:
        """Convert move to one-hot numpy array for neural network processing."""
        env_action = self.to_env_action()
        policy = action_3d_to_policy(env_action, action_dims=config.ACTION_DIM)
        return policy.tolist()


_GLOBAL_MOVES_CACHE = {}
_GLOBAL_MOVES_LOCK = threading.Lock()

def get_precreated_moves(player_id: str) -> List[TFTMove]:
    """Retrieve pre-created TFTMove objects to avoid high dynamic allocation cost."""
    global _GLOBAL_MOVES_CACHE
    if player_id in _GLOBAL_MOVES_CACHE:
        return _GLOBAL_MOVES_CACHE[player_id]
        
    with _GLOBAL_MOVES_LOCK:
        if player_id in _GLOBAL_MOVES_CACHE:
            return _GLOBAL_MOVES_CACHE[player_id]
            
        actions = []
        # Pass (0), Roll (4), Level (5)
        for i in [0, 4, 5]:
            actions.append(TFTMove(i, 0, 0, index=i, player_id=player_id))
        
        # Buy: 5 shop slots
        for i in range(5):
            actions.append(TFTMove(2, i, 0, index=2, player_id=player_id))
            
        # Sell: 37 positions
        for i in range(37):
            actions.append(TFTMove(3, i, 0, index=3, player_id=player_id))
            
        # Move: Sample a subset to avoid explosion
        for i in range(28):
            for j in [0, 5, 10, 15, 20, 25, 30, 35]: # Sampled targets
                if i != j:
                    actions.append(TFTMove(1, i, j, index=1, player_id=player_id))
                    
        _GLOBAL_MOVES_CACHE[player_id] = actions
        return actions


class TFTState(MCTS_StateBase):
    """
    Represents a TFT game state for MCTS simulation.
    Unifies raw environment observations with MuZero-style recurrent inference.
    """
    
    def __init__(self, observation: Any = None, mask: Optional[np.ndarray] = None, 
                 network=None, is_raw_observation: bool = True,
                 precomputed: Optional[Dict[str, Any]] = None,
                 current_player: str = "player_0", round_num: int = 1,
                 batch_queue=None,
                 add_root_noise: bool = False,
                 **kwargs):
        if PYMCTS_AVAILABLE:
            super().__init__()
            
        self.observation = observation if observation is not None else kwargs.get('observations')
        self.mask = mask.copy() if mask is not None else self._create_default_mask()
        self.network = network
        self.is_raw_observation = is_raw_observation
        self.current_player = current_player
        self.round_num = round_num
        self.batch_queue = batch_queue
        self.add_root_noise = add_root_noise
        
        # MCTS Values
        self.hidden_state = None
        self.policy = None
        self.value = 0.5
        
        # Cache for actions to avoid redundant generation
        self._cached_actions = None
        
        if self.observation is not None:
            self._initialize_state(self.observation, is_raw_observation, precomputed)

    @property
    def player_turn(self):
        return True # Default for MCTS integration

    @property
    def health(self):
        if self.observation is not None:
            val = extract_field_from_observation(self.observation, 'health')
            if val is not None:
                if isinstance(val, (np.ndarray, list)):
                    if hasattr(val, 'size'): # numpy
                        return float(val.flat[0]) if val.size > 0 else 100.0
                    return float(val[0]) if len(val) > 0 else 100.0
                return float(val)
        return 100.0

    @property
    def level(self):
        if self.observation is not None:
            val = extract_field_from_observation(self.observation, 'level')
            if val is not None:
                if isinstance(val, (np.ndarray, list)):
                    if hasattr(val, 'size'): # numpy
                        return int(val.flat[0]) if val.size > 0 else 1
                    return int(val[0]) if len(val) > 0 else 1
                return int(val)
        return 1

    @property
    def turns_for_combat(self):
        if self.observation is not None:
            val = extract_field_from_observation(self.observation, 'turns_for_combat')
            if val is not None:
                if isinstance(val, (np.ndarray, list)):
                    if hasattr(val, 'size'): # numpy
                        return int(val.flat[0]) if val.size > 0 else 0
                    return int(val[0]) if len(val) > 0 else 0
                return int(val)
        return 0

    def _initialize_state(self, observation, is_raw_observation, precomputed):
        if precomputed is not None:
            # Keep hidden_state as tensor if it already is one, otherwise convert to numpy
            hs = precomputed['hidden_state']
            if isinstance(hs, torch.Tensor):
                self.hidden_state = hs.detach()
            else:
                self.hidden_state = np.asarray(hs, dtype=np.float32).flatten()
                
            self.policy = np.asarray(precomputed['policy'], dtype=np.float32)
            self.value = np.asarray(precomputed['value'], dtype=np.float32)
        elif self.network is not None:
            with torch.no_grad():
                device = next(self.network.parameters()).device
                if isinstance(observation, torch.Tensor):
                    input_obs = observation.to(device)
                else:
                    input_obs = torch.tensor(observation, dtype=torch.float32).to(device)
                
                # Handle batching
                if input_obs.ndim == 1:
                    input_obs = input_obs.unsqueeze(0)
                elif is_raw_observation and input_obs.ndim == 3:
                    input_obs = input_obs.unsqueeze(0)
                
                if is_raw_observation:
                    res = self.network.initial_inference(input_obs)
                    if isinstance(res, tuple):
                        network_output = res[0]
                    else:
                        network_output = res
                    self.hidden_state = network_output["hidden_state"].squeeze(0).detach()
                    self.policy = network_output["policy_logits"].squeeze(0).cpu().numpy()
                    self.value = network_output["value"].squeeze(0).cpu().numpy()
                else:
                    # If not raw, observation is already a hidden state
                    self.hidden_state = input_obs.squeeze(0).detach()
                    self.policy, self.value = self.network.prediction(input_obs)
                    self.policy = self.policy.squeeze(0).cpu().numpy()
                    self.value = self.value.squeeze(0).cpu().numpy()
        else:
            self.hidden_state = observation
            self.value = 0.5

    def _create_default_mask(self) -> np.ndarray:
        return np.zeros((55,), dtype=bool)

    def actions_to_try(self) -> List[TFTMove]:
        """Generate all possible moves from current state."""
        return get_precreated_moves(self.current_player)

    def next_state(self, move: TFTMove) -> 'TFTState':
        """Apply a move to get the next game state."""
        if self.network is not None:
            if self.batch_queue is not None:
                res = self.batch_queue.predict(self.hidden_state, move.to_numpy())
                new_hidden = res["hidden_state"]
                precomputed = {
                    "hidden_state": new_hidden,
                    "policy": res["policy_logits"],
                    "value": res["value"],
                }
                return TFTState(new_hidden, self.mask, self.network, is_raw_observation=False,
                               precomputed=precomputed, current_player=self.current_player,
                               round_num=self.round_num, batch_queue=self.batch_queue)

            with torch.no_grad():
                device = next(self.network.parameters()).device
                # hidden_state is already a tensor on GPU
                if isinstance(self.hidden_state, torch.Tensor):
                    input_obs = self.hidden_state.unsqueeze(0).to(device)
                else:
                    input_obs = torch.tensor(self.hidden_state, dtype=torch.float32).unsqueeze(0).to(device)
                
                network_output = self.network.recurrent_inference(input_obs, move.to_numpy())
                new_hidden = network_output["hidden_state"].squeeze(0).detach()
                
                precomputed = None
                if "policy_logits" in network_output and "value" in network_output:
                    precomputed = {
                        "hidden_state": new_hidden,
                        "policy": network_output["policy_logits"].squeeze(0).cpu().numpy(),
                        "value": network_output["value"].squeeze(0).cpu().numpy()
                    }
                
                return TFTState(new_hidden, self.mask, self.network, is_raw_observation=False, 
                               precomputed=precomputed, current_player=self.current_player, 
                               round_num=self.round_num, batch_queue=self.batch_queue)
        
        # Basic state transition for prototype
        return TFTState(self.observation, self.mask, None, is_raw_observation=True,
                       current_player=self.current_player, round_num=self.round_num + 1)

    def rollout(self) -> float:
        """Estimate value of current state."""
        if hasattr(self.value, 'item'):
            return float(self.value.item())
        return float(self.value)

    def is_terminal(self) -> bool:
        return self.round_num > 50

    def is_self_side_turn(self) -> bool:
        return True

    def clone(self) -> 'TFTState':
        hs = self.hidden_state
        if isinstance(hs, torch.Tensor):
            hs = hs.clone()
        elif hs is not None:
            hs = hs.copy()
            
        return TFTState(
            observation=hs if hs is not None else self.observation,
            mask=self.mask.copy() if self.mask is not None else None,
            network=self.network,
            is_raw_observation=False,
            current_player=self.current_player,
            round_num=self.round_num,
            batch_queue=self.batch_queue
        )

    def get_action_probabilities(self, moves: Optional[List[TFTMove]] = None) -> List[float]:
        if moves is None:
            moves = self.actions_to_try()

        if self.policy is None:
            return [1.0 / len(moves)] * len(moves)

       # Split policy into 3 independent softmax distributions per ACTION_DIM
        dims = config.ACTION_DIM
        blocks = np.split(np.asarray(self.policy, dtype=np.float64), np.cumsum(dims[:-1]))
        block_probs = []
        for block in blocks:
            b = block - np.max(block)
            exp_b = np.exp(b)
            block_probs.append(exp_b / np.sum(exp_b))

        # Compute joint probability for each move
        scores = np.zeros(len(moves), dtype=np.float64)
        for i, move in enumerate(moves):
            env_action = move.to_env_action()
            prob = 1.0
            for j in range(3):
                idx = int(env_action[j])
                if 0 <= idx < dims[j]:
                    prob *= block_probs[j][idx]
                else:
                    prob = 0.0
                    break
            scores[i] = prob

        total = scores.sum()
        if total > 0:
            probs = (scores / total).tolist()
        else:
            probs = [1.0 / len(moves)] * len(moves)

        if self.add_root_noise:
            alpha = config.ROOT_DIRICHLET_ALPHA
            frac = config.ROOT_EXPLORATION_FRACTION
            noise = np.random.dirichlet([alpha] * len(probs))
            probs = ((1 - frac) * np.array(probs) + frac * noise).tolist()

        return probs
    
    def print(self):
        print(f"TFTState(player={self.current_player}, round={self.round_num}, value={self.value:.3f})")


def create_tft_state_from_env() -> TFTState:
    """Factory function for environment-based states."""
    from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
    env = parallel_env()
    observations, infos = env.reset()
    first_player = list(observations.keys())[0]
    return TFTState(observation=observations[first_player], current_player=first_player)


if __name__ == "__main__":
    print("Testing unified TFT MCTS implementation...")
    move = TFTMove(2, target_1=2)
    print(f"Created move: {move}")
    print(f"Env action: {move.to_env_action()}")
    
    # Mock observation
    obs = np.zeros(getattr(config, 'OBSERVATION_SIZE', 5152))
    state = TFTState(obs)
    print(f"Created state: {state}")
    state.print()
