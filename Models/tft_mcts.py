#!/usr/bin/env python3
"""
TFT MCTS Implementation

This module provides MCTS-compatible classes for Teamfight Tactics (TFT)
that can integrate with both the TFTSet4Gym environment and the PyMCTS library.

Following TDD approach - implementing minimal functionality to pass tests.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional
from tft_set4_gym.tft_simulator import parallel_env


class TFTMove:
    """
    Represents a single move/action in TFT that can be applied to a game state.
    
    This class encapsulates all possible TFT actions:
    - buy: Purchase a unit from shop
    - sell: Sell a unit from board/bench
    - move: Move a unit between positions
    - level: Buy experience/level up
    - reroll: Reroll the shop
    """
    
    def __init__(self, action_type: str, player_id: str, **kwargs):
        """
        Initialize a TFT move.
        
        Args:
            action_type: Type of action ("buy", "sell", "move", "level", "reroll")
            player_id: ID of the player making the move
            **kwargs: Additional parameters based on action type
                - shop_index: For buy actions
                - board_index: For sell actions  
                - from_index, to_index: For move actions
        """
        self.action_type = action_type
        self.player_id = player_id
        
        # Store additional parameters
        self.shop_index = kwargs.get('shop_index')
        self.board_index = kwargs.get('board_index')
        self.from_index = kwargs.get('from_index')
        self.to_index = kwargs.get('to_index')
    
    def __eq__(self, other) -> bool:
        """Check equality between moves."""
        if not isinstance(other, TFTMove):
            return False
        
        return (self.action_type == other.action_type and
                self.player_id == other.player_id and
                self.shop_index == other.shop_index and
                self.board_index == other.board_index and
                self.from_index == other.from_index and
                self.to_index == other.to_index)
    
    def __str__(self) -> str:
        """String representation of the move."""
        base = f"TFTMove({self.action_type}"
        
        if self.shop_index is not None:
            base += f", shop={self.shop_index}"
        if self.board_index is not None:
            base += f", board={self.board_index}"
        if self.from_index is not None and self.to_index is not None:
            base += f", from={self.from_index}, to={self.to_index}"
            
        return base + ")"
    
    def __repr__(self):
        return self.__str__()


class TFTState:
    """
    Represents a TFT game state for MCTS simulation.
    
    This class wraps the TFTSet4Gym environment state and provides
    the interface needed for MCTS tree search.
    """
    
    def __init__(self, observations: Dict[str, np.ndarray], current_player: str, 
                 env_state: Optional[Any] = None, round_num: int = 1):
        """
        Initialize TFT state.
        
        Args:
            observations: Dictionary mapping player IDs to their observations
            current_player: ID of the current player making decisions
            env_state: Optional internal environment state for advanced usage
            round_num: Current round number
        """
        self.observations = observations
        self.current_player = current_player
        self.env_state = env_state
        self.round_num = round_num
        self.players = list(observations.keys())
    
    def actions_to_try(self) -> List[TFTMove]:
        """
        Generate all possible moves from current state.
        
        Returns:
            List of TFTMove objects representing valid actions
        """
        moves = []
        
        # Always allow basic actions
        moves.append(TFTMove("reroll", self.current_player))
        moves.append(TFTMove("level", self.current_player))
        
        # Add buy actions for shop slots (assuming 5 shop slots)
        for i in range(5):
            moves.append(TFTMove("buy", self.current_player, shop_index=i))
        
        # Add sell actions for board/bench positions (assuming 28 total positions)
        for i in range(28):
            moves.append(TFTMove("sell", self.current_player, board_index=i))
        
        # Add move actions between board positions
        for from_pos in range(28):
            for to_pos in range(28):
                if from_pos != to_pos:
                    moves.append(TFTMove("move", self.current_player, 
                                       from_index=from_pos, to_index=to_pos))
        
        # For now, return a subset to avoid explosion of moves
        # In practice, you'd filter based on actual game state
        basic_moves = [m for m in moves if m.action_type in ["reroll", "level"]]
        shop_moves = [m for m in moves if m.action_type == "buy"][:3]  # First 3 shop slots
        
        return basic_moves + shop_moves
    
    def next_state(self, move: TFTMove) -> 'TFTState':
        """
        Apply a move to get the next game state.
        
        Args:
            move: TFTMove to apply
            
        Returns:
            New TFTState after applying the move
        """
        # For now, create a simple next state
        # In full implementation, this would use the TFT environment
        
        # Simulate next player (for planning phase) or same player (for combat)
        current_idx = self.players.index(self.current_player)
        next_player_idx = (current_idx + 1) % len(self.players)
        next_player = self.players[next_player_idx]
        
        # Create new state with incremented round if all players have acted
        next_round = self.round_num + (1 if next_player_idx == 0 else 0)
        
        # For minimal implementation, return similar state with different player
        return TFTState(
            observations=self.observations.copy(),
            current_player=next_player,
            env_state=self.env_state,
            round_num=next_round
        )
    
    def is_terminal(self) -> bool:
        """
        Check if this is a terminal game state.
        
        Returns:
            True if game is over, False otherwise
        """
        # For minimal implementation, games don't end
        # In practice, check if only one player has HP > 0
        return self.round_num > 50  # Arbitrary terminal condition
    
    def is_self_side_turn(self) -> bool:
        """
        Check if it's the self side's turn (needed for MCTS).
        
        Returns:
            True if it's the main player's turn
        """
        # For MCTS, typically the first player is "self"
        return self.current_player == self.players[0]
    
    def rollout(self) -> float:
        """
        Perform a random rollout from this state to estimate value.
        
        Returns:
            Value between 0.0 and 1.0 representing win probability for self
        """
        # Minimal implementation: random value
        # In practice, this would simulate the game to completion
        # or use a heuristic evaluation
        
        # Simple heuristic: earlier rounds have more uncertainty
        uncertainty = max(0.1, 1.0 - (self.round_num / 50.0))
        base_value = 0.5  # Neutral starting point
        
        # Add some randomness
        random_factor = random.uniform(-uncertainty, uncertainty)
        result = np.clip(base_value + random_factor, 0.0, 1.0)
        
        return result
    
    def clone(self) -> 'TFTState':
        """
        Create a deep copy of this state.
        
        Returns:
            Deep copy of the current state
        """
        return TFTState(
            observations={k: v.copy() for k, v in self.observations.items()},
            current_player=self.current_player,
            env_state=self.env_state,  # Shallow copy for now
            round_num=self.round_num
        )
    
    def __eq__(self, other) -> bool:
        """Check equality between states."""
        if not isinstance(other, TFTState):
            return False
        
        return (self.current_player == other.current_player and
                self.round_num == other.round_num and
                self.players == other.players)
    
    def __str__(self) -> str:
        """String representation of the state."""
        return f"TFTState(player={self.current_player}, round={self.round_num}, players={len(self.players)})"
    
    def __repr__(self):
        return self.__str__()


# Factory function for easy state creation
def create_tft_state_from_env() -> TFTState:
    """
    Create a TFT state from a fresh TFTSet4Gym environment.
    
    Returns:
        TFTState initialized from environment reset
    """
    env = parallel_env()
    observations, infos = env.reset()
    first_player = list(observations.keys())[0]
    
    return TFTState(observations=observations, current_player=first_player)


if __name__ == "__main__":
    # Quick test of the implementation
    print("Testing TFT MCTS implementation...")
    
    # Test move creation
    move = TFTMove("buy", "player_0", shop_index=2)
    print(f"Created move: {move}")
    
    # Test state creation
    try:
        state = create_tft_state_from_env()
        print(f"Created state: {state}")
        
        moves = state.actions_to_try()
        print(f"Generated {len(moves)} moves")
        
        if moves:
            next_state = state.next_state(moves[0])
            print(f"Applied move, next state: {next_state}")
            
        print("✅ Basic implementation working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")