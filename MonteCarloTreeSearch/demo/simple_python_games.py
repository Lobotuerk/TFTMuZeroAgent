#!/usr/bin/env python3
"""
Simple examples of games implemented directly in Python using PyMCTS
"""
import sys
import os
# Add the parent directory to path to find pymcts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymcts
import random

# Example 1: Simple Coin Flip Game
class CoinFlipMove(pymcts.MCTS_move):
    def __init__(self, choice: str):  # "heads" or "tails"
        super().__init__()
        self.choice = choice
    
    def __eq__(self, other) -> bool:
        return isinstance(other, CoinFlipMove) and self.choice == other.choice
    
    def sprint(self) -> str:
        return f"Choose{self.choice}"

class CoinFlipState(pymcts.MCTS_state):
    def __init__(self, guesses_left: int = 3, player1_score: int = 0, player2_score: int = 0, turn: int = 1):
        super().__init__()
        self.guesses_left = guesses_left
        self.player1_score = player1_score
        self.player2_score = player2_score
        self.turn = turn  # 1 or 2
    
    def clone(self):
        """Create a deep copy of this state"""
        return CoinFlipState(self.guesses_left, self.player1_score, self.player2_score, self.turn)
    
    def actions_to_try(self):
        if self.is_terminal():
            return []
        return [CoinFlipMove("heads"), CoinFlipMove("tails")]
    
    def next_state(self, move):
        # Simulate coin flip
        actual = random.choice(["heads", "tails"])
        correct = (move.choice == actual)
        
        new_p1_score = self.player1_score + (1 if correct and self.turn == 1 else 0)
        new_p2_score = self.player2_score + (1 if correct and self.turn == 2 else 0)
        
        return CoinFlipState(
            self.guesses_left - 1,
            new_p1_score,
            new_p2_score,
            3 - self.turn  # Switch player
        )
    
    def rollout(self) -> float:
        if self.is_terminal():
            if self.player1_score > self.player2_score:
                return 1.0
            elif self.player2_score > self.player1_score:
                return 0.0
            else:
                return 0.5
        
        # Simulate random play
        state = CoinFlipState(self.guesses_left, self.player1_score, self.player2_score, self.turn)
        while not state.is_terminal():
            moves = state.actions_to_try()
            move = random.choice(moves)
            state = state.next_state(move)
        
        return state.rollout()  # Use terminal evaluation
    
    def is_terminal(self) -> bool:
        return self.guesses_left <= 0
    
    def is_self_side_turn(self) -> bool:
        """Check if it's the self side's turn"""
        return self.turn == 1  # Player 1 is the 'self' side
    
    def print(self) -> None:
        """Print the current game state"""
        print(f"Coin Flip Game: Player 1 score: {self.player1_score}, Player 2 score: {self.player2_score}, Guesses left: {self.guesses_left}, Turn: Player {self.turn}")
    

# Example 2: Number Guessing Game
class GuessMove(pymcts.MCTS_move):
    def __init__(self, number: int):
        super().__init__()
        self.number = number
    
    def __eq__(self, other) -> bool:
        return isinstance(other, GuessMove) and self.number == other.number
    
    def sprint(self) -> str:
        return f"Guess{self.number}"

class NumberGuessingState(pymcts.MCTS_state):
    def __init__(self, target: int = None, min_val: int = 1, max_val: int = 10, turn: int = 1):
        super().__init__()
        self.target = target if target is not None else random.randint(min_val, max_val)
        self.min_val = min_val
        self.max_val = max_val
        self.turn = turn
        self.winner = None
    
    def clone(self):
        """Create a deep copy of this state"""
        new_state = NumberGuessingState(self.target, self.min_val, self.max_val, self.turn)
        new_state.winner = self.winner
        return new_state
    
    def actions_to_try(self):
        if self.is_terminal():
            return []
        return [GuessMove(i) for i in range(self.min_val, self.max_val + 1)]
    
    def next_state(self, move):
        if move.number == self.target:
            # Current player wins
            new_state = NumberGuessingState(self.target, self.min_val, self.max_val, self.turn)
            new_state.winner = self.turn
            return new_state
        else:
            # Switch turns, narrow range
            if move.number < self.target:
                new_min = max(self.min_val, move.number + 1)
                new_max = self.max_val
            else:
                new_min = self.min_val
                new_max = min(self.max_val, move.number - 1)
            
            if new_min > new_max:
                # No valid range left - current player loses
                new_state = NumberGuessingState(self.target, new_min, new_max, 3 - self.turn)
                new_state.winner = 3 - self.turn
                return new_state
            
            return NumberGuessingState(self.target, new_min, new_max, 3 - self.turn)
    
    def rollout(self) -> float:
        if self.is_terminal():
            if self.winner == 1:
                return 1.0
            elif self.winner == 2:
                return 0.0
            else:
                return 0.5
        
        # Random simulation
        state = NumberGuessingState(self.target, self.min_val, self.max_val, self.turn)
        while not state.is_terminal():
            moves = state.actions_to_try()
            if not moves:
                break
            move = random.choice(moves)
            state = state.next_state(move)
        
        return state.rollout()
    
    def is_terminal(self) -> bool:
        return self.winner is not None or self.min_val > self.max_val
    
    def is_self_side_turn(self) -> bool:
        """Check if it's the self side's turn"""
        return self.turn == 1  # Player 1 is the 'self' side
    
    def print(self) -> None:
        """Print the current game state"""
        if self.winner:
            print(f"Number Guessing Game: Winner is Player {self.winner}!")
        else:
            print(f"Number Guessing Game: Range [{self.min_val}-{self.max_val}], Target: {self.target}, Turn: Player {self.turn}")
    

def test_simple_games():
    """Test the simple Python-implemented games"""
    print("=== Testing Pure Python Game Implementations ===")
    
    # Test Coin Flip Game
    print("\n--- Coin Flip Game ---")
    coin_state = CoinFlipState()
    coin_state.print()
    
    # Wrap the Python state with SerializedPythonState for MCTS compatibility
    wrapped_coin_state = pymcts.SerializedPythonState(coin_state)
    agent = pymcts.MCTS_agent(wrapped_coin_state, max_iter=100, max_seconds=1)
    move = agent.genmove(None)
    print(f"MCTS chose: {move}")
    
    # Test Number Guessing Game
    print("\n--- Number Guessing Game ---")
    guess_state = NumberGuessingState(target=7)  # Known target for testing
    guess_state.print()
    
    # Wrap the Python state with SerializedPythonState for MCTS compatibility
    wrapped_guess_state = pymcts.SerializedPythonState(guess_state)
    agent2 = pymcts.MCTS_agent(wrapped_guess_state, max_iter=100, max_seconds=1)
    move2 = agent2.genmove(None)
    print(f"MCTS chose: {move2}")
    
    print("\n✅ All Python games working!")

if __name__ == "__main__":
    test_simple_games()