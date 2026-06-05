#!/usr/bin/env python3
"""
Simple test of the C++ ownership approach with minimal Python game
"""
import sys
import os
# Add the parent directory to path to find pymcts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymcts
import random

class SimpleMove(pymcts.MCTS_move):
    def __init__(self, value: int):
        super().__init__()
        self.value = value
    
    def __eq__(self, other) -> bool:
        return isinstance(other, SimpleMove) and self.value == other.value
    
    def sprint(self) -> str:
        return f"Move({self.value})"

class SimpleState(pymcts.MCTS_state):
    def __init__(self, turn: int = 1, moves_left: int = 3):
        super().__init__()
        self.turn = turn
        self.moves_left = moves_left
    
    def actions_to_try(self):
        if self.is_terminal():
            return []
        return [SimpleMove(1), SimpleMove(2)]
    
    def next_state(self, move):
        # Just alternate turns and reduce moves left
        new_turn = 3 - self.turn  # Switch between 1 and 2
        new_moves_left = self.moves_left - 1
        return SimpleState(new_turn, new_moves_left)
    
    def rollout(self) -> float:
        # Random result based on current turn
        return random.random()
    
    def is_terminal(self) -> bool:
        return self.moves_left <= 0
    
    def is_self_side_turn(self) -> bool:
        return self.turn == 1
    
    def print(self):
        print(f"SimpleState: turn={self.turn}, moves_left={self.moves_left}")

def test_simple_ownership():
    print("🎯 Testing Simple Python Game with C++ Ownership")
    
    state = SimpleState(turn=1, moves_left=2)
    print("✅ Created simple Python state")
    state.print()
    
    try:
        # Test basic state operations
        moves = state.actions_to_try()
        print(f"✅ Available moves: {[m.sprint() for m in moves]}")
        
        if moves:
            next_state = state.next_state(moves[0])
            print(f"✅ Next state created")
            next_state.print()
        
        # Now test with MCTS
        agent = pymcts.MCTS_agent(state, 20, 1)  # Small number of iterations
        print("✅ MCTS agent created")
        
        move = agent.genmove(None)
        if move:
            print(f"✅ Agent chose: {move.sprint()}")
        else:
            print("❌ No move generated")
            
        print("\n🎉 Simple ownership test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_ownership()