#!/usr/bin/env python3
"""
Simple working example of PyMCTS TicTacToe
"""
import sys
import os
# Add the project directory to path to find pymcts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymcts

def simple_tictactoe_game():
    print("=== PyMCTS TicTacToe Demo ===")
    
    # Create initial state
    state = pymcts.TicTacToe_state()
    print("Initial board:")
    state.print()
    
    # Create MCTS agent
    agent = pymcts.MCTS_agent(state, max_iter=100, max_seconds=2)
    
    print("\nPlaying a few moves...")
    
    # Agent makes first move
    move1 = agent.genmove(None)
    current_state = agent.get_current_state()
    print(f"\nAgent (X) plays: {move1}")
    current_state.print()
    
    # Simulate opponent move
    opponent_moves = current_state.actions_to_try()
    if opponent_moves:
        opponent_move = opponent_moves[0]  # Take first available
        print(f"\nOpponent (O) plays: {opponent_move}")
        
        # Agent processes opponent move and makes next move
        next_move = agent.genmove(opponent_move)
        if next_move:
            final_state = agent.get_current_state()
            print(f"\nAgent (X) responds: {next_move}")
            final_state.print()
            print(f"\nGame status - Terminal: {final_state.is_terminal()}")
            if final_state.is_terminal():
                winner = final_state.get_winner()
                print(f"Winner: {winner}")
    
    print("\n=== Demo completed successfully! ===")

if __name__ == "__main__":
    simple_tictactoe_game()