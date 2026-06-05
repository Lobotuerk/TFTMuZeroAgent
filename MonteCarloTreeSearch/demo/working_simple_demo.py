#!/usr/bin/env python3
"""
Working demonstration of MCTS using built-in C++ TicTacToe
"""
import sys
import os
# Add the project directory to path to find pymcts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymcts
import time

def demo_basic_mcts():
    """Demonstrate basic MCTS functionality with different parameters."""
    print("🎯 MCTS TicTacToe Performance Demo")
    print("=" * 45)
    
    # Test different iteration counts
    test_configs = [
        (50, "Low"),
        (200, "Medium"), 
        (1000, "High")
    ]
    
    time_limit = 3
    
    for iterations, label in test_configs:
        print(f"\n🧠 Test: {label} Iterations ({iterations})")
        
        # Create fresh state and agent
        state = pymcts.TicTacToe_state()
        agent = pymcts.MCTS_agent(state, iterations, time_limit)
        
        # Time the move generation
        start_time = time.time()
        move = agent.genmove(None)
        elapsed = time.time() - start_time
        
        print(f"   Time: {elapsed:.3f}s")
        print(f"   Move: {move.sprint()}")
        
        # Get updated state after move
        current_state = agent.get_current_state()
        print(f"   Terminal: {current_state.is_terminal()}")
    
    print("\n📊 Game State Analysis")
    print("-" * 30)
    
    # Analyze initial game state
    initial_state = pymcts.TicTacToe_state()
    print(f"Initial state terminal: {initial_state.is_terminal()}")
    print(f"Self side turn: {initial_state.is_self_side_turn()}")
    
    # Show available moves
    moves = initial_state.actions_to_try()
    print(f"Available moves: {len(moves)}")
    
    # Display board
    print("\n🎮 Initial Board:")
    initial_state.print()
    
    print("\n💡 Key Insights:")
    print("   • More iterations generally improve move quality")
    print("   • C++ implementation provides fast, reliable gameplay")
    print("   • MCTS algorithm balances exploration and exploitation")

def demo_complete_game():
    """Demonstrate a complete game between two MCTS agents."""
    print("\n\n🤖 MCTS vs MCTS Complete Game")
    print("=" * 50)
    
    # Create initial state 
    initial_state = pymcts.TicTacToe_state()
    
    # Create two agents with separate cloned states
    agent1 = pymcts.MCTS_agent(initial_state.clone(), 200, 2)  # Agent X (weaker)
    agent2 = pymcts.MCTS_agent(initial_state.clone(), 500, 1)  # Agent O (stronger)
    
    move_count = 0
    print("\nGame Progress:")
    print("-" * 20)
    
    # Track the actual game state separately
    game_state = initial_state.clone()
    last_move = None  # Track the last move made
    
    while not game_state.is_terminal() and move_count < 9:
        if game_state.is_self_side_turn():
            # Agent 1's turn (X)
            print(f"Move {move_count + 1}: Agent 1 (X) thinking...")
            move = agent1.genmove(last_move)  # Process opponent's last move and generate own move
            
            if move is None:
                print("   Agent 1 couldn't generate a move!")
                break
                
            print(f"   Agent 1 plays: {move.sprint()}")
            last_move = move
            game_state = game_state.next_state(move)
            
        else:
            # Agent 2's turn (O) 
            print(f"Move {move_count + 1}: Agent 2 (O) thinking...")
            move = agent2.genmove(last_move)  # Process opponent's last move and generate own move
            
            if move is None:
                print("   Agent 2 couldn't generate a move!")
                break
                
            print(f"   Agent 2 plays: {move.sprint()}")
            last_move = move
            game_state = game_state.next_state(move)
        
        move_count += 1
        
        # Show board after each few moves
        if move_count % 3 == 0 or game_state.is_terminal():
            print("\n   Current board:")
            game_state.print()
    
    print("\n🏁 Game Over!")  
    print(f"Total moves: {move_count}")
    print(f"Final state terminal: {game_state.is_terminal()}")
    
    if game_state.is_terminal():
        try:
            winner = game_state.get_winner()
            if winner:
                print(f"Winner: {winner}")
            else:
                print("Game ended in a draw")
        except:
            print("Game completed")

def main():
    """Main demo function"""
    try:
        demo_basic_mcts()
        
        # Ask if user wants to see a complete game
        print("\n" + "=" * 60)
        response = input("Would you like to see a complete MCTS vs MCTS game? (y/n): ").lower().strip()
        if response == 'y':
            demo_complete_game()
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()