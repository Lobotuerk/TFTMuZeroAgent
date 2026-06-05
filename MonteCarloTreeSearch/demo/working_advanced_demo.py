#!/usr/bin/env python3
"""
Working demonstration of MCTS with different configurations

This demo shows how different MCTS parameters affect performance
using the reliable built-in C++ TicTacToe implementation.
"""
import sys
import os
# Add the project directory to path to find pymcts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymcts
import time

def compare_mcts_configurations():
    """Compare different MCTS configurations"""
    print("🎯 MCTS Configuration Comparison")
    print("=" * 50)
    
    configurations = [
        (100, 1, "Quick & Dirty"),
        (500, 2, "Balanced"),
        (1500, 3, "Thorough")
    ]
    
    results = []
    
    for iterations, time_limit, label in configurations:
        print(f"\n🗺️ Test: {label} (iter={iterations}, time={time_limit}s)")
        
        # Create state and agent
        state = pymcts.TicTacToe_state()
        agent = pymcts.MCTS_agent(state, iterations, time_limit)
        
        # Measure performance
        start_time = time.time()
        move = agent.genmove(None)
        elapsed = time.time() - start_time
        
        results.append((label, elapsed, move.sprint()))
        
        print(f"   ⏱️  Time taken: {elapsed:.3f}s")
        print(f"   🎯 Best move: {move.sprint()}")
        
        # Show some agent feedback/statistics if available
        try:
            agent.feedback()
        except:
            pass  # Not all builds may have this method
    
    # Summary comparison
    print("\n📈 Performance Summary")
    print("-" * 40)
    for label, time_taken, move in results:
        print(f"{label:15} {time_taken:6.3f}s -> {move}")
    
    fastest = min(results, key=lambda x: x[1])
    slowest = max(results, key=lambda x: x[1])
    
    speedup = slowest[1] / fastest[1] if fastest[1] > 0 else 1.0
    print(f"\nSpeedup factor: {speedup:.2f}x ({fastest[0]} vs {slowest[0]})")

def analyze_game_progression():
    """Analyze how MCTS performs as a game progresses"""
    print("\n🔍 Game Progression Analysis")
    print("=" * 50)
    
    # Create agents
    state = pymcts.TicTacToe_state()
    agent = pymcts.MCTS_agent(state, 800, 3)
    
    move_times = []
    move_count = 0
    
    print("\nMove-by-move analysis:")
    
    while not state.is_terminal() and move_count < 5:  # Analyze first few moves
        print(f"\n--- Move {move_count + 1} ---")
        print(f"Self side turn: {state.is_self_side_turn()}")
        
        # Get available moves
        moves = state.actions_to_try()
        print(f"Available moves: {len(moves)}")
        
        # Time the decision
        start_time = time.time()
        move = agent.genmove(None)
        elapsed = time.time() - start_time
        
        move_times.append(elapsed)
        print(f"Decision time: {elapsed:.3f}s")
        print(f"Chosen move: {move.sprint()}")
        
        # Update state
        state = agent.get_current_state()
        move_count += 1
        
        # Show board every couple moves
        if move_count % 2 == 0:
            print("Current board:")
            state.print()
    
    # Analysis summary
    if move_times:
        avg_time = sum(move_times) / len(move_times)
        print(f"\n📊 Move Time Statistics:")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   Fastest: {min(move_times):.3f}s")
        print(f"   Slowest: {max(move_times):.3f}s")
        
        # Trend analysis
        if len(move_times) > 2:
            early_avg = sum(move_times[:2]) / 2
            later_avg = sum(move_times[2:]) / len(move_times[2:])
            print(f"   Early moves: {early_avg:.3f}s")
            print(f"   Later moves: {later_avg:.3f}s")

def demonstrate_agent_configurations():
    """Demonstrate different agent configurations"""
    print("\n🚀 Agent Configuration Demo")
    print("=" * 50)
    
    # Test different agent configurations
    try:
        configs = [
            (500, 1, "Fast Agent"),
            (1000, 2, "Balanced Agent"),
            (2000, 3, "Strong Agent")
        ]
        
        results = []
        
        for iterations, time_limit, label in configs:
            print(f"\n📝 Testing {label} (iter={iterations}, time={time_limit}s)...")
            
            # Create agent and measure performance
            state = pymcts.TicTacToe_state()
            agent = pymcts.MCTS_agent(state, iterations, time_limit)
            
            start_time = time.time()
            move = agent.genmove(None)
            elapsed = time.time() - start_time
            
            results.append((label, elapsed))
            print(f"   Time: {elapsed:.3f}s, Move: {move.sprint()}")
        
        # Show results
        print("\n📈 Configuration Performance:")
        baseline = results[0][1]  # Fast agent time
        for label, time_taken in results:
            ratio = time_taken / baseline if baseline > 0 else 1.0
            print(f"   {label:15}: {time_taken:.3f}s (ratio: {ratio:.2f}x)")
    
    except Exception as e:
        print(f"Configuration demo failed: {e}")

def interactive_demo():
    """Simple interactive demo"""
    print("\n🎮 Interactive MCTS Demo")
    print("=" * 50)
    
    # Create game
    state = pymcts.TicTacToe_state()
    agent = pymcts.MCTS_agent(state, 1200, 3)
    
    print("\nYou are O, MCTS is X")
    print("Available moves will be shown as options.\n")
    
    while not state.is_terminal():
        # Show current board
        print("Current board:")
        state.print()
        
        if state.is_self_side_turn():
            # MCTS turn
            print("\n🤖 MCTS is thinking...")
            start_time = time.time()
            move = agent.genmove(None)
            think_time = time.time() - start_time
            
            print(f"🎯 MCTS plays {move.sprint()} (took {think_time:.2f}s)")
            state = agent.get_current_state()
        else:
            # Human turn
            try:
                moves = state.actions_to_try()
                if not moves:
                    print("No moves available!")
                    break
                
                print(f"\nAvailable moves:")
                for i, move in enumerate(moves):
                    print(f"   {i}: {move.sprint()}")
                
                user_input = input(f"\nYour move (0-{len(moves)-1}, or 'quit'): ").strip()
                if user_input.lower() == 'quit':
                    break
                
                move_idx = int(user_input)
                if 0 <= move_idx < len(moves):
                    human_move = moves[move_idx]
                    print(f"You play: {human_move.sprint()}")
                    agent.genmove(human_move)
                    state = agent.get_current_state()
                else:
                    print(f"Please enter a number between 0-{len(moves)-1}")
                    continue
                    
            except (ValueError, IndexError) as e:
                print(f"Invalid input! Please enter a valid number. Error: {e}")
    
    # Game over
    print("\n🏁 Game Over!")
    state.print()
    
    if state.is_terminal():
        try:
            winner = state.get_winner()
            if winner:
                if winner == 'x':
                    print("🤖 MCTS wins!")
                elif winner == 'o':
                    print("🎉 You win!")
                else:
                    print("🤝 It's a draw!")
        except:
            print("Game completed!")

def main():
    """Main demo function"""
    print("🚀 Advanced MCTS Demonstration")
    print("=" * 60)
    
    try:
        # Run core demos
        compare_mcts_configurations()
        analyze_game_progression()
        demonstrate_agent_configurations()
        
        # Ask about interactive demo
        print("\n" + "=" * 60)
        response = input("Would you like to play against MCTS? (y/n): ").lower().strip()
        if response == 'y':
            interactive_demo()
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • Higher iteration counts generally improve move quality")
        print("   • Time limits provide important performance constraints")
        print("   • Different configurations offer speed vs. quality tradeoffs")
        print("   • Built-in C++ implementation offers reliability and speed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()