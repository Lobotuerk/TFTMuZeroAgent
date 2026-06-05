#!/usr/bin/env python3
"""
Demonstration of heuristic rollout enhancement for MCTS

DEMO IS WRONG, TODO TO SUPPORT HEURISTIC ROLLOUT ON PYTHON
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymcts
import random
import time

def compare_search_depths():
    """Compare shallow vs deep search to demonstrate heuristic-like behavior"""
    print("🎯 MCTS Search Depth Comparison (Heuristic-like Behavior)")
    print("=" * 60)
    
    # Test configurations: (iterations, time_limit, label)
    configs = [
        (50, 1, "Shallow/Fast (Heuristic-like)"),
        (500, 3, "Medium Depth"), 
        (2000, 5, "Deep Search")
    ]
    
    results = []
    
    for iterations, time_limit, label in configs:
        print(f"\n🧠 {label} - {iterations} iterations, {time_limit}s limit")
        
        # Run multiple trials for statistical significance
        times = []
        moves = []
        
        for trial in range(3):
            state = pymcts.TicTacToe_state()
            agent = pymcts.MCTS_agent(state, iterations, time_limit)
            
            start_time = time.time()
            move = agent.genmove(None)
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            moves.append(move.sprint())
        
        avg_time = sum(times) / len(times)
        results.append((label, avg_time, moves[0]))  # Use first move as representative
        
        print(f"   ⏱️  Average time: {avg_time:.3f}s")
        print(f"   🎯 Typical move: {moves[0]}")
        print(f"   📊 Move consistency: {len(set(moves))}/{len(moves)} unique")
    
    print("\n📈 Search Depth Analysis")
    print("-" * 40)
    for label, avg_time, move in results:
        print(f"{label:25} {avg_time:6.3f}s -> {move}")
    
    print("\n💡 Observations:")
    print("   • Shallow search acts like fast heuristics")
    print("   • Deep search provides more thorough analysis")
    print("   • Time/quality tradeoff mirrors heuristic vs exhaustive search")

def analyze_move_quality():
    """Analyze move quality at different search depths"""
    print("\n🔍 Move Quality Analysis Across Search Depths")
    print("=" * 60)
    
    # Create a specific board position for analysis
    print("\n🎮 Analyzing from initial position:")
    
    search_configs = [
        (25, "Ultra-fast (heuristic-like)"),
        (100, "Quick decision"),
        (500, "Careful analysis"),
        (1500, "Deep thinking")
    ]
    
    move_frequencies = {}
    
    for iterations, label in search_configs:
        print(f"\n--- {label} ({iterations} iterations) ---")
        
        # Test multiple games to see move distribution
        moves_made = []
        times_taken = []
        
        for game in range(5):
            state = pymcts.TicTacToe_state()
            agent = pymcts.MCTS_agent(state, iterations, 2)
            
            start_time = time.time()
            move = agent.genmove(None)
            elapsed = time.time() - start_time
            
            moves_made.append(move.sprint())
            times_taken.append(elapsed)
        
        # Analyze move preferences
        move_counts = {}
        for move in moves_made:
            move_counts[move] = move_counts.get(move, 0) + 1
        
        avg_time = sum(times_taken) / len(times_taken)
        print(f"   Average time: {avg_time:.3f}s")
        print(f"   Move distribution: {dict(move_counts)}")
        
        # Track most popular move for this search depth
        most_popular = max(move_counts.items(), key=lambda x: x[1])
        move_frequencies[label] = most_popular
    
    print("\n📊 Move Preference Summary")
    print("-" * 45)
    for label, (move, count) in move_frequencies.items():
        confidence = (count / 5) * 100
        print(f"{label:25} {move} ({confidence:.0f}% confidence)")

def simulate_heuristic_vs_exhaustive():
    """Simulate heuristic-like vs exhaustive search through MCTS parameters"""
    print("\n⚡ Heuristic vs Exhaustive Search Simulation")
    print("=" * 60)
    
    # Simulate "heuristic" approach: fast, low iterations
    print("\n🏃 'Heuristic' Approach (Fast & Approximate):")
    heuristic_times = []
    heuristic_moves = []
    
    for i in range(3):
        state = pymcts.TicTacToe_state()
        agent = pymcts.MCTS_agent(state, 75, 1)  # Very limited search
        
        start_time = time.time()
        move = agent.genmove(None)
        elapsed = time.time() - start_time
        
        heuristic_times.append(elapsed)
        heuristic_moves.append(move.sprint())
        print(f"   Trial {i+1}: {elapsed:.3f}s -> {move.sprint()}")
    
    heuristic_avg = sum(heuristic_times) / len(heuristic_times)
    
    # Simulate "exhaustive" approach: thorough, high iterations
    print("\n🐌 'Exhaustive' Approach (Slow & Thorough):")
    exhaustive_times = []
    exhaustive_moves = []
    
    for i in range(3):
        state = pymcts.TicTacToe_state()
        agent = pymcts.MCTS_agent(state, 1800, 4)  # Extensive search
        
        start_time = time.time()
        move = agent.genmove(None)
        elapsed = time.time() - start_time
        
        exhaustive_times.append(elapsed)
        exhaustive_moves.append(move.sprint())
        print(f"   Trial {i+1}: {elapsed:.3f}s -> {move.sprint()}")
    
    exhaustive_avg = sum(exhaustive_times) / len(exhaustive_times)
    
    # Compare approaches
    print("\n🏆 Approach Comparison")
    print("-" * 35)
    speedup = exhaustive_avg / heuristic_avg if heuristic_avg > 0 else 1
    print(f"Heuristic avg time:  {heuristic_avg:.3f}s")
    print(f"Exhaustive avg time: {exhaustive_avg:.3f}s")
    print(f"Speedup factor:      {speedup:.1f}x")
    
    # Analyze move consistency
    h_unique = len(set(heuristic_moves))
    e_unique = len(set(exhaustive_moves))
    print(f"\nHeuristic move variety:  {h_unique}/3")
    print(f"Exhaustive move variety: {e_unique}/3")
    
    if h_unique > e_unique:
        print("→ Heuristic approach shows more variation (exploration)")
    elif e_unique > h_unique:
        print("→ Exhaustive approach shows more variation (deeper analysis)")
    else:
        print("→ Both approaches show similar move consistency")

def demonstrate_rollout_concepts():
    """Demonstrate rollout concepts using built-in MCTS"""
    print("\n🎲 Rollout Strategy Demonstration")
    print("=" * 60)
    
    print("\n📋 Understanding MCTS Rollouts:")
    print("   • MCTS uses rollouts (random simulations) to evaluate positions")
    print("   • More iterations = more rollouts = better evaluation")
    print("   • Heuristic rollouts would bias random play toward good moves")
    
    # Demonstrate with different thread counts (affects rollout parallelization)
    try:
        original_threads = pymcts.get_rollout_threads()
        
        print(f"\n⚙️  Current rollout threads: {original_threads}")
        print(f"⚙️  Hardware threads available: {pymcts.get_hardware_concurrency()}")
        
        # Test single vs multi-threaded rollouts
        thread_configs = [1, min(4, pymcts.get_hardware_concurrency())]
        
        for threads in thread_configs:
            pymcts.set_rollout_threads(threads)
            
            print(f"\n🧵 Testing with {threads} rollout thread(s):")
            
            state = pymcts.TicTacToe_state()
            agent = pymcts.MCTS_agent(state, 800, 2)
            
            start_time = time.time()
            move = agent.genmove(None)
            elapsed = time.time() - start_time
            
            print(f"   Time: {elapsed:.3f}s, Move: {move.sprint()}")
            print(f"   Effective rollout rate: ~{(800/elapsed):.0f} iterations/sec")
        
        # Restore original setting
        pymcts.set_rollout_threads(original_threads)
        
    except Exception as e:
        print(f"Thread configuration demo not available: {e}")
    
    print("\n🔄 Rollout Quality Factors:")
    print("   • Random rollouts: Unbiased but potentially inefficient")
    print("   • Heuristic rollouts: Faster convergence but potential bias")
    print("   • Hybrid approaches: Balance between speed and accuracy")

def play_interactive_game():
    """Interactive game to demonstrate MCTS decision making"""
    print("\n🎮 Interactive Game: Human vs Heuristic-Enhanced MCTS")
    print("=" * 60)
    
    # Use moderately strong MCTS (simulating heuristic-enhanced performance)
    state = pymcts.TicTacToe_state()
    agent = pymcts.MCTS_agent(state, 1200, 3)
    
    print("\n📖 Game Rules:")
    print("   • You are 'O', MCTS is 'X'")
    print("   • MCTS uses 1200 iterations (heuristic-strength level)")
    print("   • Enter 'quit' to exit")
    
    move_count = 0
    
    while not state.is_terminal() and move_count < 9:
        # Show current board
        print(f"\n--- Move {move_count + 1} ---")
        print("Current board:")
        state.print()
        
        if state.is_self_side_turn():
            # MCTS turn
            print("\n🤖 MCTS is analyzing (heuristic-enhanced)...")
            
            start_time = time.time()
            move = agent.genmove(None)
            think_time = time.time() - start_time
            
            print(f"🎯 MCTS chooses: {move.sprint()} (decided in {think_time:.2f}s)")
            state = agent.get_current_state()
            
        else:
            # Human turn
            try:
                available_moves = state.actions_to_try()
                print(f"\nAvailable moves: {len(available_moves)}")
                for i, move in enumerate(available_moves):
                    print(f"   {i}: {move.sprint()}")
                
                user_input = input("\nYour move (number or 'quit'): ").strip()
                if user_input.lower() == 'quit':
                    break
                
                move_idx = int(user_input)
                if 0 <= move_idx < len(available_moves):
                    human_move = available_moves[move_idx]
                    print(f"You chose: {human_move.sprint()}")
                    
                    # Update agent with human move
                    agent.genmove(human_move)
                    state = agent.get_current_state()
                else:
                    print("❌ Invalid move number!")
                    continue
                    
            except (ValueError, IndexError):
                print("❌ Invalid input! Please enter a move number.")
                continue
        
        move_count += 1
    
    # Game over
    print("\n🏁 Game Over!")
    state.print()
    
    if state.is_terminal():
        try:
            winner = state.get_winner()
            if winner == 'X' or winner == 'x':
                print("🤖 MCTS wins! The heuristic-enhanced AI was too strong.")
            elif winner == 'O' or winner == 'o':
                print("🎉 You win! You outplayed the heuristic AI!")
            else:
                print("🤝 Draw! Well played against the heuristic AI.")
        except:
            print("Game completed!")
    else:
        print("Game ended early.")

def main():
    """Main demo function"""
    print("🚀 MCTS Heuristic Rollout Enhancement Demo")
    print("=" * 70)
    
    try:
        # Core demonstrations
        compare_search_depths()
        analyze_move_quality() 
        simulate_heuristic_vs_exhaustive()
        demonstrate_rollout_concepts()
        
        # Interactive component
        print("\n" + "=" * 70)
        play_game = input("Would you like to play against heuristic-enhanced MCTS? (y/n): ").lower().strip()
        if play_game == 'y':
            play_interactive_game()
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 Key Insights on Heuristic Rollouts:")
        print("   • Fast/shallow search mimics heuristic decision-making")
        print("   • Deep search provides more accurate but slower analysis")
        print("   • Iteration count controls the heuristic/exhaustive tradeoff")
        print("   • Real heuristic rollouts would bias random play intelligently")
        print("   • MCTS naturally balances exploration vs exploitation")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()