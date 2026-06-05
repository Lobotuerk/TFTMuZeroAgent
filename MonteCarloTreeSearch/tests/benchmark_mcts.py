import time
import sys
import os
import numpy as np

# Ensure pymcts can be imported from current directory or parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(script_dir)
sys.path.append(parent_dir)

try:
    import pymcts
except ImportError:
    print("Error: pymcts module not found. Please build it first.")
    sys.exit(1)

def benchmark_genmove(iterations_list=[1000, 10000, 50000]):
    print("\n--- Benchmarking MCTS_agent.genmove() ---")
    print(f"{'Iterations':>12} | {'Time (s)':>10} | {'Moves/sec':>12}")
    print("-" * 40)
    
    for iterations in iterations_list:
        state = pymcts.TicTacToe_state()
        # MCTS_agent(state, iterations, time_limit)
        agent = pymcts.MCTS_agent(state, iterations, 1000) # Large time limit to ensure iteration limit is hit
        
        start_time = time.time()
        agent.genmove(None)
        elapsed = time.time() - start_time
        
        moves_per_sec = iterations / elapsed if elapsed > 0 else float('inf')
        print(f"{iterations:12,d} | {elapsed:10.4f} | {moves_per_sec:12.2f}")

def benchmark_rollout_throughput(duration=2.0):
    print("\n--- Benchmarking Rollout Throughput ---")
    state = pymcts.TicTacToe_state()
    
    count = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        state.rollout()
        count += 1
    
    elapsed = time.time() - start_time
    throughput = count / elapsed
    print(f"Total rollouts: {count:,d}")
    print(f"Elapsed time: {elapsed:.4f} s")
    print(f"Rollout throughput: {throughput:.2f} simulations/sec")

def benchmark_parallel_performance():
    print("\n--- Benchmarking Parallel vs Single-threaded Performance ---")
    iterations = 20000
    hardware_threads = pymcts.get_hardware_concurrency()
    optimal_threads = pymcts.get_optimal_thread_count()
    
    threads_to_test = [1]
    if hardware_threads > 1:
        threads_to_test.append(2)
    if optimal_threads > 2:
        threads_to_test.append(optimal_threads)
    if hardware_threads > optimal_threads:
        threads_to_test.append(hardware_threads)
    
    # Remove duplicates and sort
    threads_to_test = sorted(list(set(threads_to_test)))
    
    print(f"{'Threads':>8} | {'Time (s)':>10} | {'Speedup':>8} | {'Efficiency':>10}")
    print("-" * 45)
    
    baseline_time = None
    
    for t in threads_to_test:
        pymcts.set_rollout_threads(t)
        actual_threads = pymcts.get_rollout_threads()
        
        state = pymcts.TicTacToe_state()
        agent = pymcts.MCTS_agent(state, iterations, 1000)
        
        start_time = time.time()
        agent.genmove(None)
        elapsed = time.time() - start_time
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
            efficiency = 1.0
        else:
            speedup = baseline_time / elapsed
            efficiency = speedup / t
            
        print(f"{t:8d} | {elapsed:10.4f} | {speedup:8.2f} | {efficiency:10.2f}")
    
    # Reset to 1 thread
    pymcts.set_rollout_threads(1)

if __name__ == "__main__":
    print("MCTS Standardized Benchmark Suite")
    print("=" * 40)
    print(f"Hardware Concurrency: {pymcts.get_hardware_concurrency()}")
    print(f"Optimal Thread Count: {pymcts.get_optimal_thread_count()}")
    
    benchmark_genmove()
    benchmark_rollout_throughput()
    benchmark_parallel_performance()
