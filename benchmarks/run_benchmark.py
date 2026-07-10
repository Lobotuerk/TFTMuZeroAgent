import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.core import BenchmarkRunner
from benchmarks.report import BenchmarkReport


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="TFT MuZero Agent - Benchmark and Profiling Suite"
    )
    parser.add_argument(
        '--num-games', type=int, default=1,
        help='Number of games to run (default: 1)',
    )
    parser.add_argument(
        '--steps-per-game', type=int, default=50,
        help='Maximum steps per game (default: 50, 0 for infinite)',
    )
    parser.add_argument(
        '--agent-setup', type=str, default='muzero_vs_random',
        choices=['muzero_vs_random', 'buying_agents', 'tournament'],
        help='Agent configuration to run (default: muzero_vs_random)',
    )
    parser.add_argument(
        '--mcts-simulations', type=int, default=10,
        help='Number of MCTS simulations for MuZero agent (default: 10)',
    )
    parser.add_argument(
        '--real-env', action='store_true',
        help='Use the real TFTSet4Gym simulator instead of BenchmarkMockEnv',
    )
    parser.add_argument(
        '--deep-mcts', action='store_true',
        help='Enable deep MCTS profiling (monkey-patches EnhancedMCTS methods)',
    )
    parser.add_argument(
        '--compare-with', type=str, default=None,
        help='Path to a reference JSON file for regression comparison',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to write JSON results (default: benchmarks/results/benchmark-<commit>-<timestamp>.json)',
    )
    return parser.parse_args(argv)


def default_output_path() -> str:
    import subprocess
    import time
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    timestamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'benchmarks', 'results',
    )
    return os.path.join(results_dir, f'benchmark-{commit}-{timestamp}.json')


def main(argv=None):
    args = parse_args(argv)

    runner = BenchmarkRunner(
        use_mock_env=not args.real_env,
        num_games=args.num_games,
        steps_per_game=args.steps_per_game,
        agent_setup=args.agent_setup,
        mcts_simulations=args.mcts_simulations,
        deep_mcts=args.deep_mcts,
    )

    print(f"Running benchmark: {args.num_games} game(s), {args.steps_per_game} steps/game, "
          f"setup={args.agent_setup}, mock_env={not args.real_env}, deep_mcts={args.deep_mcts}")
    results = runner.run()

    output_path = args.output or default_output_path()
    report = BenchmarkReport()
    report.save(results, output_path)
    print(f"Results saved to: {output_path}")

    reference = None
    if args.compare_with:
        try:
            reference = report.load(args.compare_with)
            print(f"Loaded reference: {args.compare_with}")
        except Exception as e:
            print(f"Warning: could not load reference file: {e}")

    markdown = report.generate_markdown(results, reference=reference)
    print("\n" + markdown)

    return results


if __name__ == '__main__':
    main()
