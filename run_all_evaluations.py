"""Run all evaluation comparisons: MCTS vs Random, MCTS vs PPO, PPO vs Random, DQN vs Random, DQN vs PPO, DQN vs MCTS."""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print the result."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all evaluation comparisons")
    parser.add_argument("--mcts-model", type=str, default="poker-rl/output/models/poker_model.pt",
                        help="Path to MCTS model (default: poker-rl/output/models/poker_model.pt)")
    parser.add_argument("--ppo-model", type=str, default="poker-rl/output/models/ppo_poker_model.pt",
                        help="Path to PPO model (default: poker-rl/output/models/ppo_poker_model.pt)")
    parser.add_argument("--dqn-model", type=str, default="poker-rl/output/models/dqn_poker_model.pt",
                        help="Path to DQN model (default: poker-rl/output/models/dqn_poker_model.pt)")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games per evaluation (default: 100)")
    parser.add_argument("--hands-per-game", type=int, default=None,
                        help="Number of hands per game (default: from config)")
    parser.add_argument("--skip-mcts-random", action="store_true",
                        help="Skip MCTS vs Random evaluation")
    parser.add_argument("--skip-mcts-ppo", action="store_true",
                        help="Skip MCTS vs PPO evaluation")
    parser.add_argument("--skip-ppo-random", action="store_true",
                        help="Skip PPO vs Random evaluation")
    parser.add_argument("--skip-dqn-random", action="store_true",
                        help="Skip DQN vs Random evaluation")
    parser.add_argument("--skip-dqn-ppo", action="store_true",
                        help="Skip DQN vs PPO evaluation")
    parser.add_argument("--skip-dqn-mcts", action="store_true",
                        help="Skip DQN vs MCTS evaluation")
    
    args = parser.parse_args()
    
    # Build base arguments
    games_arg = ["--games", str(args.games)]
    hands_arg = []
    if args.hands_per_game:
        hands_arg = ["--hands-per-game", str(args.hands_per_game)]
    
    print("=" * 80)
    print("RUNNING ALL EVALUATIONS")
    print("=" * 80)
    print(f"MCTS Model: {args.mcts_model}")
    print(f"PPO Model: {args.ppo_model}")
    print(f"DQN Model: {args.dqn_model}")
    print(f"Games per evaluation: {args.games}")
    if args.hands_per_game:
        print(f"Hands per game: {args.hands_per_game}")
    print("=" * 80)
    
    results = {}
    
    # 1. MCTS vs Random
    if not args.skip_mcts_random:
        cmd = ["python3", "evaluate.py", "--model", args.mcts_model, "--vs-random"] + games_arg + hands_arg
        results["MCTS vs Random"] = run_command(cmd, "MCTS vs Random")
    else:
        print("\nSkipping MCTS vs Random")
    
    # 2. MCTS vs PPO
    if not args.skip_mcts_ppo:
        cmd = ["python3", "compare_ppo_mcts.py", 
               "--ppo-model", args.ppo_model,
               "--mcts-model", args.mcts_model] + games_arg + hands_arg
        results["MCTS vs PPO"] = run_command(cmd, "MCTS vs PPO")
    else:
        print("\nSkipping MCTS vs PPO")
    
    # 3. PPO vs Random
    if not args.skip_ppo_random:
        cmd = ["python3", "evaluate_ppo.py", "--model", args.ppo_model, "--vs-random"] + games_arg + hands_arg
        results["PPO vs Random"] = run_command(cmd, "PPO vs Random")
    else:
        print("\nSkipping PPO vs Random")
    
    # 4. DQN vs Random
    if not args.skip_dqn_random:
        cmd = ["python3", "evaluate_dqn.py", "--model", args.dqn_model, "--vs-random"] + games_arg + hands_arg
        results["DQN vs Random"] = run_command(cmd, "DQN vs Random")
    else:
        print("\nSkipping DQN vs Random")
    
    # 5. DQN vs PPO
    if not args.skip_dqn_ppo:
        cmd = ["python3", "compare_dqn_ppo.py",
               "--dqn-model", args.dqn_model,
               "--ppo-model", args.ppo_model] + games_arg + hands_arg
        results["DQN vs PPO"] = run_command(cmd, "DQN vs PPO")
    else:
        print("\nSkipping DQN vs PPO")
    
    # 6. DQN vs MCTS
    if not args.skip_dqn_mcts:
        cmd = ["python3", "compare_dqn_mcts.py",
               "--dqn-model", args.dqn_model,
               "--mcts-model", args.mcts_model] + games_arg + hands_arg
        results["DQN vs MCTS"] = run_command(cmd, "DQN vs MCTS")
    else:
        print("\nSkipping DQN vs MCTS")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for eval_name, success in results.items():
        status = "SUCCESS" if success else "FAILED" if success is False else "SKIPPED"
        print(f"{eval_name}: {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()

