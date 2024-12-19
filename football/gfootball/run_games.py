"""
Script to run 'play_game.py' some number of times. Passes args to play_game for game_ids and num_games.

Total number of games played is num_iterations*num_games.

Example run (inside docker):
python3 run_games.py --num_games=3
"""

import subprocess
import argparse

PARSER = argparse.ArgumentParser("collect_simulated_tracking_script")
PARSER.add_argument(
    "--num_games", type=int, default=3, help="Number of games to collect."
)
PARSER.add_argument(
    "--start_game_id", type=int, default=0, help="Which game id to start with."
)
PARSER.add_argument(
    "--num_iterations", type=int, default=1, help="Number of times to run the script."
)
ARGS = PARSER.parse_args()

SIMULATION_SCRIPT = "play_game.py"

# Define the flags to pass to the script
args = [
    "--action_set",
    "full",
    "--game_id",
    str(ARGS.start_game_id),
    "--num_games",
    str(ARGS.num_games),
]

# Loop `ARGS.num_iterations` times and run the script with flags using subprocess
game_id = ARGS.start_game_id
for i in range(ARGS.num_iterations):
    args = [
        "--action_set",
        "full",
        "--game_id",
        str(game_id),
        "--num_games",
        str(ARGS.num_games),
    ]
    print(f"Running {SIMULATION_SCRIPT}, iteration {i+1}")
    subprocess.run(["python3", SIMULATION_SCRIPT] + args)

    game_id += ARGS.num_games
