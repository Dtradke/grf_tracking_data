import subprocess

"""
Script to run 'play_game.py' some number of times. Passes args to play_game for game_ids and num_games.

Total number of games played is n*num_games.

"""

script_to_run = 'play_game.py'
n = 1  # Number of times to run the script

# Define the flags to pass to the script
game_id = 0
num_games = 1
args = ['--action_set', 'full', '--game_id', str(game_id), '--num_games', str(num_games)]


# Loop n times and run the script with flags using subprocess
for i in range(n):
    args = ['--action_set', 'full', '--game_id', str(game_id), '--num_games', str(num_games)]
    print(f"Running {script_to_run}, iteration {i+1}")
    subprocess.run(['python3', script_to_run] + args)

    game_id += num_games
