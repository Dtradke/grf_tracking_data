# grf_tracking_data

This repository accompanies the AAMAS 2025 Demo paper "Simulating Tracking Data to Advance Sports Analytics Research." We utilize Google Research Football reinforcement learning environment to record simulated football (soccer) tracking data in a similar schema to real-world player tracking data.

Find the demo video [here](https://www.youtube.com/watch?v=2pjyxfPVsuw).

The demo includes:
- Link to a Google Drive that contains 3,000 pre-recorded simulated games
- Scripts to record new tracking data and save locally
- Scripts to train and save an expected goals (xG) model using logistic regression
- An implementation of pitch control [(Spearman et al., 2017)](https://www.researchgate.net/profile/William-Spearman/publication/315166647_Physics-Based_Modeling_of_Pass_Probabilities_in_Soccer/links/58cbfca2aca272335513b33c/Physics-Based-Modeling-of-Pass-Probabilities-in-Soccer.pdf)
- A unified demo script that uses logic to extract and plot/animate passes and shots

## Installation

Create a virtual environment to install the project's requirements. Additionally, collecting new data (step (4) below) will require Docker to be installed. To install the requirements for this project, navigate to root and do the following commands:
```sh
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install -r requirements.txt
```
Deactivate your virtual env by running `deactivate`.

## Usage and Demo Scripts

Download data files from ... to `grf_tracking_data/data/`. All of the commands listed in this section (except for recording new data) are designed to run from the root of this directory (i.e., `grf_tracking_data/`). Data and models are designed to be saved to a temp directory called: `grf_tracking_data/tmp/`.

#### 1. Collect Data or Download Pre-Recorded Games

To make events, train an expected goals (xG) model, or run pitch control, you will first need simulated tracking data. You have two options to obtain simulated tracking data. 1) You can proceed to step (4) and follow the process of simulating new tracking data or 2) you can download the pre-saved dataset located at the Google Drive linked above. Move the `.csv` files into a data directory from root.

#### 2. Making Events

To make events, run the `make_events.py` script from the root directory and include the local path to where you included the data (i.e., `data/` in this case). For example:

```
python3 scripts/make_events.py --data_path="data/"
```

#### 3. Training an xG Model

Record some data for a training dataset using the process in step (4) below or download the pre-saved dataset located at the Google Drive linked above. Move the `.csv` files into a data directory from root. From root, you can call the following script to train an xG model on that pre-saved dataset (i.e., `data/` in this case):
```
python3 scripts/expected_goals.py --data_path="data/" --event_path="events/"
```

#### 4. Recording New Simulated Tracking Data

This process runs inside a Docker container. Navigate from `grf_tracking_data/` to `football/` to record new data. Steps shown below to record 100 games (also shown in the video linked above):
```sh
$ cd football/
$ sudo docker-compose run app
$ (enter password)
$ cd gfootball
$ python3 run_games.py --num_games=100
```

#### 5. Run Full Demo on New Dataset

This demo can run on any dataset that is separate from the one used to train the xG model (to avoid in-sample prediction). In this case, we will point towards the new dataset we recorded in the previous process by passing the path to where that data was stored: `--data_path="football/gfootball/results/"`. We point the script to the local file of the pre-saved xG model. The process extracts all events from the games found in `data_path` and calculates xG on each shot and pitch control on various events.

```
Python3 scripts/full_demo.py --data_path="football/gfootball/results/" --xg_model_fname="tmp/models/xg/xg_model.pkl"
```

## Schema Details

The schema for these recorded games are designed to imitate real-world tracking data. We detail the schema below:

| Column Name        | Type           | Description  |
| ------------- |:-------------| :----- |
| `GAME_ID`       | `int` | Unique game identifier. |
| `TIMESTEP`      | `int` | Timestep of the game (increasing from 0). |
| `TEAM_O_SCORE`  | `int` | Score of team 0 at each timestep. |
| `TEAM_1_SCORE`  | `int` | Score of team 1 at each timestep. |
| `PLAYER_ID`     | `int` | Unique player identifier. Players on team 0 range from 0-10. Players on team 1 range from 100-110. The ball is -1. |
| `TEAM_ID`       | `int` | Unique team identifier of either 0 or 1 for each agent. The ball is -1. |
| `PLAYER_ROLE`   | `str` | Agent position/role on their team. The ball is the only type exception and is -1. |
| `LOCATION_X`    | `float` | Agent/ball location along the x-axis (i.e., endline to endline of the pitch). |
| `LOCATION_Y`    | `float` | Agent/ball location along the y-axis (i.e., sideline to sideline of the pitch). |
| `LOCATION_Z`    | `float` | Agent/ball location along the z-axis (i.e., off the ground). |
| `VELOCITY_X`    | `float` | Agent/ball velocity along the x-axis. |
| `VELOCITY_Y`    | `float` | Agent/ball velocity along the y-axis. |
| `VELOCITY_Z`    | `float` | Agent/ball velocity along the z-axis. |
| `ON_BALL`       | `bool`  | Indicates if an agent is in possession of the ball. This feature is always `False` for the ball. |

## Reference

When referencing this dataset, please use the following citation:

```
@article{RadkeTilbury2025SimTracking,
  title={Simulating Tracking Data to Advance Sports Analytics Research},
  author={Radke, David and Tilbury, Kyle},
  booktitle={Proceedings of the 24th International Conference on Autonomous Agents and MultiAgent Systems},
  journal={AAMAS},
  year={2025}
}
```
