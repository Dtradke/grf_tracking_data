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

## Usage

Download data files from ... to `grf_tracking_data/data/`. All of the commands listed in this section (except for recording new data) are designed to run from the root of this directory (i.e., `grf_tracking_data/`). Data and models are designed to be saved to a temp directory called: `grf_tracking_data/tmp/`.

#### Making Events

To make events, run the `make_events.py` script from the root directory and include the local path to where you included the data (i.e., `data/` in this case). For example:

```
python3 scripts/make_events.py --data_path="data/"
```

#### Training an xG Model

Pre-record simulated data or download the pre-saved dataset located at the Google Drive linked above. Move the `.csv` files into a data directory from root. From root, you can call the following script to train an xG model on that pre-saved dataset (i.e., `data/` in this case):
```
python3 scripts/expected_goals.py --data_path="data/" --event_path="events/"
```

#### Recording New Simulated Tracking Data

This process runs inside a Docker container. Navigate from `grf_tracking_data/` to `football/` to record new data. Steps shown below to record 100 games (also shown in the video linked above):
```sh
$ cd football/
$ sudo docker-compose run app
$ (enter password)
$ cd gfootball
$ python3 run_games.py --num_games=100
```

#### Run Full Demo on New Dataset

This demo can run on any dataset that is separate from the one used to train the xG model (to avoid in-sample prediction). In this case, we will point towards the new dataset we recorded in the previous process by passing the path to where that data was stored: `--data_path="football/gfootball/results/"`. We point the script to the local file of the pre-saved xG model. The process extracts all events from the games found in `data_path` and calculates xG on each shot and pitch control on various events.

```
Python3 scripts/full_demo.py --data_path="football/gfootball/results/" --xg_model_fname="tmp/models/xg/xg_model.pkl"
```

## Reference

When referencing this dataset, please use the following citation:

```
@inproceedings{RadkeTilburySimTracking2025,
  title={Simulating Tracking Data to Advance Sports Analytics Research},
  author={Radke, David and Tilbury, Kyle},
  booktitle={Autonomous Agents and Multiagent Systems},
  year={2025}
}
```
