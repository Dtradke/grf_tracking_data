"""
This script trains a simple expected goals model given shot events and their angle/distance from the goal.

If the shot events have not been saved to `event_path/shots.csv`, this script will first launch EventMaker
"""

import os
from argparse import ArgumentParser
import pandas as pd
import warnings

from make_events import make_and_save_events
from plots import make_xg_plot
from models import expected_goals_model

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = ArgumentParser(description="expected_goals_model")
    parser.add_argument(
        "--data_path", default="data/", type=str, help="Local path to data directory"
    )
    parser.add_argument(
        "--event_path", default="events/", type=str, help="Local path to data directory"
    )
    parameters = parser.parse_args()

    if not os.path.exists(f"{parameters.event_path}shots.csv"):
        make_and_save_events(parameters.event_path, parameters.data_path)

    shots = pd.read_csv(f"{parameters.event_path}shots.csv")

    print("Making simple xG...")
    shots["XG"] = expected_goals_model(shots)

    print(shots.sort_values(by="XG", ascending=False))
    print(f"Sum of xG: {shots['XG'].sum()}")
    print(f"Total Goals: {shots[shots['IS_GOAL']].shape[0]}")

    make_xg_plot(shots)
