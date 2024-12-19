"""
This script loads all the raw tracking files and saves event/high-level data.

stints.csv: continuous windows of play (i.e., play start until a shot)
shots.csv: events that signify a shot event at a specific GAME_ID/TIMESTEP
event.csv: events that signify a pass, reception, interception, or turnover at a specific GAME_ID/TIMESTEP
"""

import os
from argparse import ArgumentParser
import pandas as pd
import warnings

from event_maker import EventMaker

warnings.filterwarnings("ignore")


def make_and_save_events(event_path: str, data_path: str) -> None:
    """
    If you haven't run/saved event scripts previously, process and save them now
    """
    # collect files in path
    games = os.listdir(data_path)

    # collect data
    df = pd.concat(
        [pd.read_csv(f"{data_path}{game_fname}") for game_fname in games]
    ).sort_values(by=["GAME_ID", "TIMESTEP"], ascending=True)

    print(f"Loaded data from {df['GAME_ID'].unique().shape[0]} games")

    event_maker = EventMaker(df)

    if not os.path.exists(event_path):
        os.makedirs(event_path)

    # save data locally
    event_maker.stints.to_csv(f"{event_path}stints.csv")
    event_maker.events.to_csv(f"{event_path}events.csv")
    event_maker.shots.to_csv(f"{event_path}shots.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="make_events_script")
    parser.add_argument(
        "--data_path", default="data/", type=str, help="Local path to data directory"
    )
    parser.add_argument(
        "--event_path", default="events/", type=str, help="Local path to data directory"
    )
    parameters = parser.parse_args()

    # make events and save locally
    make_and_save_events(parameters.event_path, parameters.data_path)
