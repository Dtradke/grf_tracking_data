import os
from argparse import ArgumentParser
from typing import Dict
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import warnings

from utils import get_plot_colors, scale_locations
from plots import plot_team, make_plots

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = ArgumentParser(description="analyze_tracking_script")
    parser.add_argument(
        "--data_path", default="data/", type=str, help="Local path to data directory"
    )
    parameters = parser.parse_args()

    games = os.listdir(parameters.data_path)

    df = pd.concat(
        [pd.read_csv(f"{parameters.data_path}{game_fname}") for game_fname in games]
    )

    entity_colors = get_plot_colors(df)
    df = scale_locations(df)
    make_plots(df, entity_colors=entity_colors, path="tmp/expanded_tracking_figs/")
