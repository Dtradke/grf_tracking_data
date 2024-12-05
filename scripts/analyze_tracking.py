import os
from argparse import ArgumentParser
from typing import Dict
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def plot_team(df: pd.DataFrame, ax: plt.Axes, entity_colors: Dict[int, str]) -> None:
    """Plots team given IDs and velo"""

    for _, row in df.iterrows():

        edge_color = "k" if not row["ON_BALL"] else "gold"
        props = dict(
            boxstyle="circle",
            pad=0.1,
            facecolor=entity_colors[row["TEAM_ID"]],
            ec=edge_color,
        )

        number = str(row["PLAYER_ID"]).zfill(3)
        ax.arrow(
            row["LOCATION_X"] - 1,
            row["LOCATION_Y"],
            row["VELOCITY_X"] * 3,
            row["VELOCITY_Y"] * 3,
            length_includes_head=True,
            color=entity_colors[row["TEAM_ID"]],
            ec="k",
            width=0.5,
            head_width=1,
            zorder=99,
        )

        ax.text(
            row["LOCATION_X"] - 3,
            row["LOCATION_Y"],
            number,
            color=edge_color,
            fontsize=8,
            bbox=props,
            zorder=100,
        )


def normalize_locations(
    df: pd.DataFrame, x_dim: int = 120, y_dim: int = 80
) -> pd.DataFrame:
    """
    Locations out of GFootball are at weird scale. Map to pitch dimensions
    """

    # 120, 80
    df["LOCATION_X"] = (
        (df["LOCATION_X"] - df["LOCATION_X"].min())
        / (df["LOCATION_X"].max() - df["LOCATION_X"].min())
    ) * x_dim
    df["LOCATION_Y"] = (
        (df["LOCATION_Y"] - df["LOCATION_Y"].min())
        / (df["LOCATION_Y"].max() - df["LOCATION_Y"].min())
    ) * y_dim

    df["VELOCITY_X"] = df["VELOCITY_X"] * x_dim
    df["VELOCITY_Y"] = df["VELOCITY_Y"] * y_dim
    return df


def get_plot_colors(df: pd.DataFrame) -> Dict[int, str]:
    """
    Gets plot colors for two teams
    """

    player_df = df[df["TEAM_ID"] != -1]
    colors = ["b", "r"]
    return {
        team_id: colors[team_count]
        for team_count, team_id in enumerate(player_df["TEAM_ID"].dropna().unique())
    }


def make_plots(
    df: pd.DataFrame,
    entity_colors: Dict[int, str],
    path: str = "tmp/builtin_ai_tracking_figs/",
) -> None:
    """
    Main loop/function to make plots for each timestep
    """

    if not os.path.exists(path):
        os.makedirs(path)

    for timestep in list(df["TIMESTEP"].unique()):
        time_df = df[df["TIMESTEP"] == timestep]

        ball = time_df[time_df["PLAYER_ID"] == -1].iloc[0]
        time_player_df = time_df[time_df["TEAM_ID"] != -1]

        pitch = Pitch(pitch_color="grass", line_color="white", stripe=True)
        _, ax = pitch.draw()

        ball_size = 50 if ball["LOCATION_Z"] < 1 else (ball["LOCATION_Z"] * 50)
        ax.scatter(
            ball["LOCATION_X"],
            ball["LOCATION_Y"],
            ec="k",
            fc="w",
            s=ball_size,
            zorder=110,
        )
        plot_team(time_player_df, ax, entity_colors=entity_colors)

        ax.set_title(f"Timestep: {timestep}", fontsize=16)

        time_str = str(timestep).zfill(5)
        fname = f"{path}time_{time_str}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=100)
        plt.close()


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
    df = normalize_locations(df)
    make_plots(df, entity_colors=entity_colors, path="tmp/expanded_tracking_figs/")
