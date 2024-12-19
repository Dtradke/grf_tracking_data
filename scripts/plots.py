import os
from typing import Dict
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import imageio

from constants import X_MAX, Y_MAX
from utils import get_plot_colors


def plot_ball(ball_row: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Plots ball on the pitch depending on it's vertical location
    """

    ball_size = 50 if ball_row["LOCATION_Z"] < 1 else (ball_row["LOCATION_Z"] * 50)
    ax.scatter(
        ball_row["LOCATION_X"],
        ball_row["LOCATION_Y"],
        ec="k",
        fc="w",
        s=ball_size,
        zorder=110,
    )


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

        pitch = Pitch()
        _, ax = pitch.draw()

        plot_ball(ball, ax)
        plot_team(time_player_df, ax, entity_colors=entity_colors)

        ax.set_title(f"Timestep: {timestep}", fontsize=16)

        time_str = str(timestep).zfill(5)
        fname = f"{path}time_{time_str}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=100)
        plt.close()


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


def make_shot_plot(
    df: pd.DataFrame, shots: pd.DataFrame, animate_top_shots: bool = False
) -> None:
    """
    Makes shot plot diagram
    """

    if animate_top_shots:
        for _, row in shots[shots["IS_GOAL"]].head(15).iterrows():
            make_event_animation(
                df,
                game_id=row["GAME_ID"],
                shot_timestep=row["TIMESTEP"],
                animation_frames=20,
                path="tmp/goals/",
            )

    shots["MAPPED_LOCATION_X"] = shots["LOCATION_X"]
    shots["MAPPED_LOCATION_Y"] = shots["LOCATION_Y"]
    shots.loc[shots["TEAM_ID"] == 1, "MAPPED_LOCATION_Y"] = -1 * (
        shots["MAPPED_LOCATION_Y"] - Y_MAX
    )
    shots.loc[shots["TEAM_ID"] == 1, "MAPPED_LOCATION_X"] = (
        X_MAX - shots["MAPPED_LOCATION_X"]
    )

    pitch = Pitch(half=True)
    _, ax = pitch.draw()
    misses = shots[~shots["IS_GOAL"]]
    ax.scatter(
        misses["MAPPED_LOCATION_X"], misses["MAPPED_LOCATION_Y"], c="b", alpha=0.1
    )
    ax.set_title(
        f"Misses: {misses.shape[0]} ({round(100 * (misses.shape[0] / shots.shape[0]), 3)} %)"
    )
    fname = f"tmp/shots/shots_misses_half.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()

    pitch = Pitch(half=True)
    _, ax = pitch.draw()
    goals = shots[shots["IS_GOAL"]]
    ax.scatter(goals["MAPPED_LOCATION_X"], goals["MAPPED_LOCATION_Y"], c="r", alpha=0.1)
    ax.set_title(
        f"Goals: {goals.shape[0]} ({round(100 * (goals.shape[0] / shots.shape[0]), 3)} %)"
    )

    fname = f"tmp/shots/shots_goals_half.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()


def make_xg_plot(shots: pd.DataFrame) -> None:
    """
    Makes shot plot diagram
    """

    pitch = Pitch(half=True)
    fig, ax = pitch.draw()
    im_obj = ax.scatter(
        shots["LOCATION_X"],
        shots["LOCATION_Y"],
        c=shots["XG"],
        cmap="Reds",
        alpha=0.3,
        s=60,
        vmin=0.0,
        zorder=0,
    )
    cbar = fig.colorbar(im_obj, pad=0.0001, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("EXPECTED GOALS", fontsize=16, rotation=270)

    fname = f"tmp/shots/shots_xg.png"
    os.makedirs("tmp/shots", exist_ok=True)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()


def make_event_animation(
    df: pd.DataFrame,
    game_id: int,
    shot_timestep: int,
    animation_frames: int = 20,
    path: str = "tmp/shots/",
) -> None:
    """
    Plots event that happens at game_id and shot_timestep
    """
    print(f"Making animation for game {game_id} timestep {shot_timestep}")

    entity_colors = get_plot_colors(df)

    events_df = df[
        (df["GAME_ID"] == game_id)
        & (df["TIMESTEP"] >= shot_timestep)
        & (df["TIMESTEP"] <= (shot_timestep + animation_frames))
    ]
    make_plots(events_df, entity_colors, f"{path}{game_id}_{shot_timestep}/")


def make_pass_plot(
    df: pd.DataFrame,
    events: pd.DataFrame,
    num_animations: int = 10,
    path: str = "tmp/passes/",
) -> None:
    """
    Makes pass plot animations and diagram
    """

    print(f"Making {num_animations} animations of passes")

    if not os.path.exists(path):
        os.makedirs(path)

    events = events.sort_values(by=["GAME_ID", "TIMESTEP"], ascending=True)

    events["NEXT_LOCATION_X"] = events["LOCATION_X"].shift(-1)
    events["NEXT_LOCATION_Y"] = events["LOCATION_Y"].shift(-1)
    events["NEXT_EVENT_NAME"] = events["EVENT_NAME"].shift(-1)
    events["NEXT_TIMESTEP"] = events["TIMESTEP"].shift(-1)
    passes = events[
        (events["EVENT_NAME"] == "pass") & (events["NEXT_EVENT_NAME"] == "reception")
    ]

    for _, row in passes.head(num_animations).iterrows():
        pass_duration = row["NEXT_TIMESTEP"] - row["TIMESTEP"]
        make_event_animation(
            df,
            row["GAME_ID"],
            row["TIMESTEP"],
            animation_frames=pass_duration + 5,  # plot pass duration + 5 timesteps
            path="tmp/passes/",
        )

    passes["DX"] = passes["NEXT_LOCATION_X"] - passes["LOCATION_X"]
    passes["DY"] = passes["NEXT_LOCATION_Y"] - passes["LOCATION_Y"]

    pitch = Pitch()
    _, ax = pitch.draw()
    team_zero = passes[passes["TEAM_ID"] == 0]
    for _, row in team_zero.iterrows():
        ax.arrow(
            row["LOCATION_X"],
            row["LOCATION_Y"],
            row["DX"],
            row["DY"],
            head_width=1,
            fc="b",
            ec="b",
            alpha=0.1,
        )

    ax.set_title(f"Team 0 Passes: {team_zero.shape[0]}")

    fname = f"{path}/team_0_passes.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()

    pitch = Pitch()
    _, ax = pitch.draw()
    team_one = passes[passes["TEAM_ID"] == 1]
    for _, row in team_one.iterrows():
        ax.arrow(
            row["LOCATION_X"],
            row["LOCATION_Y"],
            row["DX"],
            row["DY"],
            head_width=1,
            fc="r",
            ec="r",
            alpha=0.1,
        )

    ax.set_title(f"Team 1 Passes: {team_one.shape[0]}")
    fname = f"{path}/team_1_passes.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()


def plot_ts_pitch(tracking_zero, tracking_one, tracking_ball):
    pitch = Pitch()
    fig, ax = pitch.draw()

    entity_colors = get_plot_colors(pd.concat([tracking_zero, tracking_one]))
    plot_team(tracking_zero, ax, entity_colors)
    plot_team(tracking_one, ax, entity_colors)
    plot_ball(tracking_ball.iloc[0], ax)
    plt.show()
    exit()


def plot_pitchcontrol_timestep(
    tracking_zero,
    tracking_one,
    tracking_ball,
    PPCF,
    path: str = "tmp/pitch_control/",
    show: bool = False,
    title_str: str = None,
):
    """
    Plots the pitch control at a specific timestep of the game
    """
    pitch = Pitch()
    fig, ax = pitch.draw()

    entity_colors = get_plot_colors(pd.concat([tracking_zero, tracking_one]))
    plot_team(tracking_zero, ax, entity_colors)
    plot_team(tracking_one, ax, entity_colors)
    plot_ball(tracking_ball.iloc[0], ax)

    cmap = "bwr_r"
    im_obj = ax.imshow(
        np.flipud(PPCF),
        extent=(0, X_MAX, 0, Y_MAX),
        interpolation="spline36",
        vmin=0.0,
        vmax=1.0,
        cmap="bwr_r",
        alpha=0.6,
    )

    cbar = fig.colorbar(im_obj, fraction=0.029, pad=0.01, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Team Control", fontsize=16, rotation=270)

    game_id = tracking_zero.iloc[0]["GAME_ID"]
    timestep = tracking_zero.iloc[0]["TIMESTEP"]

    if title_str is None:
        title_str = f"Game: {game_id}, TS: {timestep}"

    fig.suptitle(title_str, fontsize=14, y=0.88)

    if show:
        plt.show()
        plt.close()
    else:
        path = f"{path}game_{game_id}/"
        if not os.path.exists(path):
            os.makedirs(path)

        time_str = str(timestep).zfill(5)
        fname = f"{path}time_{time_str}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=100)
        plt.close()
        print(f"Saved: {fname}")


def create_gifs(input_dir, duration=0.2):
    """
    Create GIFs for all game folders containing PNG images.

    Args:
        input_dir (str): Path to the base folder containing game subfolders.
        duration (float): Duration between frames in seconds.
    """
    for game_folder in os.listdir(input_dir):
        game_path = os.path.join(input_dir, game_folder)

        if os.path.isdir(game_path):
            images = []
            for filename in sorted(os.listdir(game_path)):
                if filename.endswith(".png") and filename.startswith("time_"):
                    images.append(os.path.join(game_path, filename))

            if images:
                output_file = os.path.join(game_path, "animation.gif")
                frames = [imageio.imread(img) for img in images]
                imageio.mimsave(output_file, frames, duration=duration, loop=0)
                print(f"GIF created for {game_folder} and saved as {output_file}")
            else:
                print(f"No PNG files found in {game_folder}")
