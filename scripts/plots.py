import os
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt

from constants import X_MAX, Y_MAX
from analyze_tracking import get_plot_colors, make_plots


def make_shot_plot(
    df: pd.DataFrame, shots: pd.DataFrame, animate_top_shots: bool = False
) -> None:
    """
    Makes shot plot diagram
    """

    if animate_top_shots:
        for _, row in shots[shots["IS_GOAL"]].head(15).iterrows():
            make_shot_animation(
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
    cbar = fig.colorbar(im_obj, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("EXPECTED GOAL", fontsize=12, rotation=270)

    ax.set_title(f"Shot Expected Goals (xG)")

    fname = f"tmp/shots/shots_xg.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()


def make_shot_animation(
    df: pd.DataFrame,
    game_id: int,
    shot_timestep: int,
    animation_frames: int = 20,
    path: str = "tmp/shots/",
) -> None:
    """
    Plots shot that happens at game_id and shot_timestep
    """
    print(f"Making shot animation for game {game_id} timestep {shot_timestep}")

    entity_colors = get_plot_colors(df)

    shot_df = df[
        (df["GAME_ID"] == game_id)
        & (df["TIMESTEP"] >= shot_timestep)
        & (df["TIMESTEP"] <= (shot_timestep + animation_frames))
    ]
    make_plots(shot_df, entity_colors, f"{path}{game_id}_{shot_timestep}/")


def make_pass_plot(
    df: pd.DataFrame, events: pd.DataFrame, path: str = "tmp/passes/"
) -> None:
    """
    Makes shot plot diagram
    """

    if not os.path.exists(path):
        os.makedirs(path)

    events = events.sort_values(by=["GAME_ID", "TIMESTEP"], ascending=True)

    events["NEXT_LOCATION_X"] = events["LOCATION_X"].shift(-1)
    events["NEXT_LOCATION_Y"] = events["LOCATION_Y"].shift(-1)
    events["NEXT_EVENT_NAME"] = events["EVENT_NAME"].shift(-1)
    passes = events[
        (events["EVENT_NAME"] == "pass") & (events["NEXT_EVENT_NAME"] == "reception")
    ]

    passes = passes[
        (passes["LOCATION_X"] > (X_MAX / 2))
        & (passes["LOCATION_X"] < ((X_MAX / 2) + 10))
        & (passes["NEXT_LOCATION_Y"] > (Y_MAX / 2))
        & (passes["NEXT_LOCATION_X"] < 67)
        & (passes["NEXT_LOCATION_Y"] < 58)
    ]

    for _, row in passes.head(10).iterrows():
        make_shot_animation(
            df, row["GAME_ID"], row["TIMESTEP"], animation_frames=10, path="tmp/passes/"
        )

    print(passes)
    exit()

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

    plt.show()
    exit()

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
