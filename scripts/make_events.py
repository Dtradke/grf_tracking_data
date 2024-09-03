import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


X_MAX = 120
Y_MAX = 80

MAIN_COLS = [
    "GAME_ID",
    "TIMESTEP",
    "TEAM_0_SCORE",
    "TEAM_1_SCORE",
    "PLAYER_ID",
    "TEAM_ID",
    "PLAYER_ROLE",
    "LOCATION_X",
    "LOCATION_Y",
    "LOCATION_Z",
    "VELOCITY_X",
    "VELOCITY_Y",
    "VELOCITY_Z",
    "ON_BALL",
]


def make_stints(df: pd.DataFrame, thresh: int = 5) -> pd.DataFrame:
    """
    Makes stints based on ball movement
    """

    # trim last timestep
    df = df[df["TIMESTEP"] < (df["TIMESTEP"].max())]

    distances = pd.DataFrame()
    for player_id in list(df["PLAYER_ID"].unique()):

        player_df = df[df["PLAYER_ID"] == player_id]

        player_df["NEXT_LOCATION_X"] = player_df["LOCATION_X"].shift(-1)
        player_df["NEXT_LOCATION_Y"] = player_df["LOCATION_Y"].shift(-1)
        player_df = player_df.dropna(subset="NEXT_LOCATION_X")

        player_df["DISTANCE"] = np.linalg.norm(
            player_df[["NEXT_LOCATION_X", "NEXT_LOCATION_Y"]].values
            - player_df[["LOCATION_X", "LOCATION_Y"]].values,
            axis=1,
        )

        if "TIMESTEP" not in list(distances.columns):
            distances["TIMESTEP"] = player_df["TIMESTEP"]
            distances["GAME_ID"] = player_df["GAME_ID"]

        distances[f"DISTANCE_{player_id}"] = player_df["DISTANCE"].values

    max_dist_change = pd.DataFrame()
    max_dist_change["TIMESTEP"] = distances["TIMESTEP"]
    max_dist_change["GAME_ID"] = distances["GAME_ID"]
    max_dist_change["MAX_CHANGE"] = distances[list(distances.columns)[1:]].values.max(
        axis=1
    )

    big_changes = max_dist_change[max_dist_change["MAX_CHANGE"] > thresh]

    # add last timestep
    big_changes.loc[big_changes.shape[0]] = [
        df["TIMESTEP"].max(),
        df.iloc[-1]["GAME_ID"],
        50,
    ]

    game_stints = []
    for game_id in list(big_changes["GAME_ID"].unique()):
        stint_df = big_changes[big_changes["GAME_ID"] == game_id]
        stint_df["STINT_NUMBER"] = np.arange(stint_df.shape[0])
        game_stints.append(stint_df)

    big_changes = pd.concat(game_stints)

    df = pd.merge_asof(
        df.sort_values(by=["GAME_ID", "TIMESTEP"]),
        big_changes.sort_values(by=["GAME_ID", "TIMESTEP"])[
            ["GAME_ID", "TIMESTEP", "STINT_NUMBER"]
        ],
        on="TIMESTEP",
        by=["GAME_ID"],
        direction="forward",
        suffixes=("", "_stint"),
    )

    return df[["GAME_ID", "TIMESTEP", "STINT_NUMBER"]]


def make_shots(
    df: pd.DataFrame, events: pd.DataFrame, post_y: float = 0.044
) -> pd.DataFrame:
    """
    Makes shots
    """

    possession = df[
        (df["ON_BALL"]) & (df["LOCATION_X"] > 0) & (df["LOCATION_X"] < X_MAX)
    ]

    out_of_play = df[
        (df["PLAYER_ID"] == -1)
        & ((df["LOCATION_X"] > 119) | (df["LOCATION_X"] < 1))
        & (~df["ON_BALL"])
    ]

    df = pd.merge_asof(
        possession.sort_values(by=["GAME_ID", "TIMESTEP"]),
        out_of_play.sort_values(by=["GAME_ID", "TIMESTEP"]),
        on="TIMESTEP",
        by=["GAME_ID"],
        direction="forward",
        suffixes=("", "_shot"),
    )

    df["NEXT_LOCATION_X_shot"] = df["LOCATION_X_shot"].shift(-1)

    df = df.loc[
        (df["LOCATION_X_shot"] != df["NEXT_LOCATION_X_shot"])
        & (~df["ON_BALL_shot"].astype(bool))
    ]

    df["EVENT_NAME"] = "shot"

    # remove goals that may be labeled as passes/givaways
    events = events[
        ~(
            (events["GAME_ID"].isin(list(df["GAME_ID"])))
            & events["TIMESTEP"].isin(list(df["TIMESTEP"]))
        )
    ]

    df = pd.merge_asof(
        df.sort_values(by=["GAME_ID", "TIMESTEP"]),
        events.sort_values(by=["GAME_ID", "TIMESTEP"]),
        on="TIMESTEP",
        by=["GAME_ID"],
        direction="forward",
        suffixes=("", "_nextevent"),
    )

    df["IS_GOAL"] = (df["TEAM_0_SCORE"] != df["TEAM_0_SCORE_nextevent"]) | (
        df["TEAM_1_SCORE"] != df["TEAM_1_SCORE_nextevent"]
    )

    # if shot is last thing in game, check if its in the posts
    scaled_post_high = ((post_y + 0.42) / 0.84) * 80
    scaled_post_low = ((-post_y + 0.42) / 0.84) * 80
    df.loc[
        # df["TEAM_0_SCORE_nextevent"].isna(),
        ((df["TEAM_0_SCORE_nextevent"].isna()) | df["TEAM_1_SCORE_nextevent"].isna()),
        "IS_GOAL",
    ] = (df["LOCATION_Y_shot"] >= scaled_post_low) & (
        df["LOCATION_Y_shot"] <= scaled_post_high
    )

    events["IS_GOAL"] = False

    return df[MAIN_COLS + ["EVENT_NAME", "IS_GOAL"]], events


def make_ball_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes passes table from the tracking data
    """

    # filter to on-ball events
    df = df[df["ON_BALL"]]

    # shift up player and team
    df["NEXT_PLAYER_ID"] = df["PLAYER_ID"].shift(-1)
    df["NEXT_TEAM_ID"] = df["TEAM_ID"].shift(-1)
    df["NEXT_GAME_ID"] = df["GAME_ID"].shift(-1)

    df["PREV_PLAYER_ID"] = df["PLAYER_ID"].shift(1)
    df["PREV_TEAM_ID"] = df["TEAM_ID"].shift(1)
    df["PREV_GAME_ID"] = df["GAME_ID"].shift(1)

    ball_trans = df.loc[
        (
            (df["NEXT_PLAYER_ID"] != df["PLAYER_ID"])
            | (df["PREV_PLAYER_ID"] != df["PLAYER_ID"])
        )
    ].dropna(
        subset=["NEXT_PLAYER_ID", "PREV_PLAYER_ID"]
    )  # might cut off last event of game

    # transfers between a team, not same player
    passes = ball_trans[
        (ball_trans["TEAM_ID"] == ball_trans["NEXT_TEAM_ID"])
        & (ball_trans["PLAYER_ID"] != ball_trans["NEXT_PLAYER_ID"])
        & (ball_trans["GAME_ID"] == ball_trans["NEXT_GAME_ID"])
    ]
    passes["EVENT_NAME"] = "pass"

    receptions = ball_trans[
        (ball_trans["TEAM_ID"] == ball_trans["PREV_TEAM_ID"])
        & (ball_trans["PLAYER_ID"] != ball_trans["PREV_PLAYER_ID"])
        & (ball_trans["GAME_ID"] == ball_trans["PREV_GAME_ID"])
    ]
    receptions["EVENT_NAME"] = "reception"

    # transfers from one team to another
    turnovers = ball_trans[
        (ball_trans["TEAM_ID"] != ball_trans["NEXT_TEAM_ID"])
        & (ball_trans["GAME_ID"] == ball_trans["NEXT_GAME_ID"])
    ]
    turnovers["EVENT_NAME"] = "turnover"

    # previous team id not current team
    interception = ball_trans[
        (ball_trans["TEAM_ID"] != ball_trans["PREV_TEAM_ID"])
        & (ball_trans["GAME_ID"] == ball_trans["PREV_GAME_ID"])
    ]
    interception["EVENT_NAME"] = "interception"

    events = pd.concat([passes, receptions, turnovers, interception])
    return events[MAIN_COLS + ["EVENT_NAME"]]


def scale_locations(
    df: pd.DataFrame, x_dim: int = 120, y_dim: int = 80
) -> pd.DataFrame:
    """
    Normalizes from x \in (-1,1) and y \in (-0.42, 0.42) to 120/80
    """

    # 120, 80
    df["LOCATION_X"] = ((df["LOCATION_X"] + 1) / 2) * x_dim
    df["LOCATION_Y"] = ((df["LOCATION_Y"] + 0.42) / 0.84) * y_dim

    df["VELOCITY_X"] = df["VELOCITY_X"] * x_dim
    df["VELOCITY_Y"] = df["VELOCITY_Y"] * y_dim

    return df


def simple_xg(shots: pd.DataFrame) -> pd.DataFrame:
    """
    Make simple xG model from distance and angle to goal
    """

    # flip shots going left
    shots.loc[shots["LOCATION_X"] < (X_MAX / 2), "LOCATION_Y"] = -1 * (
        shots["LOCATION_Y"] - Y_MAX
    )
    shots.loc[shots["LOCATION_X"] < (X_MAX / 2), "LOCATION_X"] = (
        X_MAX - shots["LOCATION_X"]
    )

    shots["DISTANCE_TO_NET"] = shots.apply(
        lambda x: (
            ((X_MAX - x.LOCATION_X) ** 2) + (((Y_MAX / 2) - x.LOCATION_Y) ** 2) ** 0.5
        ),
        axis=1,
    )

    shots["ANGLE_TO_NET"] = shots.apply(
        lambda x: 90
        - math.degrees(math.atan2(X_MAX - x.LOCATION_X, (Y_MAX / 2) - x.LOCATION_Y)),
        axis=1,
    )

    reg = LogisticRegression(random_state=9)

    X = shots[["DISTANCE_TO_NET", "ANGLE_TO_NET"]]
    y = shots["IS_GOAL"].values

    reg.fit(X, y)

    preds = reg.predict_proba(X)

    # return probability of goal
    return preds[:, 1]


if __name__ == "__main__":
    parser = ArgumentParser(description=f"make_events_script")
    parser.add_argument(
        "--data_path", default="data/", type=str, help="Local path to data directory"
    )
    parser.add_argument(
        "--simple_xg", action="store_true", help="Train a simple xG model."
    )
    parameters = parser.parse_args()

    games = os.listdir(parameters.data_path)

    df = pd.concat(
        [pd.read_csv(f"{parameters.data_path}{game_fname}") for game_fname in games]
    )

    # scales to new dims
    df = scale_locations(df)

    # makes stints using ball location
    stints = make_stints(df)
    events = make_ball_events(df)
    shots, events = make_shots(df, events)

    if parameters.simple_xg:
        xg_results = simple_xg(shots)
        print(f"xG: {xg_results}")

    all_events = pd.concat([events, shots]).sort_values(
        by=["GAME_ID", "TIMESTEP"], ascending=True
    )

    all_events.to_csv(f"{parameters.data_path}/events.csv")
