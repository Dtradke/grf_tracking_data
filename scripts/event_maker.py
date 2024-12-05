import pandas as pd
import numpy as np

from constants import X_MAX, Y_MAX, MAIN_COLS


class EventMaker:
    """
    This class takes in a dataframe and preprocesses the locations to scale to pitch size
    and uses general logic to extract various features like
    stints, shots, and events (passes, interceptions, turnovers, and receptions).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Main class to make different types of events

        df: main dataframe with all tracking data
        stints: periods of uninterrupted gameplay (i.e., between shots/goals)
        shots: dataframe of shots, both successful and failed
        events: all on-ball events that are not shots (i.e., passes, receptions, turnovers, interceptions)
        """

        # scales to new dims
        self.df = self.scale_locations(df)

        # makes stints using ball location
        self.stints = self.make_stints(df)
        events = self.make_ball_events(df)
        self.shots, self.events = self.make_shots(df, events)

    def scale_locations(
        self, df: pd.DataFrame, x_dim: int = 120, y_dim: int = 80
    ) -> pd.DataFrame:
        """
        Normalizes from x \in (-1,1) and y \in (-0.42, 0.42) to 120/80
        """

        df["LOCATION_X"] = ((df["LOCATION_X"] + 1) / 2) * X_MAX
        df["LOCATION_Y"] = ((df["LOCATION_Y"] + 0.42) / 0.84) * Y_MAX
        df["VELOCITY_X"] = df["VELOCITY_X"] * x_dim
        df["VELOCITY_Y"] = df["VELOCITY_Y"] * y_dim
        return df

    def make_stints(self, df: pd.DataFrame, thresh: int = 5) -> pd.DataFrame:
        """
        Makes stints based on ball movement
        """

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
        max_dist_change["MAX_CHANGE"] = distances[
            list(distances.columns)[2:]
        ].values.max(axis=1)

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
            df[df["PLAYER_ID"] == -1].sort_values(by=["TIMESTEP"], ascending=True),
            big_changes.sort_values(by=["TIMESTEP"], ascending=True)[
                ["GAME_ID", "TIMESTEP", "STINT_NUMBER"]
            ],
            on="TIMESTEP",
            by=["GAME_ID"],
            direction="forward",
            suffixes=("", "_stint"),
        )

        return df[
            ["GAME_ID", "TIMESTEP", "STINT_NUMBER", "TEAM_0_SCORE", "TEAM_1_SCORE"]
        ]

    def make_ball_events(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def make_shots(
        self, df: pd.DataFrame, events: pd.DataFrame, post_y: float = 0.044
    ) -> pd.DataFrame:
        """
        Makes shots. Post y-loc is given in GRF documentation
        """

        possession = df[
            (df["ON_BALL"]) & (df["LOCATION_X"] > 0) & (df["LOCATION_X"] < X_MAX)
        ]

        out_of_play = df[
            (
                (df["PLAYER_ID"] == -1)
                & ((df["LOCATION_X"] > (X_MAX - 1)) | (df["LOCATION_X"] < 1))
                & (~df["ON_BALL"])
            )
            | ((df["PLAYER_ROLE"] == "goalkeeper") & (df["ON_BALL"]))
        ]

        # this gets goals
        goals = df[df["PLAYER_ID"] == -1].drop_duplicates(
            subset=["GAME_ID", "TEAM_0_SCORE", "TEAM_1_SCORE"], keep="first"
        )

        possession = pd.merge_asof(
            possession.sort_values(by=["TIMESTEP"], ascending=True),
            out_of_play.sort_values(by=["TIMESTEP"], ascending=True),
            on="TIMESTEP",
            by=["GAME_ID"],
            direction="forward",
            suffixes=("", "_shot"),
        )

        possession["NEXT_LOCATION_X_shot"] = possession["LOCATION_X_shot"].shift(-1)
        possession["NEXT_LOCATION_Y_shot"] = possession["LOCATION_Y_shot"].shift(-1)
        possession["NEXT_LOCATION_Z_shot"] = possession["LOCATION_Z_shot"].shift(-1)

        possession = possession.loc[
            (possession["LOCATION_X_shot"] != possession["NEXT_LOCATION_X_shot"])
            & (
                (
                    (possession["PLAYER_ROLE"] != "goalkeeper")
                    & (~possession["ON_BALL_shot"].astype(bool))
                )
                | (
                    (possession["PLAYER_ROLE"] == "goalkeeper")
                    & (possession["ON_BALL_shot"].astype(bool))
                )
            )
        ]

        possession = possession.drop_duplicates(
            subset=[
                "TEAM_0_SCORE_shot",
                "TEAM_1_SCORE_shot",
                "LOCATION_X_shot",
                "LOCATION_Y_shot",
                "LOCATION_Z_shot",
            ],
            keep="last",
        )

        # remove goalie possessions where previous player is on same team
        possession = possession[
            ~(
                (possession["PLAYER_ROLE_shot"] == "goalkeeper")
                & (possession["TEAM_ID"] == possession["TEAM_ID_shot"])
            )
        ]

        # remove "shots" where team is not directing it at opponent goal line
        possession = possession[
            ((possession["TEAM_ID"] == 1) & (possession["LOCATION_X_shot"] < 1))
            | (
                (possession["TEAM_ID"] == 0)
                & (possession["LOCATION_X_shot"] > (X_MAX - 1))
            )
        ]

        possession["EVENT_NAME"] = "shot"

        # remove shots that may be labeled as passes/givaways
        events = events[
            ~(
                (events["GAME_ID"].isin(list(possession["GAME_ID"])))
                & events["TIMESTEP"].isin(list(possession["TIMESTEP"]))
            )
        ]

        # add event ids as column/combination of game and timestep
        events["EVENT_ID"] = (
            events["GAME_ID"].astype(str) + "_" + events["TIMESTEP"].astype(str)
        )
        stints = self.stints.drop_duplicates(
            subset=["GAME_ID", "STINT_NUMBER"], keep="first"
        )
        stints["EVENT_ID"] = (
            stints["GAME_ID"].astype(str) + "_" + stints["TIMESTEP"].astype(str)
        )
        goals["EVENT_ID"] = (
            goals["GAME_ID"].astype(str) + "_" + goals["TIMESTEP"].astype(str)
        )

        # join stints and events to get next scores
        stints_events = pd.concat(
            [
                events[
                    ["GAME_ID", "TIMESTEP", "TEAM_0_SCORE", "TEAM_1_SCORE", "EVENT_ID"]
                ],
                stints[
                    ["GAME_ID", "TIMESTEP", "TEAM_0_SCORE", "TEAM_1_SCORE", "EVENT_ID"]
                ],
                goals[
                    ["GAME_ID", "TIMESTEP", "TEAM_0_SCORE", "TEAM_1_SCORE", "EVENT_ID"]
                ],
            ]
        )

        possession = pd.merge_asof(
            possession.sort_values(by=["TIMESTEP"], ascending=True),
            stints_events.sort_values(by=["TIMESTEP"], ascending=True),
            on="TIMESTEP",
            by=["GAME_ID"],
            direction="forward",
            suffixes=("", "_nextevent"),
        )

        possession["IS_GOAL"] = (
            possession["TEAM_0_SCORE"] != possession["TEAM_0_SCORE_nextevent"]
        ) | (possession["TEAM_1_SCORE"] != possession["TEAM_1_SCORE_nextevent"])

        # if shot is last thing in game, check if its within the posts
        scaled_post_high = ((post_y + 0.42) / 0.84) * 80
        scaled_post_low = ((-post_y + 0.42) / 0.84) * 80
        possession.loc[
            # df["TEAM_0_SCORE_nextevent"].isna(),
            (
                (possession["TEAM_0_SCORE_nextevent"].isna())
                | possession["TEAM_1_SCORE_nextevent"].isna()
            ),
            "IS_GOAL",
        ] = (possession["LOCATION_Y_shot"] >= scaled_post_low) & (
            possession["LOCATION_Y_shot"] <= scaled_post_high
        )

        # rename on-net ball location
        possession = possession.rename(
            columns={
                "NEXT_LOCATION_X_shot": "LOCATION_X_ON_NET",
                "NEXT_LOCATION_Y_shot": "LOCATION_Y_ON_NET",
                "NEXT_LOCATION_Z_shot": "LOCATION_Z_ON_NET",
            }
        )

        events["IS_GOAL"] = False
        return (
            possession[
                MAIN_COLS
                + [
                    "LOCATION_X_ON_NET",
                    "LOCATION_Y_ON_NET",
                    "LOCATION_Z_ON_NET",
                    "EVENT_NAME",
                    "IS_GOAL",
                ]
            ],
            events,
        )
