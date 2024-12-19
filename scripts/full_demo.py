"""
Full demo script. Once we've recorded a game, make events and run xG/pitch control on shots

To run through demo (from root):
1) Collect data from game
    - Option 1: Run collection process outlined in `/football/README.md`
    - Option 2: Save pre-recorded dataset from Google Drive

2) Train expected goals model on large dataset
    python3 scripts/expected_goals.py --data_path=[root to data dir] --event_path=[root to saved events (if saved)]

3) python3 scripts/full_demo.py --data_path=[root to data dir to predict on] --xg_model_fname=[root path to saved xg model]
"""

import os
from argparse import ArgumentParser
import pandas as pd
import warnings


from models import expected_goals_model, calculate_xg_features
from event_maker import EventMaker
from pitch_control import default_model_params, generate_pitch_control_for_timestep
from models import load_expected_goals_model
from plots import plot_pitchcontrol_timestep, make_pass_plot, create_gifs


warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = ArgumentParser(description="expected_goals_model")
    parser.add_argument(
        "--data_path",
        default="football/gfootball/results/",
        type=str,
        help="Local path to data directory",
    )
    parser.add_argument(
        "--xg_model_fname",
        default="tmp/models/xg/xg_model.pkl",
        type=str,
        help="Local path and fname to pre-trained xG model",
    )
    parameters = parser.parse_args()

    games = [
        fname for fname in os.listdir(parameters.data_path) if fname[-4:] == ".csv"
    ]

    # collect data
    df = pd.concat(
        [pd.read_csv(f"{parameters.data_path}{game_fname}") for game_fname in games]
    ).sort_values(by=["GAME_ID", "TIMESTEP"], ascending=True)

    print(f"Plotting information for {df['GAME_ID'].unique().shape[0]} games")

    # make events from game(s)
    event_maker = EventMaker(df)

    # load pre-trained xg model
    xg_model = load_expected_goals_model(parameters.xg_model_fname)

    # get xG features on shots
    X, y = calculate_xg_features(event_maker.shots)

    # make xg prediction on shots extracted from games
    event_maker.shots["XG"] = xg_model.predict_proba(X)[:, 1]

    # get pitch control params
    params = default_model_params()

    print(f"Plotting pitch control/xG for {event_maker.shots.shape[0]} shots")

    # plot xG and pitch control on shots
    for _, row in event_maker.shots[["GAME_ID", "TIMESTEP"]].iterrows():

        shot_tracking_ts = event_maker.df[
            (event_maker.df["GAME_ID"] == row["GAME_ID"])
            & (event_maker.df["TIMESTEP"] == row["TIMESTEP"])
        ]

        # split teams
        team_zero = shot_tracking_ts[shot_tracking_ts["TEAM_ID"] == 0]
        team_one = shot_tracking_ts[shot_tracking_ts["TEAM_ID"] == 1]
        ball_df = shot_tracking_ts[shot_tracking_ts["TEAM_ID"] == -1]

        # calculate pitch control
        pitch_control, xgrid, ygrid = generate_pitch_control_for_timestep(
            team_zero, team_one, ball_df, params
        )

        # pull xG on shot
        xg = event_maker.shots[
            (event_maker.shots["GAME_ID"] == row["GAME_ID"])
            & (event_maker.shots["TIMESTEP"] == row["TIMESTEP"])
        ].iloc[0]["XG"]

        title_str = (
            f"Game: {row['GAME_ID']}, TS: {row['TIMESTEP']}, xG: {round(100 * xg, 1)}%"
        )

        # plot pitch control
        plot_pitchcontrol_timestep(
            team_zero, team_one, ball_df, pitch_control, show=False, title_str=title_str
        )

    # plot passes and make into gifs
    make_pass_plot(df, event_maker.events, num_animations=10)
    create_gifs("tmp/passes")
