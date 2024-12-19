"""
Credit: This model is adapted from the script provided in the LaurieOnTracking tutorial

https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Tutorial3_PitchControl.py
"""

import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import warnings

from constants import X_MAX, Y_MAX
from utils import scale_locations
from plots import plot_pitchcontrol_timestep, plot_ts_pitch

warnings.filterwarnings("ignore")


class player(object):
    """
    player class to store information and execute functions about specific player

    simple_time_to_intercept(r_final): time take for player to get to target position (r_final) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept
    """

    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self, pid, team, params):
        self.id = pid
        self.is_gk = (
            team[team["PLAYER_ID"] == pid].iloc[0]["PLAYER_ROLE"] == "goalkeeper"
        )
        self.vmax = params[
            "max_player_speed"
        ]  # player max speed in m/s. Could be individualised
        self.reaction_time = params[
            "reaction_time"
        ]  # player reaction time in 's'. Could be individualised
        self.tti_sigma = params[
            "tti_sigma"
        ]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params[
            "lambda_att"
        ]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = (
            params["lambda_gk"] if self.is_gk else params["lambda_def"]
        )  # factor of 3 ensures that anything near the GK is likely to be claimed by the GK

        # get location/velo data
        self.position = team[team["PLAYER_ID"] == self.id][
            ["LOCATION_X", "LOCATION_Y"]
        ].values
        self.velocity = team[team["PLAYER_ID"] == self.id][
            ["VELOCITY_X", "VELOCITY_Y"]
        ].values
        self.PPCF = 0.0  # initialise this for later

    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0.0  # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity * self.reaction_time
        self.time_to_intercept = (
            self.reaction_time + np.linalg.norm(r_final - r_reaction) / self.vmax
        )
        return self.time_to_intercept

    def probability_intercept_ball(self, T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1 / (
            1.0
            + np.exp(
                -np.pi / np.sqrt(3.0) / self.tti_sigma * (T - self.time_to_intercept)
            )
        )
        return f


def initialise_players(team, params):
    """
    create a list of player objects that holds their positions and velocities from the tracking data dataframe
    """
    return [player(pid, team, params) for pid in list(team["PLAYER_ID"].unique())]


def default_model_params(time_to_control_veto=3):
    """
    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
    """
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params["max_player_accel"] = (
        7.0  # maximum player acceleration m/s/s, not used in this implementation
    )
    params["max_player_speed"] = 5.0  # maximum player speed m/s
    params["reaction_time"] = (
        0.7  # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    )
    params["tti_sigma"] = (
        0.45  # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    )
    params["kappa_def"] = (
        1.0  # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    )
    params["lambda_att"] = 4.3  # ball control parameter for attacking team
    params["lambda_def"] = (
        4.3 * params["kappa_def"]
    )  # ball control parameter for defending team
    params["lambda_gk"] = (
        params["lambda_def"] * 3.0
    )  # make goal keepers must quicker to control ball (because they can catch it)
    params["average_ball_speed"] = 15.0  # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params["int_dt"] = 0.04  # integration timestep (dt)
    params["max_int_time"] = 10  # upper limit on integral time
    params["model_converge_tol"] = (
        0.01  # assume convergence when PPCF>0.99 at a given location.
    )
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start.
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params["time_to_control_att"] = (
        time_to_control_veto
        * np.log(10)
        * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        time_to_control_veto
        * np.log(10)
        * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_def"])
    )
    return params


def calculate_pitch_control_at_target(
    target_position, attacking_players, defending_players, ball_start_pos, params
):
    """
    Calculates the pitch control probability for a specific location on the pitch
    """

    # calculate ball travel time from start position to end position.
    if (
        ball_start_pos is None or np.isnan(ball_start_pos).any()
    ):  # assume that ball is already at location
        ball_travel_time = 0.0
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = (
            np.linalg.norm(target_position - ball_start_pos)
            / params["average_ball_speed"]
        )

    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin(
        [p.simple_time_to_intercept(target_position) for p in attacking_players]
    )
    tau_min_def = np.nanmin(
        [p.simple_time_to_intercept(target_position) for p in defending_players]
    )

    # check whether we actually need to solve equation 3
    if (
        tau_min_att - max(ball_travel_time, tau_min_def)
        >= params["time_to_control_def"]
    ):
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0.0, 1.0
    elif (
        tau_min_def - max(ball_travel_time, tau_min_att)
        >= params["time_to_control_att"]
    ):
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1.0, 0.0
    else:
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [
            p
            for p in attacking_players
            if p.time_to_intercept - tau_min_att < params["time_to_control_att"]
        ]
        defending_players = [
            p
            for p in defending_players
            if p.time_to_intercept - tau_min_def < params["time_to_control_def"]
        ]
        # set up integration arrays
        dT_array = np.arange(
            ball_travel_time - params["int_dt"],
            ball_travel_time + params["max_int_time"],
            params["int_dt"],
        )
        PPCFatt = np.zeros_like(dT_array)
        PPCFdef = np.zeros_like(dT_array)
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1 - ptot > params["model_converge_tol"] and i < dT_array.size:
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (
                    (1 - PPCFatt[i - 1] - PPCFdef[i - 1])
                    * player.probability_intercept_ball(T)
                    * player.lambda_att
                )
                # make sure it's greater than zero
                assert (
                    dPPCFdT >= 0
                ), "Invalid attacking player probability (calculate_pitch_control_at_target)"
                player.PPCF += (
                    dPPCFdT * params["int_dt"]
                )  # total contribution from individual player
                PPCFatt[
                    i
                ] += (
                    player.PPCF
                )  # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (
                    (1 - PPCFatt[i - 1] - PPCFdef[i - 1])
                    * player.probability_intercept_ball(T)
                    * player.lambda_def
                )
                # make sure it's greater than zero
                assert (
                    dPPCFdT >= 0
                ), "Invalid defending player probability (calculate_pitch_control_at_target)"
                player.PPCF += (
                    dPPCFdT * params["int_dt"]
                )  # total contribution from individual player
                PPCFdef[
                    i
                ] += player.PPCF  # add to sum over players in the defending team
            ptot = PPCFdef[i] + PPCFatt[i]  # total pitch control probability
            i += 1
        if i >= dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot))
        return PPCFatt[i - 1], PPCFdef[i - 1]


def generate_pitch_control_for_timestep(
    tracking_zero,
    tracking_one,
    tracking_ball,
    params,
    n_grid_cells_x=50,
):
    """
    Evaluates pitch control surface over the field based on the locations of all players at the current timestep
    """

    ball_start_pos = np.array(
        tracking_ball.iloc[0][["LOCATION_X", "LOCATION_Y"]].values, dtype=np.float16
    )

    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x * Y_MAX / X_MAX)
    dx = X_MAX / n_grid_cells_x
    dy = Y_MAX / n_grid_cells_y

    xgrid = np.arange(n_grid_cells_x) * dx + (dx / 2.0)
    ygrid = np.arange(n_grid_cells_y) * dy + (dy / 2.0)
    # initialise pitch control grids for attacking and defending teams
    ppcfa = np.zeros(shape=(len(ygrid), len(xgrid)))
    ppcfd = np.zeros(shape=(len(ygrid), len(xgrid)))

    # create player objects
    attacking_players = initialise_players(tracking_zero, params)
    defending_players = initialise_players(tracking_one, params)

    # calculate pitch pitch control model at each location on the pitch
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array([xgrid[j], ygrid[i]])
            ppcfa[i, j], ppcfd[i, j] = calculate_pitch_control_at_target(
                target_position,
                attacking_players,
                defending_players,
                ball_start_pos,
                params,
            )
    # check probabilitiy sums within convergence
    checksum = np.sum(ppcfa + ppcfd) / float(n_grid_cells_y * n_grid_cells_x)
    assert 1 - checksum < params["model_converge_tol"], "Checksum failed: %1.3f" % (
        1 - checksum
    )
    return ppcfa, xgrid, ygrid


if __name__ == "__main__":
    parser = ArgumentParser(description="pitch_control_model")
    parser.add_argument(
        "--data_path", default="data/", type=str, help="Local path to data directory"
    )
    parser.add_argument(
        "--event_path", default="events/", type=str, help="Local path to data directory"
    )
    parser.add_argument(
        "--num_frames",
        default=200,
        type=int,
        help="Number of frames to do pitch control on",
    )
    parameters = parser.parse_args()

    # collect files in path
    game = os.listdir(parameters.data_path)[0]

    # get a game from local
    df = pd.read_csv(f"{parameters.data_path}{game}")

    for game in list(df["GAME_ID"].unique()):
        game_df = df[df["GAME_ID"] == game]

        # scale locations
        game_df = scale_locations(game_df)

        # get pitch control params
        params = default_model_params()

        for timestep in list(game_df["TIMESTEP"].unique())[: parameters.num_frames]:
            ts_df = game_df[game_df["TIMESTEP"] == timestep].drop_duplicates(
                subset=["PLAYER_ID"]
            )

            # split teams
            team_zero = ts_df[ts_df["TEAM_ID"] == 0]
            team_one = ts_df[ts_df["TEAM_ID"] == 1]
            ball_df = ts_df[ts_df["TEAM_ID"] == -1]

            # calculate pitch control
            pitch_control, xgrid, ygrid = generate_pitch_control_for_timestep(
                team_zero, team_one, ball_df, params
            )

            # plot pitch control
            plot_pitchcontrol_timestep(team_zero, team_one, ball_df, pitch_control)
