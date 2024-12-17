# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Script allowing to play the game by multiple players."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from gfootball.env import config
from gfootball.env import football_env
import pandas as pd

FLAGS = flags.FLAGS

# flags.DEFINE_string('players', 'keyboard:left_players=1',
#                     'Semicolon separated list of players, single keyboard '
#                     'player on the left by default')

flags.DEFINE_string("players", "", "Semicolon separated list of players")

# flags.DEFINE_string('players', 'bot',
#                     'Semicolon separated list of players')
flags.DEFINE_string("level", "", "Level to play")
flags.DEFINE_enum("action_set", "default", ["default", "full"], "Action set")
flags.DEFINE_bool(
    "real_time", False, "If true, environment will slow down so humans can play."
)
flags.DEFINE_bool("render", False, "Whether to do game rendering.")
flags.DEFINE_integer("game_id", 0, "Integer representing the game id to start at.")
flags.DEFINE_integer(
    "num_games", 5, "Integer representing the number of games to play."
)


def main(_):
    players = FLAGS.players.split(";") if FLAGS.players else ""
    assert not (
        any(["agent" in player for player in players])
    ), "Player type 'agent' can not be used with play_game."
    cfg_values = {
        "action_set": FLAGS.action_set,
        "dump_full_episodes": True,
        "players": players,
        "real_time": FLAGS.real_time,
    }
    if FLAGS.level:
        cfg_values["level"] = FLAGS.level
    cfg = config.Config(cfg_values)
    env = football_env.FootballEnv(cfg)
    if FLAGS.render:
        env.render()
    env.reset()

    count = 0
    game_id = FLAGS.game_id

    print(FLAGS.num_games)
    print(flags)
    exit()

    player_roles = {
        0: "goalkeeper",
        1: "centre back",
        2: "left back",
        3: "right back",
        4: "defence midfield",
        5: "central midfield",
        6: "left midfield",
        7: "right midfield",
        8: "attack midfield",
        9: "central front",
    }

    df = pd.DataFrame(
        columns=[
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
    )

    try:
        while True:

            obs, reward, done, _ = env.step([])

            ball_info = (
                [game_id, count, obs["score"][0], obs["score"][1], -1, -1, -1]
                + list(obs["ball"])
                + list(obs["ball_direction"])
                + [False]
            )
            df.loc[df.shape[0]] = ball_info

            left_team = obs["left_team"]
            for player_count in range(left_team.shape[0]):
                player_id = f"0{str(player_count).zfill(2)}"
                player_role = player_roles[obs["left_team_roles"][player_count]]
                has_ball = (
                    True
                    if obs["ball_owned_team"] == 0
                    and obs["ball_owned_player"] == player_count
                    else False
                )
                player_info = (
                    [
                        game_id,
                        count,
                        obs["score"][0],
                        obs["score"][1],
                        player_id,
                        0,
                        player_role,
                    ]
                    + list(left_team[player_count])
                    + [-1]
                    + list(obs["left_team_direction"][player_count])
                    + [-1]
                    + [has_ball]
                )
                df.loc[df.shape[0]] = player_info

            right_team = obs["right_team"]
            for player_count in range(right_team.shape[0]):
                player_id = f"1{str(player_count).zfill(2)}"
                player_role = player_roles[obs["right_team_roles"][player_count]]
                has_ball = (
                    True
                    if obs["ball_owned_team"] == 1
                    and obs["ball_owned_player"] == player_count
                    else False
                )
                player_info = (
                    [
                        game_id,
                        count,
                        obs["score"][0],
                        obs["score"][1],
                        player_id,
                        1,
                        player_role,
                    ]
                    + list(right_team[player_count])
                    + [-1]
                    + list(obs["right_team_direction"][player_count])
                    + [-1]
                    + [has_ball]
                )
                df.loc[df.shape[0]] = player_info

            count += 1

            if done:
                print(f"Game {game_id} done!")
                game_id += 1
                print(
                    f"game: {game_id}, greater than {FLAGS.num_games} + {FLAGS.game_id}"
                )
                if game_id >= (FLAGS.num_games + FLAGS.game_id):
                    save_file_name = f"results/gfootball_results_games_{FLAGS.game_id}-{str(game_id-1)}.csv"
                    logging.warning(
                        f"Games finished, saving data to {save_file_name}..."
                    )
                    df.to_csv(save_file_name, index=False)
                    exit(1)
                count = 0
                env.reset()

    except KeyboardInterrupt:
        df.to_csv("results/gfootball_results_interrupted.csv", index=False)
        logging.warning("Game stopped, writing dump...")
        env.write_dump("shutdown")
        exit(1)


if __name__ == "__main__":
    app.run(main)
