from typing import Dict
import pandas as pd

from constants import X_MAX, Y_MAX


def get_plot_colors(df: pd.DataFrame) -> Dict[int, str]:
    """
    Gets plot colors for two teams
    """

    player_df = df[df["TEAM_ID"] != -1]
    colors = ["dodgerblue", "r"]
    return {
        team_id: colors[team_count]
        for team_count, team_id in enumerate(player_df["TEAM_ID"].dropna().unique())
    }


def scale_locations(
    df: pd.DataFrame, x_dim: int = X_MAX, y_dim: int = Y_MAX
) -> pd.DataFrame:
    """
    Normalizes from x \in (-1,1) and y \in (-0.42, 0.42) to 120/80
    """

    df["LOCATION_X"] = ((df["LOCATION_X"] + 1) / 2) * X_MAX
    df["LOCATION_Y"] = ((df["LOCATION_Y"] + 0.42) / 0.84) * Y_MAX
    df["VELOCITY_X"] = df["VELOCITY_X"] * x_dim
    df["VELOCITY_Y"] = df["VELOCITY_Y"] * y_dim
    return df
