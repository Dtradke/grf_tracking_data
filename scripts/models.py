"""
Houses the expected goals (xG) model and how to load/save it locally
"""

import os
from typing import Sequence
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.linear_model import LogisticRegression

from constants import X_MAX, Y_MAX


def calculate_xg_features(shots: pd.DataFrame) -> Sequence[np.ndarray]:
    """
    Calculates angle and distance from the net for xG model
    """
    print("Calculating features for xG...")

    # align all shots onto right goal
    shots.loc[shots["TEAM_ID"] == 1, "LOCATION_Y"] = -1 * (shots["LOCATION_Y"] - Y_MAX)
    shots.loc[shots["TEAM_ID"] == 1, "LOCATION_X"] = X_MAX - shots["LOCATION_X"]

    # distance from the goal center
    shots["GOAL_X"] = X_MAX
    shots["GOAL_Y"] = Y_MAX / 2
    shots["GOAL_X"] = shots["GOAL_X"].astype(np.float64)
    shots["GOAL_Y"] = shots["GOAL_Y"].astype(np.float64)
    shots["DISTANCE_TO_NET"] = np.sqrt(
        (shots["LOCATION_X"] - shots["GOAL_X"]) ** 2
        + (shots["LOCATION_Y"] - shots["GOAL_Y"]) ** 2
    )

    # absolute angle from straight on
    shots["ANGLE_TO_NET"] = shots.apply(
        lambda x: 90
        - math.degrees(math.atan2(X_MAX - x.LOCATION_X, (Y_MAX / 2) - x.LOCATION_Y)),
        axis=1,
    )
    shots["ANGLE_TO_NET"] = shots["ANGLE_TO_NET"].abs()

    # build xG dataset
    X = shots[["DISTANCE_TO_NET", "ANGLE_TO_NET"]]
    y = shots["IS_GOAL"].values
    return X, y


def expected_goals_model(
    shots: pd.DataFrame, model_dir: str = "tmp/models/xg/"
) -> pd.DataFrame:
    """
    Make xG model from distance and angle to goal
    """
    print("Training new xG model...")

    # build basic logistic regression model
    model = LogisticRegression(random_state=9)

    # get xG features on shots
    X, y = calculate_xg_features(shots)

    # fit model
    model.fit(X, y)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save model locally
    pkl_filename = f"{model_dir}xg_model.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Saved: {pkl_filename}")

    # make xG predictions (in-sample)
    preds = model.predict_proba(X)

    # return probability of goal
    return preds[:, 1]


def load_expected_goals_model(
    model_fname: str = "tmp/models/xg/xg_model.pkl",
) -> pd.DataFrame:
    """
    Loads a previously trained xG model for deployment
    """
    print(f"Loading previously trained xG model: {model_fname}")

    if not os.path.exists(model_fname):
        raise FileNotFoundError(
            f"Error, no xG model at: {model_fname}. Pre-train a model (expected_goals.py) or point to existing model"
        )

    with open(model_fname, "rb") as file:
        model = pickle.load(file)
    return model
