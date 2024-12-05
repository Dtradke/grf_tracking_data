import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression

from constants import X_MAX, Y_MAX


def expected_goals_model(shots: pd.DataFrame) -> pd.DataFrame:
    """
    Make xG model from distance and angle to goal
    """

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

    # build basic logistic regression model
    reg = LogisticRegression(random_state=9)

    # build simple dataset
    X = shots[["DISTANCE_TO_NET", "ANGLE_TO_NET"]]
    y = shots["IS_GOAL"].values

    # fit model
    reg.fit(X, y)

    # make xG predictions (in-sample)
    preds = reg.predict_proba(X)

    # return probability of goal
    return preds[:, 1]
