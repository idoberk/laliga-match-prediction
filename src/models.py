import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# Bet365 odds column for each result
ODDS_COLUMNS = {"H": "OddHome", "D": "OddDraw", "A": "OddAway"}


def bookmaker_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the Bet365 odds of each match into margin-free probabilities.

    Args:
        matches (pd.DataFrame): Matches with the ODDS_COLUMNS (decimal odds).

    Returns:
        pd.DataFrame: Columns H, D and A with probabilities that sum to 1 per
            match, on the same index as matches.
    """
    # Decimal odds of 2.0 imply a 1 / 2.0 = 50% chance
    implied = 1 / matches[list(ODDS_COLUMNS.values())]
    implied.columns = list(ODDS_COLUMNS)

    # The implied chances sum to slightly more than 100%: the excess is the bookmaker's margin.
    # Dividing by the sum removes it
    return implied.div(implied.sum(axis=1), axis=0)


def make_outcome_model() -> Pipeline:
    """
    Build the outcome model: Scale the features, then fit a logistic regression.

    Returns:
        Pipeline: An unfitted StandardScaler + LogisticRegression pipeline.
            With three classes (H, D, A), the logistic regression is multinomial.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def predict_outcomes(
    model: Pipeline, matches: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """
    Predict the home win, draw and away win probabilities of each match.

    Args:
        model (Pipeline): A fitted outcome model.
        matches (pd.DataFrame): Matches with the feature columns.
        features (list[str]): The columns the model was fitted on.

    Returns:
        pd.DataFrame: Columns H, D and A, one row per match (same index).
    """
    probabilities = model.predict_proba(matches[features])

    # model.classes_ holds the column order of predict_proba (A, D, H).
    # We reorder the columns to H, D, A
    return pd.DataFrame(probabilities, columns=model.classes_, index=matches.index)[
        ["H", "D", "A"]
    ]
