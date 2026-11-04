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


def make_goals_model() -> Pipeline:
    """
    Build a goals model: Scale the features, then fit a Poisson regression.

    Returns:
        Pipeline: An unfitted StandardScaler + PoissonRegressor pipeline.
    """
    # alpha is a penalty that shrinks the coefficients. The default (1.0) is
    # strong and made the model worse, so we turn it off
    return make_pipeline(StandardScaler(), PoissonRegressor(alpha=0, max_iter=1000))


def poisson_probabilities(rates: np.ndarray, max_goals: int) -> np.ndarray:
    """
    Compute the chance of scoring 0, 1, ..., max_goals goals for each scoring rate.

    Args:
        rates (np.ndarray): Expected goals of a team, one value per match.
        max_goals (int): The highest number of goals to include.

    Returns:
        np.ndarray: Shape (matches, max_goals + 1). Row m, column k is the
            chance that the team scores exactly k goals in match m.
    """
    goals = np.arange(max_goals + 1)
    factorials = np.array([math.factorial(k) for k in goals])

    # The Poisson formula: P(k goals) = e^(-rate) * rate^k / k!
    rates = np.asarray(rates)[:, np.newaxis]

    return np.exp(-rates) * rates**goals / factorials


def score_matrices(
    home_rates: np.ndarray, away_rates: np.ndarray, max_goals: int
) -> np.ndarray:
    """
    Build the score probability matrix of each match.

    We assume the home and away goals are independent, so the chance of a score
    is the home team's chance times the away team's chance.

    Args:
        home_rates (np.ndarray): Expected home goals, one value per match.
        away_rates (np.ndarray): Expected away goals, one value per match.
        max_goals (int): The highest number of goals per team in the matrix.

    Returns:
        np.ndarray: Shape (matches, max_goals + 1, max_goals + 1). Entry
            [m, h, a] is the chance that match m ends h:a (home:away).
    """
    home = poisson_probabilities(home_rates, max_goals)
    away = poisson_probabilities(away_rates, max_goals)
    matrices = home[:, :, np.newaxis] * away[:, np.newaxis, :]

    # Scores above max_goals are cut off. Rescale so each matrix sums to 1
    return matrices / matrices.sum(axis=(1, 2), keepdims=True)


def outcomes_from_scores(matrices: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """
    Add up the score matrix of each match into home win, draw and away win.

    Args:
        matrices (np.ndarray): Output of score_matrices.
        index (pd.Index): The index of the matches, for the result.

    Returns:
        pd.DataFrame: Columns H, D and A, one row per match.
    """
    # Below the diagonal the home team scores more. On the diagonal it's a draw
    return pd.DataFrame(
        {
            "H": np.tril(matrices, k=-1).sum(axis=(1, 2)),
            "D": np.trace(matrices, axis1=1, axis2=2),
            "A": np.triu(matrices, k=1).sum(axis=(1, 2)),
        },
        index=index,
    )


def most_likely_scores(matrices: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """
    Find the most likely score of each match and its probability.

    Args:
        matrices (np.ndarray): Output of score_matrices.
        index (pd.Index): The index of the matches, for the result.

    Returns:
        pd.DataFrame: ScoreHome, ScoreAway and ScoreProbability, one row per
            match.
    """
    # Flatten each matrix to one row and find its largest cell. divmod turns
    # the cell's position back into (home goals, away goals)
    flat = matrices.reshape(len(matrices), -1)
    home_goals, away_goals = np.divmod(flat.argmax(axis=1), matrices.shape[2])

    return pd.DataFrame(
        {
            "ScoreHome": home_goals,
            "ScoreAway": away_goals,
            "ScoreProbability": flat.max(axis=1),
        },
        index=index,
    )
