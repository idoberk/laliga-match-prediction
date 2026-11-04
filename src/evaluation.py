import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.models import make_outcome_model, predict_outcomes


def score_probabilities(
    results: pd.Series, probabilities: pd.DataFrame
) -> dict[str, float]:
    """
    Score predicted outcome probabilities against the actual results.

    Args:
        results (pd.Series): Actual results ("H" / "D" / "A").
        probabilities (pd.DataFrame): One column per result with its predicted
            probability, one row per match (same order as results).

    Returns:
        dict[str, float]: Log loss and Brier score (lower is better), and the
            accuracy of the most likely result (higher is better).
    """
    # sklearn assumes the probability columns are in alphabetical label order (A, D, H).
    # Any other order silently scores the wrong columns
    labels = sorted(probabilities.columns)
    probabilities = probabilities[labels]

    return {
        "log_loss": log_loss(results, probabilities, labels=labels),
        "brier": brier_score_loss(results, probabilities, labels=labels),
        "accuracy": accuracy_score(results, probabilities.idxmax(axis=1)),
    }


def forward_chaining_log_loss(
    matches: pd.DataFrame, features: list[str], validation_seasons: list[int]
) -> pd.Series:
    """
    Score a feature set with season based forward chaining validation.

    For each validation season, the outcome model is trained on all earlier
    seasons and scored on that season.
    This copies real use: Learn from the past, then predict the next season.

    Args:
        matches (pd.DataFrame): Training matches with SeasonStart, FTResult and
            the feature columns.
        features (list[str]): The feature columns to use.
        validation_seasons (list[int]): SeasonStart years to validate on.

    Returns:
        pd.Series: Log loss per validation season.
    """
    log_losses = {}
    for season in validation_seasons:
        earlier = matches[matches["SeasonStart"] < season]
        validation = matches[matches["SeasonStart"] == season]

        model = make_outcome_model().fit(earlier[features], earlier["FTResult"])
        probabilities = predict_outcomes(model, validation, features)
        scores = score_probabilities(validation["FTResult"], probabilities)
        log_losses[season] = scores["log_loss"]

    return pd.Series(log_losses, name="log_loss")
