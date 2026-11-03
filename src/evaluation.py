import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


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
