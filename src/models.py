import pandas as pd

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
