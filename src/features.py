import numpy as np
import pandas as pd

# Columns that describe the match itself, identical for both teams
MATCH_CONTEXT_COLUMNS = ["Division", "SeasonStart", "Season", "MatchDate"]

# {original column: team-view column} for each side of a match.
# The home team's goals are "for" in its own view and "against" in the opponent's view
HOME_VIEW_COLUMNS = {
    "HomeTeam": "Team",
    "AwayTeam": "Opponent",
    "HomeElo": "Elo",
    "Form3Home": "Form3",
    "Form5Home": "Form5",
    "FTHome": "GoalsFor",
    "FTAway": "GoalsAgainst",
    "HomeShots": "ShotsFor",
    "AwayShots": "ShotsAgainst",
    "HomeTarget": "TargetFor",
    "AwayTarget": "TargetAgainst",
}

AWAY_VIEW_COLUMNS = {
    "AwayTeam": "Team",
    "HomeTeam": "Opponent",
    "AwayElo": "Elo",
    "Form3Away": "Form3",
    "Form5Away": "Form5",
    "FTAway": "GoalsFor",
    "FTHome": "GoalsAgainst",
    "AwayShots": "ShotsFor",
    "HomeShots": "ShotsAgainst",
    "AwayTarget": "TargetFor",
    "HomeTarget": "TargetAgainst",
}


def build_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape matches to one row per team per match (each match appears twice).

    In this "long" view all of a team's matches sit in one Team column, so per-team
    history (previous Elo, rolling form, rolling goals) becomes a groupby("Team").

    Args:
        matches (pd.DataFrame): Cleaned matches with a unique index, the MATCH_CONTEXT_COLUMNS
            and every column in HOME_VIEW_COLUMNS.

    Returns:
        pd.DataFrame: One row per (match, team) with MatchId (the index label of
            the match in matches), the context columns, the team-view columns,
            IsHome, Points (3 / 1 / 0) and Outcome ("W" / "D" / "L"), sorted by team and date.
    """
    # Keep the match's index label as a column, so per-team features can be joined back to their match later
    matches = matches.rename_axis("MatchId").reset_index()
    context_columns = ["MatchId"] + MATCH_CONTEXT_COLUMNS

    home_view = matches[context_columns + list(HOME_VIEW_COLUMNS)].rename(
        columns=HOME_VIEW_COLUMNS
    )

    away_view = matches[context_columns + list(AWAY_VIEW_COLUMNS)].rename(
        columns=AWAY_VIEW_COLUMNS
    )

    team_matches = pd.concat(
        [home_view.assign(IsHome=True), away_view.assign(IsHome=False)],
        ignore_index=True,
    )

    goal_margin = team_matches["GoalsFor"] - team_matches["GoalsAgainst"]
    is_win, is_draw = goal_margin > 0, goal_margin == 0
    team_matches["Points"] = np.select([is_win, is_draw], [3, 1], default=0)
    team_matches["Outcome"] = np.select([is_win, is_draw], ["W", "D"], default="L")

    # Per-team history must be in date order
    return team_matches.sort_values(["Team", "MatchDate"]).reset_index(drop=True)


def add_rolling_averages(
    team_matches: pd.DataFrame, columns: list[str], window: int, min_periods: int
) -> pd.DataFrame:
    """
    Add teach team's average of `columns` over its previous `window` matches.

    shift(1) runs before rolling(), so the row of match M only averages matches
    played before M. The match's own stats never enter its feature (no leakage).

    Args:
        team_matches (pd.DataFrame): Output of build_team_matches, sorted by team and date.
        columns (list[str]): Per-match stats to average, e.g "GoalsFor".
        window (int): Number of previous matches in the average.
        min_periods (int): Minimum number of previous matches with a value.
            Below it, the average is NaN rather than a one-match value.

    Returns:
        pd.DataFrame: team_matches with one new column per stat, named f"{column}Avg{window}".
            e.g "GoalsForAvg10".
    """
    rolling_averages = team_matches.groupby("Team")[columns].transform(
        lambda stats: stats.shift(1).rolling(window, min_periods=min_periods).mean()
    )

    return team_matches.join(rolling_averages.add_suffix(f"Avg{window}"))
