import pandas as pd


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count and percentage of missing values per column, most-missing first.

    Args:
        df (pd.DataFrame): Any DataFrame to profile.

    Returns:
        pd.DataFrame: One row per column that has at least one missing value,
            with columns missing_count and missing_pct (0 - 100, rounded to 1 decimal),
            sorted most-missing first. Empty if nothing is missing.
    """
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(1),
        }
    )

    return summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def add_season_columns(
    df: pd.DataFrame,
    date_col: str = "MatchDate",
    cutoff_month: int = 8,
    cutoff_day: int = 10,
) -> pd.DataFrame:
    """
    Add SeasonStart (e.g 2025) and Season (e.g "2025/26) columns from a date.

    A match on or after the cutoff date belongs to the season starting that calender year.
    Anything earlier belongs to the season that started the year before.
    The 10 August default fits every La Liga / Segunda match: The latest season end is
    7 Aug 2020 and the earliest start is 11 Aug.

    Args:
        df (pd.DataFrame): Matches with a datetime column.
        date_col (str, optional): Name of the datetime column. Defaults to "MatchDate".
        cutoff_month (int, optional): Month a new season starts. Defaults to 8.
        cutoff_day (int, optional): Day of cutoff_month a new season starts. Defaults to 10.

    Returns:
        pd.DataFrame: A copy of df with SeasonStart (int) and Season (str) added.
    """
    dates = df[date_col]
    on_or_after_cutoff = (dates.dt.month > cutoff_month) | (
        (dates.dt.month == cutoff_month) & (dates.dt.day >= cutoff_day)
    )

    # Before the cutoff we subtract 1 from the year (True / False becomes 1 / 0)
    season_start = dates.dt.year - (~on_or_after_cutoff).astype(int)

    # 2025 becomes "26": Last two digits of the end year, zero-padded (2008 becomes "09")
    season_end_short = ((season_start + 1) % 100).astype(str).str.zfill(2)

    return df.assign(
        SeasonStart=season_start,
        Season=season_start.astype(str) + "/" + season_end_short,
    )


def missing_pct_by_group(
    df: pd.DataFrame, group_col: str, columns: list[str]
) -> pd.DataFrame:
    """
    Percentage of missing values per group (e.g per season) for chosen columns.

    Args:
        df (pd.DataFrame): Data to profile.
        group_col (str): Column to group by, e.g "Season".
        columns (list[str]): Columns whose missingness to measure.

    Returns:
        pd.DataFrame: One row per group and one column per entry in columns, holding the
            missing percentage (0 - 100, rounded to 1 decimal).
    """
    return (df[columns].isna().groupby(df[group_col]).mean() * 100).round(1)


def format_season(season_start: int) -> str:
    """
    Turn a season's start year into its label, e.g 2025 -> "2025/26".

    Args:
        season_start (int): Calendar year the season starts in.

    Returns:
        str: Season label in "YYYY/YY" format.
    """
    return f"{season_start}/{(season_start + 1) % 100:02d}"


def build_team_season_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build final league tables (one row per team per division-season) from results.

    For EDA and reporting only, never as model features. These are end-of-season totals, so they include matches played after any given date in the season.

    Teams are ranked by points, then goal difference, then goals scored. La Liga's official tiebreak is head-to-head, but no season in the data has two teams level on points at the top, so champions are unaffected.

    Args:
        matches (pd.DataFrame): Cleaned matches with Division, SeasonStart, Season, HomeTeam, AwayTeam, FTHome and FTAway.

    Returns:
        pd.DataFrame: One row per (Division, Season, Team) with Played, Wins, Draws, Losses, GoalsFor, GoalsAgainst, GoalDiff, Points, PointsPerMatch and Rank, sorted by division, season and rank.
    """
    context_columns = ["Division", "SeasonStart", "Season"]
    long_columns = context_columns + ["Team", "GoalsFor", "GoalsAgainst"]

    # Each match seen from both sides. The home team's goals are "for" in the home view and "against" in the away view. set_axis renames by position
    home_view = matches[context_columns + ["HomeTeam", "FTHome", "FTAway"]].set_axis(
        long_columns, axis=1
    )

    away_view = matches[context_columns + ["AwayTeam", "FTAway", "FTHome"]].set_axis(
        long_columns, axis=1
    )

    team_matches = pd.concat([home_view, away_view], ignore_index=True)

    goal_margin = team_matches["GoalsFor"] - team_matches["GoalsAgainst"]
    team_matches = team_matches.assign(
        Win=goal_margin > 0, Draw=goal_margin == 0, Loss=goal_margin < 0
    )

    # Summing True / False counts the Trues: Wins = numbers of matches won
    table = (
        team_matches.groupby(context_columns + ["Team"])
        .agg(
            Played=("Win", "size"),
            Wins=("Win", "sum"),
            Draws=("Draw", "sum"),
            Losses=("Loss", "sum"),
            GoalsFor=("GoalsFor", "sum"),
            GoalsAgainst=("GoalsAgainst", "sum"),
        )
        .reset_index()
    )

    table = table.assign(
        GoalDiff=table["GoalsFor"] - table["GoalsAgainst"],
        Points=3 * table["Wins"] + table["Draws"],
    )

    # Points per match, not raw points: 2004/05 is incomplete (27 matches per team)
    table["PointsPerMatch"] = table["Points"] / table["Played"]

    # After sorting, the position within each division-season is the rank
    table = table.sort_values(
        ["Division", "SeasonStart", "Points", "GoalDiff", "GoalsFor"],
        ascending=[True, True, False, False, False],
    )

    table["Rank"] = table.groupby(["Division", "SeasonStart"]).cumcount() + 1

    return table.reset_index(drop=True)
