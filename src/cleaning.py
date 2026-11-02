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
