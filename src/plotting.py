"""Shared chart helpers, so every figure in the notebook is formatted the same."""

from collections.abc import Sequence

from matplotlib.axes import Axes

from src.cleaning import format_season


def set_season_xaxis(ax: Axes, seasons: Sequence[int], step: int = 2) -> None:
    """
    Label an x-axis of season start years as "YYYY/YY", showing every step-th one.

    Args:
        ax (Axes): The chart to format.
        seasons (Sequence[int]): Season start years on the x-axis, in order.
        step (int, optional): Show every step-th label so they don't overlap. Defaults to 2.

    Returns:
        None: The axis is modified in place.
    """
    tick_years = list(seasons)[::step]
    ax.set_xticks(tick_years, [format_season(year) for year in tick_years], rotation=45)
