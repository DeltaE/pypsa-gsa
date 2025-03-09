"""Utility Functions"""

from typing import Any
import pandas as pd


def create_salib_problem(parameters: pd.DataFrame) -> dict[str, list[Any]]:
    """Creates SALib problem from scenario configuration.

    Arguments
    ---------
    parameters: pd.DataFrame
        Dataframe describing the prameters

    Returns
    -------
    problem: dict
        SALib formatted problem dictionary

    Raises
    ------
    ValueError
        If only one variable is givin, OR
        If only one group is given
    """

    df = parameters.copy()

    problem = {}
    problem["num_vars"] = len(df)
    if problem["num_vars"] <= 1:
        raise ValueError(
            f"Must define at least two variables in problem. User defined "
            f"{problem['num_vars']} variable(s)."
        )

    df["bounds"] = df.apply(lambda row: [row.min_value, row.max_value], axis=1)

    names = df.name.to_list()
    bounds = df.bounds.to_list()
    groups = df.group.to_list()

    problem["names"] = names
    problem["bounds"] = bounds
    problem["groups"] = groups

    num_groups = len(set(groups))
    if num_groups <= 1:
        raise ValueError(
            f"Must define at least two groups in problem. User defined "
            f"{num_groups} group(s)."
        )

    return problem

def calculate_annuity(lifetime: int, dr: float | int):
    """
    Calculate the annuity factor for an asset. 
    """
    if dr > 0:
        return dr / (1.0 - 1.0 / (1.0 + dr) ** lifetime)
    else:
        return 1 / lifetime