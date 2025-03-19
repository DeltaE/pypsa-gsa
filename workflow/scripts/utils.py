"""Utility Functions"""

from typing import Any
import pandas as pd
import pypsa


def create_salib_problem(parameters: pd.DataFrame) -> dict[str, list[Any]]:
    """Creates SALib problem from scenario configuration."""

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
    
def _get_existing_lv(n: pypsa.Network) -> float:
    """Gets exisitng line volume.
    
    Used in solve and to scale the sample"""
    ac_links_existing = n.links.carrier == "AC" if not n.links.empty else pd.Series()
    return n.links.loc[ac_links_existing, "p_nom"] @ n.links.loc[ac_links_existing, "length"]