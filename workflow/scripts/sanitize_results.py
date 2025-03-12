"""Sanitizes result file name."""

import pandas as pd
from constants import VALID_RESULTS
from sanitize_params import sanitize_component_name

def strip_whitespace(results: pd.DataFrame) -> pd.DataFrame:
    """Strips any leading/trailing whitespace from naming columns."""
    
    df = results.copy()
    df["name"] = df.name.str.strip()
    df["component"] = df.component.str.strip()
    df["carriers"] = df.carriers.str.strip()
    df["variable"] = df.variable.str.strip()
    df["unit"] = df.unit.str.strip()
    return df

def is_valid_variables(results: pd.DataFrame) -> bool:
    """Confirm variables are valid.

    Assumes component names are valid.
    """

    def _check_attribute(c: str, var: str) -> None:
        valid = var in VALID_RESULTS[c]
        if not valid:
            raise ValueError(f"Attribute of {var} for component {c} is not valid")

    df = results.copy()
    df.apply(
        lambda row: _check_attribute(row["component"], row["attribute"]), axis=1
    )
    return True

if __name__ == "__main__":

    if "snakemake" in globals():
        in_results = snakemake.params.results
        out_results = snakemake.output.results
    else:
        in_results = "config/results.csv"
        out_results = "results/Test/results.csv"
    
    df = pd.read_csv(in_results)
    
    df = sanitize_component_name(df)
    df = strip_whitespace(df)
    assert is_valid_variables(df)
    
    df.to_csv(df, index=False)