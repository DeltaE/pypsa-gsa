"""Processes RPS and CES policies."""

import pandas as pd

from constants import CES_CARRIERS, RPS_CARRIERS

import logging
logger = logging.getLogger(__name__)

def process_reeds_data(filepath, carriers, value_col):
    """Helper function to process RPS or CES REEDS data."""
    reeds = pd.read_csv(filepath)

    # Handle both wide and long formats
    if "rps_all" not in reeds.columns:
        reeds = reeds.melt(
            id_vars="st",
            var_name="planning_horizon",
            value_name=value_col,
        )

    # Standardize column names
    reeds = reeds.rename(
        columns={"st": "region", "t": "planning_horizon", "rps_all": "pct"},
    )
    reeds["carrier"] = [", ".join(carriers)] * len(reeds)

    # Extract and create new rows for `rps_solar` and `rps_wind`
    additional_rows = []
    for carrier_col, carrier_name in [
        ("rps_solar", "solar"),
        ("rps_wind", "onwind, offwind, offwind_floating"),
    ]:
        if carrier_col in reeds.columns:
            temp = reeds[["region", "planning_horizon", carrier_col]].copy()
            temp = temp.rename(columns={carrier_col: "pct"})
            temp["carrier"] = carrier_name
            additional_rows.append(temp)

    # Combine original data with additional rows
    if additional_rows:
        additional_rows = pd.concat(additional_rows, ignore_index=True)
        reeds = pd.concat([reeds, additional_rows], ignore_index=True)

    # Ensure the final dataframe has consistent columns
    reeds = reeds[["region", "planning_horizon", "carrier", "pct"]]
    reeds = reeds[
        reeds["pct"] > 0.0
    ]  # Remove any rows with zero or negative percentages

    return reeds


if __name__ == "__main__":

    if "snakemake" in globals():
        in_policy = snakemake.input.policy
        out_policy = snakemake.output.policy
        policy = snakemake.wildcards.policy
    else:
        in_policy = ""
        out_policy = ""
        policy = "rps"

    if policy == "rps":
        cars = RPS_CARRIERS
    elif policy == "ces":
        cars = CES_CARRIERS
    else:
        raise ValueError(policy)

    df = process_reeds_data(in_policy, cars, value_col="pct")

    df.to_csv(out_policy, index=False)
