"""Filters params to prepare for uncertainity propagation.

This will assign the average value over the uncertainity range and remove any paramerters
that are investigated through the uncertainity charasterication.
"""

import pandas as pd


def set_average(params: pd.DataFrame) -> pd.DataFrame:
    """Gets average value over the uncertaintiy range to apply to the network."""

    df = params.copy()
    df["value"] = (df["min_value"] + df["max_value"]).div(2).round(5)
    df = df.drop(columns=["min_value", "max_value", "source", "notes"])
    return df


def remove_uncertain_params(params: pd.DataFrame, to_remove: list[str]) -> pd.DataFrame:
    """Removes parameters that will be interated over in the uncertainity analysis."""

    df = params.copy()
    
    valid_index_names = df.name.to_list()
    valid_group_names = df.group.to_list()

    valid_names = set(valid_index_names + valid_group_names)
    
    assert all(
        [x in valid_names for x in to_remove]
    ), f"Invalid parameters to remove: {set(to_remove) - set(valid_names)}"

    
    df_removed = df[~df.name.isin(to_remove)].copy()
    return df_removed[~df_removed.group.isin(to_remove)]


if __name__ == "__main__":

    if "snakemake" in globals():
        in_params = snakemake.input.parameters
        out_params = snakemake.output.parameters
        to_remove = snakemake.params.to_remove
    else:
        in_params = "results/caiso/gsa/parameters.csv"
        out_params = "results/caiso/ua/static.csv"
        to_remove = ["trn_veh_hvy_demand", "trn_veh_lgt_demand", "eff_elec_water_heater"]

    df = pd.read_csv(in_params)
    df = set_average(df)
    df = remove_uncertain_params(df, to_remove)

    df.to_csv(out_params, index=False)
