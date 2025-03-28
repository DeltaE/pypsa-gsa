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
    
    assert all([x in df.name for x in to_remove])
    
    return df[~df.name.isin(to_remove)]

if __name__ == "__main__":

    if "snakemake" in globals():
        in_params = snakemake.input.parameters
        out_params = snakemake.output.parameters
        to_remove = snakemake.params.to_remove
    else:
        in_params = "results/Testing/gsa/parameters.csv"
        out_params = "results/Testing/ua/static.csv"
        to_remove = []
        
    df = pd.read_csv(in_params)
    df = set_average(df)
    df = remove_uncertain_params(df, to_remove)
    
    df.to_csv(out_params, index=False)
    
    