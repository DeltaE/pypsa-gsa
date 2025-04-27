"""Retrieves National Emissions from the EIA."""

from eia import Emissions
import pandas as pd
from constants import STATE_2_CODE

import logging
logger = logging.getLogger(__name__)

def retrieve_emissions(year: int, api: str) -> pd.DataFrame:
    """Gets all state level emissions."""
    df = Emissions("total", year, api).get_data(pivot=False)
    return format_emissions(df) 

def format_emissions(eia: pd.DataFrame) -> pd.DataFrame:
    """Formates EIA emissions to match PyPSA-USA schema."""
    df = eia.copy()
    df = df.reset_index(names="year")
    df["state"] = df.state.map(STATE_2_CODE)
    df["co2_limit_mmt"] = df.value
    df["sector"] = "all"
    df = df.dropna(subset="state").reset_index(drop=True)
    return df[["year", "state", "sector", "co2_limit_mmt"]]

if __name__ == "__main__":
    if "snakemake" in globals():
        api = snakemake.params.api
        co2_2005_f = snakemake.output.co2_2005
        co2_2030_f = snakemake.output.co2_2030
    else:
        api = ""
        co2_2005_f = "resources/emissions/co2_2005.csv"
        co2_2030_f = "resources/emissions/co2_2005_50pct.csv"

    co2_2005 = retrieve_emissions(2005, api)
    co2_2005.to_csv(co2_2005_f, index=False)
    
    co2_2030 = co2_2005.copy()
    co2_2030["co2_limit_mmt"] = co2_2005.co2_limit_mmt.mul(0.50)
    co2_2030.to_csv(co2_2030_f, index=False)