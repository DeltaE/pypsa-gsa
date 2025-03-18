"""Generates Emission Targets for the GSA."""

from eia import Emissions
import pandas as pd
import pypsa
from constants import STATE_2_CODE, GSA_COLUMNS

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

def filter_on_model_scope(n: pypsa.Network, df: pd.DataFrame) -> pd.DataFrame:
    """Filters emissions to only include states in the model."""
    
    states = [x for x in n.buses.reeds_state.unique() if x]
    return df[df.state.isin(states)].copy()

if __name__ == "__main__":
    if "snakemake" in globals():
        api = snakemake.params.api
        network = snakemake.input.network
        min_value_pct = snakemake.params.min_value
        max_value_pct = snakemake.params.max_value
        co2_2005_f = snakemake.output.co2_2005
        co2_2030_f = snakemake.output.co2_2030
        co2_gsa_f = snakemake.output.co2_gsa
    else:
        api = ""
        network = "config/pypsa-usa/elec_s40_c4m_ec_lv1.0_12h_E-G.nc"
        min_value_pct = 40
        max_value_pct = 50
        co2_2005_f = "resources/policy/co2_2005.csv"
        co2_2030_f = "resources/policy/co2_2030.csv"
        co2_gsa_f = "resources/generated/co2L_gsa.csv"

    n = pypsa.Network(network)

    co2_2005 = retrieve_emissions(2005, api)
    co2_2005.to_csv(co2_2005_f, index=False)
    
    co2_2030 = co2_2005.copy()
    co2_2030["co2_limit_mmt"] = co2_2005.co2_limit_mmt.mul(0.50)
    co2_2030.to_csv(co2_2030_f, index=False)
    
    emissions = filter_on_model_scope(n, co2_2005)
    total_2005_emissions = emissions["co2_limit_mmt"].sum()
    
    # input values are given a percent reduction from 2005 levels
    # so swap the min/max identifier
    min_value = round(total_2005_emissions * (1 - float(max_value_pct) / 100), 2)
    max_value = round(total_2005_emissions * (1 - float(min_value_pct) / 100), 2)
    
    note = f"{min_value_pct} to {max_value_pct} % reduction from 2005 levels"
    
    df = pd.DataFrame([["emission_limit", "emission_limit", "Emission Limit", "store", "co2", "co2L", "absolute", "mmt", min_value, max_value, "https://www.eia.gov/outlooks/aeo/", note]], columns=GSA_COLUMNS)
    
    df.to_csv(co2_gsa_f, index=False)