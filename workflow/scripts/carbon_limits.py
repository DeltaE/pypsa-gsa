"""Preps carbon limits"""

import pypsa
import pandas as pd


def process_co2L(n: pypsa.Network, co2L: pd.DataFrame) -> pd.DataFrame:
    """Preps the co2 limit file"""

    states = n.buses.reeds_state.unique().tolist()
    states.extend(["USA", "usa", "Usa"])
    years = n.investment_periods.to_list()

    df = co2L.copy()

    df = df[(df.state.isin(states)) & (df.year.isin(years))]

    return df


if __name__ == "__main__":

    if "snakemake" in globals():
        network = snakemake.input.network
        co2_policy = snakemake.input.co2L
        csv = snakemake.output.csv
    else:
        network = "resources/elec_s50_c35_ec_lv1.0_48SEG_E-G.nc"
        co2_policy = "resources/policy/sector_co2_limits.csv"
        csv = ""

    n = pypsa.Network(network)

    co2L = pd.read_csv(co2_policy)
    co2L = process_co2L(n, co2L)

    co2L.to_csv(csv, index=False)
