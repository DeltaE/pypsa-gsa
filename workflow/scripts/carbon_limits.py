"""Preps carbon limits"""

import pypsa
import pandas as pd


def process_co2L(n: pypsa.Network, co2L: pd.DataFrame) -> pd.DataFrame:
    """Preps the co2 limit file"""

    states = n.buses.reeds_state.unique().to_list()
    states.extend("USA", "usa", "Usa")
    years = n.investment_periods.to_list()

    df = co2L.copy()

    df = df[(df.state.isin(states)) & (df.year.isin(years))]

    return df


if __name__ == "__main__":

    if "snakemake" in globals():
        network = snakemake.inputs.network
        co2_policy = snakemake.inputs.co2L
        csv = snakemake.outputs.csv
    else:
        network = ""
        co2_policy = ""
        csv = ""

    n = pypsa.Network(network)

    co2L = pd.read_csv(co2_policy)
    co2L = process_co2L(n, co2L)

    co2L.to_csv(csv)
