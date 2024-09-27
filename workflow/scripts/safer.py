"""Regional reserve margins"""

import pandas as pd
import pypsa


def get_nerc_memberships(n: pypsa.Network) -> dict[str, str]:

    return (
        n.buses.groupby("nerc_reg")["reeds_zone"]
        .apply(lambda x: ", ".join(x))
        .to_dict()
    )


def process_safer(n: pypsa.Network, safer: pd.DataFrame) -> pd.DataFrame:

    df = safer.copy()

    nerc_memberships = get_nerc_memberships(n)

    df["region"] = df.index.map(nerc_memberships)
    df = df.dropna(subset="region")
    df = df.drop(
        columns=["none", "ramp2025_20by50", "ramp2025_25by50", "ramp2025_30by50"]
    )

    df = df.rename(columns={"static": "prm", "t": "planning_horizon"})

    return df


def filter_safer(n: pypsa.Network, safer: pd.DataFrame) -> pd.DataFrame:

    # todo: move some carrier/region filtering into here

    years = n.investment_periods.to_list()
    df = safer.copy()
    df = df[df.planning_horizon.isin(years)]

    return df


if __name__ == "__main__":

    if "snakemake" in globals():
        network = snakemake.input.network
        reeds_safer = snakemake.input.reeds_rps
        csv = snakemake.output.csv
    else:
        network = ""
        reeds_safer = ""
        csv = ""

    n = pypsa.Network(network)

    safer = pd.read_csv(reeds_safer, index_col=0)
    safer = process_safer(n, safer)
    safer = filter_safer(n, safer)

    safer.to_csv(csv)
