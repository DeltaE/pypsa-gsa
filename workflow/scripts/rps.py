"""Collapses RPS policy to a single file to ingest"""

import pandas as pd
import pypsa

from constants import CES_CARRIERS, RPS_CARRIERS


def get_state_memberships(n: pypsa.Network) -> dict[str, str]:

    return (
        n.buses.groupby("reeds_state")["reeds_zone"]
        .apply(lambda x: ", ".join(x))
        .to_dict()
    )


def process_rps(n: pypsa.Network, rps: pd.DataFrame) -> pd.DataFrame:

    df = rps.copy()
    state_membership = get_state_memberships(n)

    df["region"] = df["st"].map(state_membership)
    df = df.dropna(subset="region")
    df["carrier"] = [", ".join(RPS_CARRIERS)] * len(df)
    df = df.rename(
        columns={"t": "planning_horizon", "rps_all": "pct", "st": "name"},
    )
    df = df.drop(columns=["rps_solar", "rps_wind"])

    return df


def process_ces(n: pypsa.Network, ces: pd.DataFrame) -> pd.DataFrame:

    df = ces.copy()
    state_membership = get_state_memberships(n)

    df = df.melt(id_vars="st", var_name="planning_horizon", value_name="pct")
    df["region"] = df["st"].map(state_membership)
    df = df.dropna(subset="region")
    df["carrier"] = [", ".join(CES_CARRIERS)] * len(df)
    df = df.rename(columns={"st": "name"})

    return df


def collapse(rps: pd.DataFrame, ces: pd.DataFrame) -> pd.DataFrame:

    df = pd.concat([rps, ces])
    df = df[df.pct > 0.0]
    df = df.set_index("name")

    return df


def filter_policy(n: pypsa.Network, policy: pd.DataFrame) -> pd.DataFrame:

    # todo: move some carrier/region filtering into here

    years = n.investment_periods.to_list()
    df = policy.copy()
    df = df[df.planning_horizon.isin(years)]

    return df


if __name__ == "__main__":

    if "snakemake" in globals():
        network = snakemake.input.network
        reeds_rps = snakemake.input.rps
        reeds_ces = snakemake.input.ces
        csv = snakemake.output.csv
    else:
        network = "resources/elec_s50_c35_ec_lv1.0_48SEG_E-G.nc"
        reeds_rps = "resources/reeds/rps_fraction.csv"
        reeds_ces = "resources/reeds/ces_fraction.csv"
        csv = ""

    n = pypsa.Network(network)

    rps = pd.read_csv(reeds_rps)
    rps = process_rps(n, rps)

    ces = pd.read_csv(reeds_ces)
    ces = process_ces(n, ces)

    policy = collapse(rps, ces)

    policy = filter_policy(n, policy)

    policy.to_csv(csv, index=True)
