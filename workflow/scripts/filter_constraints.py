"""Filters constraint files to be applied to a network."""

import pandas as pd
import pypsa

from constants import CES_CARRIERS, RPS_CARRIERS, STATES_TO_EXCLUDE


def get_state_memberships(n: pypsa.Network) -> dict[str, str]:

    return (
        n.buses.groupby("reeds_state")["reeds_zone"]
        .apply(lambda x: ", ".join(x))
        .to_dict()
    )
    
def get_states(n: pypsa.Network) -> list[str]:
    return n.buses.reeds_state.unique().tolist()


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


def filter_rps_ces(n: pypsa.Network, policy: pd.DataFrame) -> pd.DataFrame:

    # todo: move some carrier/region filtering into here

    years = n.investment_periods.to_list()
    df = policy.copy()
    df = df[df.planning_horizon.isin(years)]

    return df

def filter_ng(n: pypsa.Network, ng_trade: pd.DataFrame) -> pd.DataFrame:
    
    states = get_states(n)
    df = ng_trade.copy()
    
    df[["from_state", "to_state"]] = df.state.str.split(pat="-", expand=True)
    df = df[~(df.from_state.isin(STATES_TO_EXCLUDE) | df.to_state.isin(STATES_TO_EXCLUDE))]
    
    # only want where one of the two states is in modelled scope
    df = df[df.from_state.isin(states) ^ df.to_state.isin(states)]
    
    return df.drop(columns=["from_state", "to_state"])
    
    

if __name__ == "__main__":

    if "snakemake" in globals():
        network = snakemake.input.network
        in_ng_domestic = snakemake.input.ng_domestic
        in_ng_international = snakemake.input.ng_international
        out_ng_domestic = snakemake.output.ng_domestic
        out_ng_international = snakemake.output.ng_international
        # reeds_rps = snakemake.input.rps
        # reeds_ces = snakemake.input.ces
        # csv = snakemake.output.csv
    else:
        network = "results/Testing/base.nc"
        in_ng_domestic = "resources/natural_gas/domestic.csv"
        in_ng_international = "resources/natural_gas/international.csv"
        out_ng_domestic = "results/Testing/constraints/ng_domestic.csv"
        out_ng_international = "results/Testing/constraints/ng_international.csv"
        # reeds_rps = "resources/reeds/rps_fraction.csv"
        # reeds_ces = "resources/reeds/ces_fraction.csv"
        # csv = ""

    n = pypsa.Network(network)

    ng_domestic = pd.read_csv(in_ng_domestic)
    ng_domestic = filter_ng(n, ng_domestic)
    ng_domestic.to_csv(out_ng_domestic, index=False)
    
    ng_international = pd.read_csv(in_ng_international)
    ng_international = filter_ng(n, ng_international)
    ng_international.to_csv(out_ng_international, index=False)
    

    """
    rps = pd.read_csv(reeds_rps)
    rps = process_rps(n, rps)

    ces = pd.read_csv(reeds_ces)
    ces = process_ces(n, ces)

    rps_ces = collapse(rps, ces)
    rps_ces = filter_rps_ces(n, renewable_policy)

    rps_ces.to_csv(csv, index=True)
    """
    
    