"""Filters constraint files to be applied to a network."""

import pandas as pd
import pypsa
from constants import STATES_TO_EXCLUDE

import logging
logger = logging.getLogger(__name__)

def get_state_memberships(n: pypsa.Network) -> dict[str, str]:

    return (
        n.buses.groupby("reeds_state")["reeds_zone"]
        .apply(lambda x: ", ".join(x))
        .to_dict()
    )
    
def get_states(n: pypsa.Network) -> list[str]:
    return n.buses.reeds_state.unique().tolist()


def collapse(rps: pd.DataFrame, ces: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([rps, ces])
    df = df[df.pct > 0.0]
    df = df.set_index("name")

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
    else:
        network = "results/Testing/base.nc"
        in_ng_domestic = "resources/natural_gas/domestic.csv"
        in_ng_international = "resources/natural_gas/international.csv"
        out_ng_domestic = "results/Testing/gsa/constraints/ng_domestic.csv"
        out_ng_international = "results/Testing/gsa/constraints/ng_international.csv"

    n = pypsa.Network(network)

    ng_domestic = pd.read_csv(in_ng_domestic)
    ng_domestic = filter_ng(n, ng_domestic)
    ng_domestic.to_csv(out_ng_domestic, index=False)
    
    ng_international = pd.read_csv(in_ng_international)
    ng_international = filter_ng(n, ng_international)
    ng_international.to_csv(out_ng_international, index=False)
