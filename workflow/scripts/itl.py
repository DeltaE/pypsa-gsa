"""Processes REEDs Interface Transmission Limits"""

import pandas as pd
import pypsa


def process_itls(n: pypsa.Network, itl: pd.DataFrame) -> pd.DataFrame:
    
    df = itl.copy()
    
    reeds_zones = n.buses.reeds_zone.unique().to_list()
    
    df = df[
        (df.r.isin(reeds_zones)) & (df.rr.isin(reeds_zones))
    ]
    
    return df
        
if __name__ == "__main__":

    if "snakemake" in globals():
        network = snakemake.inputs.network
        reeds_itls = snakemake.inputs.itls
        csv = snakemake.outputs.csv
    else:
        network = ""
        reeds_rps = ""
        reeds_ces = ""
        csv = ""

    n = pypsa.Network(network)

    itl = pd.read_csv(reeds_itls)
    itl = process_itls(n, itl)
    
    itl.to_csv(csv)

