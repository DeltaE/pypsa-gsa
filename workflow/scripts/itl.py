"""Processes REEDs Interface Transmission Limits"""

import pandas as pd
import pypsa


def process_itls(n: pypsa.Network, itl: pd.DataFrame) -> pd.DataFrame:

    df = itl.copy()

    reeds_zones = n.buses.reeds_zone.unique().tolist()

    df = df[(df.r.isin(reeds_zones)) & (df.rr.isin(reeds_zones))]

    return df


if __name__ == "__main__":

    if "snakemake" in globals():
        network = snakemake.input.network
        reeds_itls = snakemake.input.itl
        csv = snakemake.output.csv
    else:
        network = "resources/elec_s50_c35_ec_lv1.0_48SEG_E-G.nc"
        reeds_itls = "resources/reeds/transmission_capacity_init_AC_ba_NARIS2024.csv"
        csv = ""

    n = pypsa.Network(network)

    itl = pd.read_csv(reeds_itls)
    itl = process_itls(n, itl)

    itl.to_csv(csv, index=False)
