"""Generates Emission Targets for the GSA."""

import pandas as pd
import pypsa
from constants import GSA_COLUMNS

import logging

logger = logging.getLogger(__name__)


def filter_on_model_scope(n: pypsa.Network, df: pd.DataFrame) -> pd.DataFrame:
    """Filters emissions to only include states in the model."""

    states = [x for x in n.buses.reeds_state.unique() if x]
    return df[df.state.isin(states)].copy()


if __name__ == "__main__":
    if "snakemake" in globals():
        network = snakemake.input.network
        include_co2L = snakemake.params.include_co2L
        min_value_pct = snakemake.params.min_value
        max_value_pct = snakemake.params.max_value
        co2_2005_f = snakemake.input.co2_2005
        co2_gsa_f = snakemake.output.co2_gsa
    else:
        network = "config/pypsa-usa/elec_s40_c4m_ec_lv1.0_12h_E-G.nc"
        include_co2L = True
        min_value_pct = 40
        max_value_pct = 50
        co2_2005_f = "resources/policy/co2_2005.csv"
        co2_gsa_f = "results/scenario/generated/co2L_gsa.csv"

    if include_co2L:

        assert (
            min_value_pct and max_value_pct
        ), "min_value_pct and max_value_pct must be provided for CO2L"

        # create and populate co2L constraint data
        n = pypsa.Network(network)
        co2_2005 = pd.read_csv(co2_2005_f)

        emissions = filter_on_model_scope(n, co2_2005)
        total_2005_emissions = emissions["co2_limit_mmt"].sum()

        # input values are given a percent reduction from 2005 levels
        # so swap the min/max identifier
        min_value = round(total_2005_emissions * (1 - float(max_value_pct) / 100), 5)
        max_value = round(total_2005_emissions * (1 - float(min_value_pct) / 100), 5)

        note = f"{min_value_pct} to {max_value_pct} % reduction from 2005 levels"

        df = pd.DataFrame(
            [
                [
                    "emission_limit",
                    "emission_limit",
                    "Emission Limit",
                    "store",
                    "co2",
                    "co2L",
                    "absolute",
                    "mmt",
                    min_value,
                    max_value,
                    "https://www.eia.gov/outlooks/aeo/",
                    note,
                ]
            ],
            columns=GSA_COLUMNS,
        )

    else:
        # create empty co2L constraint data
        logger.info("No CO2L constraint data generated")
        df = pd.DataFrame(columns=GSA_COLUMNS)

    df.to_csv(co2_gsa_f, index=False)
