"""Sanitizes result file name."""

import pandas as pd
import pypsa
import itertools
from constants import VALID_RESULTS
from sanitize_params import sanitize_component_name
from utils import configure_logging

import logging

logger = logging.getLogger(__name__)


def strip_whitespace(results: pd.DataFrame) -> pd.DataFrame:
    """Strips any leading/trailing whitespace from naming columns."""

    df = results.copy()
    df["name"] = df.name.str.strip()
    df["component"] = df.component.str.strip()
    df["carriers"] = df.carriers.str.strip()
    df["variable"] = df.variable.str.strip()
    df["unit"] = df.unit.str.strip()
    df["gsa_plot"] = df.gsa_plot.str.strip()
    return df


def is_valid_variables(results: pd.DataFrame) -> bool:
    """Confirm variables are valid.

    Assumes component names are valid.
    """

    def _check_variable(c: str, var: str) -> None:
        valid = var in VALID_RESULTS[c]
        if not valid:
            raise ValueError(f"Variable of {var} for component {c} is not valid")

    df = results.copy()
    df.apply(lambda row: _check_variable(row["component"], row["variable"]), axis=1)
    return True


def is_valid_carrier(n: pypsa.Network, results: pd.DataFrame) -> bool:
    """Check all defined carriers are in the network."""

    df = results.copy()

    sa_cars = df.carriers.dropna().to_list()
    sa_cars_split = []
    for car in sa_cars:
        sa_cars_split.append(car.split(";"))
    sa_cars_flat = set(list(itertools.chain(*sa_cars_split)))

    n_cars = n.carriers.index.to_list()
    n_cars.append("load")  # load shedding added during sample

    # we need to split the 'gas trade' carrier into 'gas imports' and 'gas exports'
    # ideally this would happen upstream in the pypsa-usa repo
    # but its just manually done here for the time being
    # the result calcs have been adjusted to handle this
    n_cars.append("gas imports")
    n_cars.append("gas exports")

    errors = []

    for car in sa_cars_flat:
        if car not in n_cars:
            errors.append(car)

    if errors:
        logger.error(f"{errors} are not defined in network.")
        return False
    else:
        return True


def is_unique_names(results: pd.DataFrame) -> bool:
    """Checks that all result names are unique."""

    df = results.copy()

    df = df[df.duplicated("name")]
    if not df.empty:
        duplicates = set(df.name.to_list())
        logger.error(f"Duplicate definitions of {duplicates}")
        return False
    else:
        return True


def no_nans(results: pd.DataFrame) -> bool:
    """Checks that there are no NaNs in the result plots."""
    df = results.copy()
    if df.gsa_plot.isna().any():
        nan_plots_rows = df[df.gsa_plot.isna()]
        for _, row in nan_plots_rows.iterrows():
            logger.error(
                f"NaN found in plots column for {row['name']} with {row['component']} and {row['variable']}."
            )
        return False
    else:
        return True


if __name__ == "__main__":
    if "snakemake" in globals():
        network = snakemake.input.network
        in_results = snakemake.params.results
        out_results = snakemake.output.results
        configure_logging(snakemake)
    else:
        network = "results/caiso/base.nc"
        in_results = "config/results.csv"
        out_results = "results/caiso/gsa/results.csv"

    df = pd.read_csv(in_results)

    df = sanitize_component_name(df)
    df = strip_whitespace(df)
    assert is_valid_variables(df)
    assert is_unique_names(df)
    if "plots" in df.columns:  # only needed for gsa
        assert no_nans(df)

    n = pypsa.Network(network)
    assert is_valid_carrier(n, df)

    df.to_csv(out_results, index=False)
