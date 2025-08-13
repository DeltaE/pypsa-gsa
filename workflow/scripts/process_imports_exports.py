"""Processes Electrical import and export data."""

from pathlib import Path
import pypsa
import pandas as pd
import duckdb
from eia import FuelCosts
from constants import ISO_STATES, REGION_2_ISO


def load_yearly_interchange_data(pudl_path: str, year: int) -> pd.DataFrame:
    """Loads yearly data from PUDL."""
    df = duckdb.query(
        f"SELECT * FROM read_parquet('{pudl_path}/core_eia930__hourly_interchange.parquet') WHERE EXTRACT(YEAR FROM datetime_utc) = {year}"
    )
    df = df.to_df()
    return df.fillna(0).set_index(pd.to_datetime(df.datetime_utc))


def format_interchange_data(
    interchange_data: pd.DataFrame,
    ba_2_region: dict[str, str],
    region_2_iso: dict[str, str],
    remove_international: bool = True,
) -> pd.DataFrame:
    """Formats interchange data for ISO mappings."""
    df = interchange_data.copy()
    df["from"] = df.balancing_authority_code_eia.map(ba_2_region).map(region_2_iso)
    df["to"] = df.balancing_authority_code_adjacent_eia.map(ba_2_region).map(
        region_2_iso
    )
    if remove_international:
        df = df[
            ~(df["from"].isin(["canada", "mexico"]))
            & ~(df["to"].isin(["canada", "mexico"]))
        ]
    return df[["from", "to", "interchange_reported_mwh"]]


def _get_state_to_iso_map(iso_2_state: dict[str, list[str]]) -> dict[str, str]:
    """Returns a map of state to ISO.

    This is manually mapped to match the GSA workflow. Same mappings from the
    dashboard/components/shared.py file.
    """
    state_2_iso = {}
    for iso, states in iso_2_state.items():
        for state in states:
            state_2_iso[state] = iso
    return state_2_iso


def _get_ba_2_regions_map(eia_ba_mapper: pd.DataFrame) -> dict[str, str]:
    """Returns a map of BA to regions.

    See https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48 for raw data.
    """
    return eia_ba_mapper.set_index("Code")["Region"].to_dict()


def _get_ba_2_iso_map(memberships: pd.DataFrame) -> dict[str, str]:
    """Returns a map of BA to ISO.

    Taken from PyPSA-USA membership file.
    https://github.com/PyPSA/pypsa-usa/blob/master/workflow/repo_data/ReEDS_Constraints/membership.csv
    """
    state_2_iso = _get_state_to_iso_map(ISO_STATES)
    df = memberships.copy()
    df.loc[(df.country == "MEX"), "st"] = (
        "MX"  # collapes all mexican states into one iso
    )
    df["iso"] = df["st"].map(state_2_iso)
    return df.set_index("ba")["iso"].to_dict()


def get_zones_in_network(n: pypsa.Network) -> list[str]:
    """Returns a list of zones in the network."""
    return n.buses[n.buses.carrier == "AC"].country.unique().tolist()


def get_flowgate_data(
    flowgates: pd.DataFrame, ba_2_iso: dict[str, str]
) -> dict[str, str]:
    """Returns a map of flowgates to regions."""
    flowgates["r_iso"] = flowgates.r.map(ba_2_iso)
    flowgates["rr_iso"] = flowgates.rr.map(ba_2_iso)
    return flowgates[["r", "r_iso", "rr", "rr_iso", "MW_f0", "MW_r0"]]


def get_flowgates_in_model(flowgates: pd.DataFrame, zones: list[str]) -> dict[str, str]:
    """Returns a map of flowgates to regions."""
    df = flowgates.copy()
    # keep only flowgates that are in the zones
    df = df[df["r"].isin(zones) ^ df["rr"].isin(zones)]
    df["r0"] = df.r.where(df.r.isin(zones), df.r_iso)
    df["r1"] = df.rr.where(df.rr.isin(zones), df.rr_iso)
    df = df[["r0", "r1", "MW_f0", "MW_r0"]].rename(columns={"r0": "r", "r1": "rr"})
    # organize so r is always the ba and rr is always the iso
    mask = ~((df.r.str.startswith("p")) & (df.r.str.len() <= 4))
    df.loc[mask, ["r", "rr"]] = df.loc[mask, ["rr", "r"]].values
    df.loc[mask, ["MW_f0", "MW_r0"]] = df.loc[mask, ["MW_r0", "MW_f0"]].values
    return df.groupby(["r", "rr"]).sum().round(2).reset_index()


def get_aggregated_interchange_data(
    interchange_data: pd.DataFrame, balancing_period: str
) -> dict[str, str]:
    """Aggregates interchange data by balancing period."""
    binned = interchange_data.copy()

    if balancing_period == "month":
        binned["month"] = binned.index.month
    else:
        raise NotImplementedError(
            f"Balancing period {balancing_period} not implemented."
        )
    df = binned.groupby(["from", "to", "month"]).sum().reset_index()

    # this now contains all the inter-region flows
    # however, both directions are included, so we need to drop the duplicate rows
    # ie. the values (r1->r2 = 10) AND (r2->r1 = -10) are included
    # its not garunteed that the to/from flows are the same, so we need to keep the bigger value
    df = df[df["from"] != df["to"]]

    # Sort from/to alphabetically and adjust interchange values accordingly
    mask = df["from"] > df["to"]
    df.loc[mask, ["from", "to"]] = df.loc[mask, ["to", "from"]].values
    df.loc[mask, "interchange_reported_mwh"] *= -1

    # check absolute value for the biggest to keep
    df["abs"] = df["interchange_reported_mwh"].abs()
    df = df.sort_values(
        ["from", "to", "month", "abs"], ascending=[True, True, True, False]
    )
    df = df.drop_duplicates(subset=["from", "to", "month"], keep="first")
    df = df[["from", "to", "month", "interchange_reported_mwh"]]

    return _expand_interchange_data(df)


def _expand_interchange_data(interchange_data: pd.DataFrame) -> pd.DataFrame:
    """Expands interchange data to be all permutations.

    For example, instead of showing negative net flows, we show one of net imports and exports
    as positive values, and the otehr as zero. This makes setting up constraints easier.
    """
    df = interchange_data.copy()
    mask = df.interchange_reported_mwh < 0
    df.loc[mask, ["from", "to"]] = df.loc[mask, ["to", "from"]].values
    df.loc[mask, "interchange_reported_mwh"] = df.loc[mask, "interchange_reported_mwh"].mul(-1)

    all_isos = list(set(df["from"].unique().tolist() + df["to"].unique().tolist()))
    all_months = df.month.unique()
    index = pd.MultiIndex.from_product(
        [
            all_isos,
            all_isos,
            all_months,
        ],
        names=["from", "to", "month"],
    )
    df_template = pd.DataFrame(index=index, columns=["interchange_reported_mwh"])
    df_template = df_template[
        df_template.index.get_level_values("from")
        != df_template.index.get_level_values("to")
    ]

    df = df.set_index(["from", "to", "month"])

    expanded = df.combine_first(df_template).reset_index()
    expanded["interchange_reported_mwh"] = expanded["interchange_reported_mwh"].fillna(
        0
    )

    return expanded


def format_fuel_costs(fuel_costs: pd.DataFrame) -> pd.DataFrame:
    """Formats fuel costs for ISO mappings."""
    df = fuel_costs.copy()
    data = []

    iso_2_states = ISO_STATES.copy()
    for country in ["canada", "mexico"]:
        iso_2_states.pop(country)

    for iso, states in iso_2_states.items():
        for period in df.index.unique():
            temp = df[(df.index == period) & (df.state.isin(states))]
            value = temp.value.mean()
            data.append([period, iso, value, "usd/mwh"])
    return pd.DataFrame(data, columns=["period", "iso", "value", "units"]).set_index(
        "period"
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        api = snakemake.params.api
        network = snakemake.input.network
        year = snakemake.params.year
        balancing_period = snakemake.params.balancing_period
        pudl_path = snakemake.params.pudl_path
        regions_f = snakemake.input.regions
        membership_f = snakemake.input.membership
        flowgates_f = snakemake.input.flowgates
        net_flows_f = snakemake.output.net_flows
        capacities_f = snakemake.output.capacities
        elec_costs_f = snakemake.output.costs
    else:
        api = ""
        network = Path("results", "caiso", "base.nc")
        year = 2019
        balancing_period = "month"
        pudl_path = "s3://pudl.catalyst.coop/v2025.2.0"
        regions_f = Path("resources", "interchanges", "regions.csv")
        membership_f = Path("resources", "interchanges", "membership.csv")
        flowgates_f = Path(
            "resources",
            "interchanges",
            "transmission_capacity_init_AC_ba_NARIS2024.csv",
        )
        net_flows_f = "netflows.csv"
        capacities_f = "capacities.csv"
        elec_costs_f = "elec_costs.csv"

    assert balancing_period in ["month"]

    n = pypsa.Network(network)

    # mappers between bas and isos
    ba_2_regions = _get_ba_2_regions_map(pd.read_csv(regions_f))
    ba_2_iso = _get_ba_2_iso_map(pd.read_csv(membership_f))
    region_2_iso = REGION_2_ISO

    # all interchange data for the year
    # "core_eia930__hourly_interchange" from https://viewer.catalyst.coop/
    interchange_data = load_yearly_interchange_data(pudl_path, year)
    interchange_data = format_interchange_data(
        interchange_data, ba_2_regions, region_2_iso, True
    )

    # aggregated interchange data by balancing period
    monthly_aggregated_interchange_data = get_aggregated_interchange_data(
        interchange_data, balancing_period
    )

    # extract flowgate data
    # these will be the link capacities implemented in the network
    zones = get_zones_in_network(n)
    flowgate_data = get_flowgate_data(pd.read_csv(flowgates_f), ba_2_iso)
    flowgate_data = get_flowgates_in_model(flowgate_data, zones)

    # extract monthly fuel costs
    elec_costs = FuelCosts(
        fuel="electricity", year=year, api=api, sector="all"
    ).get_data()
    elec_costs = format_fuel_costs(elec_costs)

    # write to csv
    monthly_aggregated_interchange_data.to_csv(net_flows_f, index=False)
    flowgate_data.to_csv(capacities_f, index=False)
    elec_costs.to_csv(elec_costs_f, index=True)
