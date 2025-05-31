"""Processes Electrical import and export data."""

from pathlib import Path
import pypsa
import pandas as pd
import duckdb

# See ./dashboard/components/shared.py for same mappings.
ISO_STATES = {
    "caiso": ["CA"],
    "ercot": ["TX"],
    "isone": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "miso": ["AR", "IL", "IN", "IA", "LA", "MI", "MN", "MO", "MS", "WI"],
    "nyiso": ["NY"],
    "pjm": ["DE", "KY", "MD", "NJ", "OH", "PA", "VA", "WV"],
    "spp": ["KS", "ND", "NE", "OK", "SD"],
    "northwest": ["ID", "MT", "OR", "WA", "WY"],
    "southeast": ["AL", "FL", "GA", "NC", "SC", "TN"],
    "southwest": ["AZ", "CO", "NM", "NV", "UT"],
    "mexico": ["MX"],
    "canada": ["BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "NL", "NFI", "PEI"],
}

# manually mapped to match EIA regions to ISOs
REGION_2_ISO = {
    "California": "caiso",
    "Canada": "canada",
    "Carolinas": "southeast",
    "Central": "spp",
    "Florida": "southeast",
    "Mexico": "mexico",
    "Mid-Atlantic": "pjm",
    "Midwest": "miso",
    "New England": "isone",
    "New York": "nyiso",
    "Northwest": "northwest",
    "Southeast": "southeast",
    "Southwest": "southwest",
    "Tennessee": "southeast",
    "Texas": "ercot",
}

STATE_2_ISO = {}
for iso, states in ISO_STATES.items():
    for state in states:
        STATE_2_ISO[state] = iso


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
    df = df[df["r"].isin(zones) ^ df["rr"].isin(zones)]
    return df[["r", "rr_iso", "MW_f0", "MW_r0"]].rename(
        columns={"r": "ba", "rr_iso": "iso", "MW_f0": "ba2iso_MW", "MW_r0": "iso2ba_MW"}
    )


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

    return df[["from", "to", "month", "interchange_reported_mwh"]]


if __name__ == "__main__":
    if "snakemake" in globals():
        network = snakemake.input.network
        year = snakemake.params.year
        balancing_period = snakemake.params.balancing_period
        pudl_path = snakemake.params.pudl_path
        regions_f = snakemake.input.regions
        membership_f = snakemake.input.membership
        flowgates_f = snakemake.input.flowgates
        net_flows_f = snakemake.output.net_flows
        capacities_f = snakemake.output.capacities
    else:
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

    # write to csv
    monthly_aggregated_interchange_data.to_csv(net_flows_f, index=False)
    flowgate_data.to_csv(capacities_f, index=False)
