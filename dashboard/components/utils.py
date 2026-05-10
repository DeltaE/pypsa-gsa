"""Utility functions for the dashboard."""

from typing import Any
import pandas as pd
from pathlib import Path
import json
import plotly.colors as pc
import plotly.express as px
import plotly.io as pio

import logging

logger = logging.getLogger(__name__)

# scenarios must follow these names as they are tied to geographic locations
ISOS = {
    "caiso": "California (CAISO)",
    "ercot": "Texas (ERCOT)",
    "isone": "New England (ISO-NE)",
    "miso": "Midcontinent (MISO)",
    "nyiso": "New York (NYISO)",
    "pjm": "PJM Interconnection (PJM)",
    "spp": "Southwest Power Pool (SPP)",
    "northwest": "Northwest",
    "southeast": "Southeast",
    "southwest": "Southwest",
}

# https://www.ferc.gov/power-sales-and-markets/rtos-and-isos
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
}

STATES = {
    "AL": "Alabama",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

DEFAULT_CONTINOUS_COLOR_SCALE = "pubu"
DEFAULT_DISCRETE_COLOR_SCALE = "Set3"
DEFAULT_PLOTLY_THEME = "plotly"
DEFAULT_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
DEFAULT_HEIGHT = 600
DEFAULT_OPACITY = 0.7

DEFAULT_2005_EMISSION_LIMIT = dict(
    line_dash="dot",
    line_color="black",
    line_width=2,
    annotation_text="2005 Emissions",
    annotation_position="top left",
    annotation_font_color="black",
)

DEFAULT_2030_EMISSION_LIMIT = dict(
    line_dash="dot",
    line_color="black",
    line_width=2,
    annotation_text="2030 Emission Target",
    annotation_position="bottom left",
    annotation_font_color="black",
)

DEFAULT_Y_LABEL = {
    "cost": "Cost ($)",
    "marginal_cost": "Marginal Costs ($/MWh)",
    "emissions": "Emissions (MMT CO2e)",
    "new_capacity": "New Capacity (MW)",
    "total_capacity": "Total Capacity (MW)",
    "new_capacity_trn": "New Capacity (kVMT)",
    "total_capacity_trn": "Total Capacity (kVMT)",
    "generation": "Generation (MWh)",
    "generation_trn": "Generation (kVMT)",
    "other": "Other",
    "demand_response": "Load (MWh)",
    "utilization": "Utilization Factor (%)",
}


def _convert_to_dropdown_options(options: dict[str, str]) -> list[dict[str, str]]:
    """Convert a dictionary to a list of dropdown options, sorted alphabetically by label."""
    options_list = [{"label": v, "value": k} for k, v in options.items()]
    return sorted(
        options_list, key=lambda x: x["label"].lower()
    )  # alphabetical label order


def _unflatten_dropdown_options(options: list[dict[str, str]]) -> dict[str, str]:
    """Unflatten a dictionary of options."""
    return {x["value"]: x["label"] for x in options}


def get_metadata(root: str) -> dict[str, str]:
    """Get the metadata."""
    with open(Path(root, "data", "locked", "metadata.json"), "r") as f:
        loaded = json.load(f)
    return loaded


def get_emissions(root: str) -> dict[str, dict[str, float]]:
    """Get the emissions."""
    with open(Path(root, "data", "locked", "emissions.json"), "r") as f:
        loaded = json.load(f)
    return {x: loaded[x] for x in loaded.keys() if x in STATES}


def get_state_dropdown_options() -> list[dict[str, str]]:
    """Get the State dropdown options."""
    return _convert_to_dropdown_options(STATES)


def get_y_label(df: str, result_type: str) -> str:
    """Get y label for UA scatter plot.

    This is SUUUUUUUPER hacky, but I dont have access the know what sector
    the data is in. And this just needs to get done! :|
    """
    results = df.result.unique()

    try:
        # transport sector
        if any([x.endswith(" EV") for x in results]):
            return DEFAULT_Y_LABEL[result_type + "_trn"]
        else:
            return DEFAULT_Y_LABEL[result_type]
    except KeyError:
        return "Value"


def get_gsa_params_dropdown_options(metadata: dict) -> list[dict[str, str]]:
    """Get the GSA parameters dropdown options."""

    options = []
    for value, data in metadata["groups"].items():
        label = data["label"] if "label" in data else value
        options.append({"label": label, "value": value})

    return options


def get_ua_params_dropdown_options(
    root: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the UA parameters dropdown options."""
    with open(Path(root, "data", "system", "ua_params.json"), "r") as f:
        loaded = json.load(f)
    if flatten:
        return _convert_to_dropdown_options(loaded)
    else:
        return loaded


def get_gsa_results_dropdown_options(
    metadata: dict, results: list[str]
) -> list[dict[str, str]]:
    """Get the GSA results dropdown options."""

    options = []

    for result in results:
        if result in ["param", "state"]:
            pass
        elif result in metadata["results"]:
            if not metadata["results"][result].get("visible", True):
                continue
            if "label2" in metadata["results"][result]:
                options.append(
                    {"label": metadata["results"][result]["label2"], "value": result}
                )
            else:
                options.append(
                    {"label": metadata["results"][result]["label"], "value": result}
                )
        else:
            options.append({"label": result, "value": result})

    return sorted(options, key=lambda x: x["label"].lower())


def get_ua_results_dropdown_options(metadata: dict) -> list[dict[str, str]]:
    """Get the UA results dropdown options."""

    options = []
    for value, data in metadata["results"].items():
        if not data.get("visible", True):
            continue
        label = (
            data["label2"]
            if (
                "label2" in data
                and not data["label2"].startswith(("Trn.", "Srvc.", "Ind.", "Pwr."))
            )
            else data["label"]
        )
        options.append({"label": label, "value": value})

    return options


def get_ua2_result_types_dropdown_options(
    metadata: dict, sector: str | None = None
) -> list[dict[str, str]]:
    """Get the UA2 result types dropdown options.

    If a sector is provided, it returns the result types available for that sector.
    Otherwise, it returns the overall result summary types.
    """
    if not sector:
        options = []
        for x, y in metadata["nice_names"]["results_summary"].items():
            options.append({"label": y, "value": x})
        return sorted(options, key=lambda x: x["label"])

    available_results = set()
    for value, data in metadata["results"].items():
        if not data.get("visible", True):
            continue
        label2_val = data.get("label2", "")
        is_service_match = sector == "service" and (
            "Residential" in label2_val or "Commercial" in label2_val
        )
        is_direct_match = data.get("sector") == sector or data.get("sector2") == sector
        if is_direct_match or is_service_match:
            available_results.add(data.get("result", ""))

    options = []
    for x, y in metadata["nice_names"]["results"].items():
        if x in available_results:
            options.append({"label": y, "value": x})

    if not options:
        options = [{"label": "", "value": ""}]
    return sorted(options, key=lambda x: x["label"])


def get_ua2_result_dropdown_options(
    metadata: dict, result_type: str, sector: str | None = None
) -> list[dict[str, str]]:
    """Get the UA result summary type dropdown options."""
    options = []
    for value, data in metadata["results"].items():
        if not data.get("visible", True):
            continue

        result_value = data.get("result", "")
        if result_value != result_type:
            continue

        if sector:
            keywords = ["Residential", "Commercial"]
            is_service_match = sector == "service" and any(
                k in data.get("label2", "") for k in keywords
            )
            is_direct_match = (
                data.get("sector") == sector or data.get("sector2") == sector
            )
            if is_direct_match or is_service_match:
                label = (
                    data.get("label2", data.get("label"))
                    if is_service_match
                    else data.get("label")
                )
                options.append({"label": label, "value": value})
        else:
            summary = data.get("summary", False)
            if summary:
                options.append({"label": data["label"], "value": value})

    if not options:
        logger.debug(f"No results found for {result_type} (sector: {sector})")
        options = [{"label": "", "value": ""}]
    return options


def get_ua_param_sector_mapper(metadata: dict = None) -> list[dict[str, str]]:
    """Get the UA sectors dropdown options."""

    options = []
    for value, data in metadata["results"].items():
        if not data.get("visible", True):
            continue
        options.append({"label": data["sector"], "value": value})
        if "sector2" in data:
            options.append({"label": data["sector2"], "value": value})

    return options


def get_ua_param_result_mapper(metadata: dict = None) -> list[dict[str, str]]:
    """Get the UA parameter result mapper."""
    options = []
    for value, data in metadata["results"].items():
        if not data.get("visible", True):
            continue
        options.append({"label": data["result"], "value": value})
    return options


def _build_sector_mapper_cache(metadata: dict) -> dict[str, list[str]]:
    """Build a {sector_label -> [result_value, ...]} lookup for fast filtering."""
    cache: dict[str, list[str]] = {}
    for value, data in metadata["results"].items():
        if not data.get("visible", True):
            continue
        for key in ("sector", "sector2"):
            label = data.get(key)
            if label:
                cache.setdefault(label, []).append(value)
    return cache


def _build_result_mapper_cache(metadata: dict) -> dict[str, list[str]]:
    """Build a {result_type_label -> [result_value, ...]} lookup for fast filtering."""
    cache: dict[str, list[str]] = {}
    for value, data in metadata["results"].items():
        if not data.get("visible", True):
            continue
        label = data.get("result", "")
        cache.setdefault(label, []).append(value)
    return cache


def get_cr_params_dropdown_options(
    root: str, state: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the Custom Result parameters dropdown options."""
    params_f = Path(root, "data", "state", state, "ua_params.json")
    if not params_f.exists():
        logger.error(f"No UA params for {state}: {params_f}")
        return {}
    with open(params_f, "r") as f:
        loaded = json.load(f)
    if flatten:
        return _convert_to_dropdown_options(loaded)
    else:
        return loaded


def get_cr_result_types_dropdown_options(
    metadata: dict, sector: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the Custom Result result types dropdown options."""
    available_results = [
        y["result"]
        for _, y in metadata["results"].items()
        if y["visible"] and y["sector"] == sector
    ]
    data = {
        x: y
        for x, y in metadata["nice_names"]["results"].items()
        if x in available_results
    }
    if flatten:
        return _convert_to_dropdown_options(data)
    else:
        return data


def get_cr_results_dropdown_options(
    metadata: dict, sector: str, result_type: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the Custom Result results dropdown options."""
    logger.info(f"CR data: Sector: {sector}, Result Type: {result_type}")
    data = {
        x: y["label"]
        for x, y in metadata["results"].items()
        if y["visible"] and y["sector"] == sector and y["result"] == result_type
    }
    if flatten:
        return _convert_to_dropdown_options(data)
    else:
        return data


def get_cr_data_by_state(root: Path, state: str) -> pd.DataFrame:
    """Get CR data by State."""
    results = _get_cr_run_results(root, state)
    samples = _get_cr_run_samples(root, state)
    return pd.merge(results, samples, left_index=True, right_index=True)


def _get_cr_run_results(root: Path, state: str) -> pd.DataFrame:
    """Get CR run results."""
    results_f = Path(root, "data", "state", state, "ua_runs.parquet")
    if not results_f.exists():
        logger.error(f"No result data for {state}: {results_f}")
        return pd.DataFrame()
    df = pd.read_parquet(results_f).reset_index()
    return df.round(2).drop(columns=["state"])


def _get_cr_run_samples(root: Path, state: str) -> list[str]:
    """Get CR run samples."""
    names_f = Path(root, "data", "state", state, "ua_params.json")
    samples_f = Path(root, "data", "state", state, "sample_data.parquet")

    if not names_f.exists():
        logger.error(f"No Custom Result names for {state}: {names_f}")
        return pd.DataFrame()
    with open(names_f, "r") as f:
        names = json.load(f)

    if not samples_f.exists():
        logger.error(f"No Custom Result sample for {state}: {samples_f}")
        return pd.DataFrame()

    # Read parquet and reset the run index
    sample = (
        pd.read_parquet(samples_f)
        .reset_index()
        .set_index("run")
        .drop(columns=["state"])
    )

    if not all(x in sample.columns for x in names):
        missing = [x for x in names if x not in sample.columns]
        # logger.error(f"Missing result from {state}: {missing}")
        return pd.DataFrame()

    sample = sample[list(names)]

    # Export link marginal costs are negative in the reference network, so
    # export values in sample_scaled.csv are also negative (ref * sample).
    # This is just a plotting fix, the underlying network has correct negative costs.
    if "ng_marginal_cost_export" in sample.columns:
        sample["ng_marginal_cost_export"] = sample["ng_marginal_cost_export"] * (-1)
    if "elec_export_price" in sample.columns:
        sample["elec_export_price"] = sample["elec_export_price"] * (-1)

    return sample


def get_continuous_color_scale_options() -> list[str]:
    """Get the continuous color scale options."""
    return sorted(pc.named_colorscales())


def get_discrete_color_scale_options() -> list[str]:
    """Get the discrete color scale options."""
    return sorted(
        [
            k
            for k in px.colors.qualitative.__dict__.keys()
            if not k.startswith("__")
            and not k.endswith("_r")
            and k not in ("swatches", "_swatches")  # errors on these. idk why
        ]
    )


def get_plotly_plotting_themes() -> list[str]:
    """Get the plotly plotting themes."""
    return list(pio.templates.keys())


def get_emission_limits(emissions: list[dict[str, Any]]) -> tuple[float, float]:
    """Reads in serialized emissions data."""
    emissions_2005 = 0
    emissions_2030 = 0
    for _, data in emissions.items():
        emissions_2005 += data["2005_mmt"]
        emissions_2030 += data["2030_mmt"]

    return emissions_2005, emissions_2030
