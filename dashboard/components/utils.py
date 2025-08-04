"""Utility functions for the dashboard."""

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

DEFAULT_CONTINOUS_COLOR_SCALE = "pubu"
DEFAULT_DISCRETE_COLOR_SCALE = "Set3"
DEFAULT_PLOTLY_THEME = "plotly"
DEFAULT_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
DEFAULT_HEIGHT = 600
DEFAULT_OPACITY = 0.7

DEFAULT_Y_LABEL = {
    "cost": "Cost ($)",
    "marginal_cost": "Marginal Costs ($/MWh)",
    "emissions": "Emissions (TCO2e)",
    "new_capacity": "New Capacity (MW)",
    "total_capacity": "Total Capacity (MW)",
    "new_capacity_trn": "New Capacity (kVMT)",
    "total_capacity_trn": "Total Capacity (kVMT)",
    "generation": "Generation (MWh)",
    "generation_trn": "Generation (kVMT)",
    "other": "Other",
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


def get_iso_dropdown_options() -> list[dict[str, str]]:
    """Get the ISO dropdown options."""
    return _convert_to_dropdown_options(ISOS)

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
        if result in ["param", "iso"]:
            pass
        elif result in metadata["results"]:
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

    return options


def get_ua_results_dropdown_options(metadata: dict) -> list[dict[str, str]]:
    """Get the UA results dropdown options."""

    options = []
    for value, data in metadata["results"].items():
        label = data["label2"] if "label2" in data else data["label"]
        options.append({"label": label, "value": value})

    return options


def get_ua_param_sector_mapper(metadata: dict = None) -> list[dict[str, str]]:
    """Get the UA sectors dropdown options."""

    options = []
    for value, data in metadata["results"].items():
        options.append({"label": data["sector"], "value": value})
        if "sector2" in data:
            options.append({"label": data["sector2"], "value": value})

    return options


def get_ua_param_result_mapper(metadata: dict = None) -> list[dict[str, str]]:
    """Get the UA parameter result mapper."""
    options = []
    for value, data in metadata["results"].items():
        options.append({"label": data["result"], "value": value})
    return options


def get_cr_params_dropdown_options(
    root: str, iso: str, flatten: bool = True
) -> list[dict[str, str]]:
    """Get the Custom Result parameters dropdown options."""
    params_f = Path(root, "data", "iso", iso, "ua_params.json")
    if not params_f.exists():
        logger.error(f"No UA params for {iso}: {params_f}")
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


def get_cr_data_by_iso(root: Path, iso: str) -> pd.DataFrame:
    """Get CR data by ISO."""
    results = _get_cr_run_results(root, iso)
    samples = _get_cr_run_samples(root, iso)
    return pd.merge(results, samples, left_index=True, right_index=True)


def _get_cr_run_results(root: Path, iso: str) -> pd.DataFrame:
    """Get CR run results."""
    results_f = Path(root, "data", "iso", iso, "ua_runs.csv")
    if not results_f.exists():
        logger.error(f"No result data for {iso}: {results_f}")
        return pd.DataFrame()
    df = pd.read_csv(results_f, index_col=0)
    return df.round(2).drop(columns=["iso"])


def _get_cr_run_samples(root: Path, iso: str) -> list[str]:
    """Get CR run samples."""
    names_f = Path(root, "data", "iso", iso, "ua_params.json")
    samples_f = Path(root, "data", "iso", iso, "sample_data.csv")

    if not names_f.exists():
        logger.error(f"No Custom Result names for {iso}: {names_f}")
        return pd.DataFrame()
    with open(names_f, "r") as f:
        names = json.load(f)

    if not samples_f.exists():
        logger.error(f"No Custom Result sample for {iso}: {samples_f}")
        return pd.DataFrame()
    sample = pd.read_csv(samples_f, index_col="run").drop(columns=["iso"])

    if not all(x in sample.columns for x in names):
        missing = [x for x in names if x not in sample.columns]
        logger.error(f"Missing result from {iso}: {missing}")
        return pd.DataFrame()

    return sample[list(names)]


def get_continuous_color_scale_options() -> list[str]:
    """Get the continuous color scale options."""
    return sorted(pc.named_colorscales())


def get_discrete_color_scale_options() -> list[str]:
    """Get the discrete color scale options."""
    return sorted(
        [
            k
            for k in px.colors.qualitative.__dict__.keys()
            if not k.startswith("__") and not k.endswith("_r")
        ]
    )


def get_plotly_plotting_themes() -> list[str]:
    """Get the plotly plotting themes."""
    return list(pio.templates.keys())
