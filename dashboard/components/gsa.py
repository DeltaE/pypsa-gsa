"""Component to display GSA data."""

from typing import Any
from dash import dcc, html
from pathlib import Path
from .utils import (
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
    _unflatten_dropdown_options
)
from . import ids as ids
import dash_bootstrap_components as dbc

import pandas as pd
import plotly
import plotly.express as px

import logging

logger = logging.getLogger(__name__)

root = Path(__file__).parent.parent

GSA_PARM_OPTIONS = get_gsa_params_dropdown_options(root)
GSA_RESULT_OPTIONS = get_gsa_results_dropdown_options(root)

def _default_gsa_params_value() -> list[str]:
    """Default value for the GSA parameters dropdown."""
    defaults = [x["value"] for x in GSA_PARM_OPTIONS if x["value"].endswith(("_electrical_demand", "_veh_lgt_demand"))]
    if not defaults:
        return GSA_PARM_OPTIONS[0]["value"]
    return defaults

def _default_gsa_results_value() -> list[str]:
    """Default value for the GSA results dropdown."""
    defaults = [x["value"] for x in GSA_RESULT_OPTIONS if any(y in x["value"] for y in ["_energy_", "_carbon"])]
    if not defaults:
        return GSA_RESULT_OPTIONS[0]["value"]
    return defaults

def gsa_options_block() -> html.Div:
    """GSA options block component."""
    return html.Div(
        [
            gsa_filtering_rb(),
            dbc.Collapse(
                [
                    gsa_params_dropdown(),
                ],
                id=ids.GSA_PARAMS_RESULTS_COLLAPSE,
                is_open=True,
            ),
            dbc.Collapse(
                [
                    gsa_params_slider(),
                ],
                id=ids.GSA_PARAMS_SLIDER_COLLAPSE,
                is_open=True,
            ),
            gsa_results_dropdown(),
        ],
    )

def gsa_params_slider() -> html.Div:
    """GSA slider component."""
    max_num = len(GSA_PARM_OPTIONS)
    marks = {
        0: "0",
        max_num//2: str(max_num//2),
        max_num: str(max_num),
    }
    
    return html.Div(
        [
            html.H6("Select top number of parameters:"),
            dcc.Slider(
                id=ids.GSA_PARAMS_SLIDER,
                min=0,
                max=max_num,
                value=5,
                step=1,
                included=False,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ],
    )

def gsa_filtering_rb() -> html.Div:
    """GSA filtering radio buttons component."""
    return html.Div(
        [
            html.H6(
                "Select parameters by:",
                className="card-title",
            ),
            dcc.RadioItems(
                id=ids.GSA_PARAM_SELECTION_RB,
                # add margin to the left of the label text
                options=[
                    {"label": html.Span("Name", className="ms-2"), "value": "name"},
                    {"label": html.Span("Rank", className="ms-2"), "value": "rank"}
                ],
                value="name",
                inline=True,
                className="me-3",
                labelStyle={"marginRight": "20px"},
            ),
        ],
        className="dropdown-container",
    )
    
def gsa_params_dropdown() -> html.Div:
    """GSA parameters dropdown component."""
    return html.Div(
        [
            html.H6("Select Parameter(s)"),
            dcc.Dropdown(
                id=ids.GSA_PARAM_DROPDOWN,
                options=GSA_PARM_OPTIONS,
                value=_default_gsa_params_value(),
                multi=True
            ),
            html.Div(
                [
                    dbc.Button(
                        "Select All",
                        id=ids.GSA_PARAM_SELECT_ALL,
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                    dbc.Button(
                        "Remove All",
                        id=ids.GSA_PARAM_REMOVE_ALL,
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                ],
            ),
        ],
        className="dropdown-container",
    )

def gsa_results_dropdown() -> html.Div:
    """GSA results dropdown component."""
    return html.Div(
        [
            html.H6("Select Result(s)"),
            dcc.Dropdown(
                id=ids.GSA_RESULTS_DROPDOWN,
                options=GSA_RESULT_OPTIONS,
                value=_default_gsa_results_value(),
                multi=True
            ),
            html.Div(
                [
                  dbc.Button(
                        "Select All",
                        id=ids.GSA_RESULTS_SELECT_ALL,
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                    dbc.Button(
                        "Remove All",
                        id=ids.GSA_RESULTS_REMOVE_ALL,
                        size="sm",
                        color="secondary",
                        outline=True,
                    ),
                ]
            )
        ],
        className="dropdown-container",
    )

def get_top_n_params(raw: pd.DataFrame, num_params: int, results: list[str]) -> list[str]:
    """Get the top n most impactful parameters."""
    
    df = raw.copy()[results]
    
    if df.empty:
        logger.debug("No top_n parameters found")
        return []
    
    top_n = []
    for col in df.columns:
        top_n.extend(df[col].sort_values(ascending=False).index[:num_params].to_list())
    return sorted(list(set(top_n)))
    
def _get_heatmap_height(num_params: int) -> int:
    """Get the height of the heatmap based on the number of parameters."""
    height = num_params * 20
    if height < 800:
        return 800
    return height
    
def get_gsa_heatmap(data: dict[str, Any], nice_names: bool = True, **kwargs: Any) -> plotly.graph_objects.Figure:
    """GSA heatmap component."""
    
    if not data:
        logger.debug("No heatmap data found")
        return px.imshow(
            pd.DataFrame(),
            color_continuous_scale="Bluered",
            color_continuous_midpoint=0,
            zmin=0,
            zmax=1,
            aspect="auto",
        )
    
    df = pd.DataFrame(data).set_index("param")
    logger.debug(f"Heatmap data shape: {df.shape}")
    
    if nice_names:
        logger.debug("Applying nice names to heatmap")
        gsa_params = _unflatten_dropdown_options(GSA_PARM_OPTIONS)
        gsa_results = _unflatten_dropdown_options(GSA_RESULT_OPTIONS)
        df = df.rename(columns=gsa_results).rename(index=gsa_params)
    
    color_scale = kwargs.get("color_scale", "PuBu")
    logger.debug(f"Color scale: {color_scale}")
    
    fig = px.imshow(
        df,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=0,
        zmin=0, 
        zmax=1,
        aspect="auto", 
        labels=dict(x="Parameters", y="Results", color="Scaled Elementary Effect"),
    )
    
    fig.update_layout(
        title="Global Sensitivity Analysis",
        xaxis_title="",
        yaxis_title="",
        height=_get_heatmap_height(len(df.index)),
        xaxis={'side': 'bottom'}
    )
    fig.update_xaxes(tickangle=45)
    
    return fig

    