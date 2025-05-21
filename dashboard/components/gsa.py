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

def gsa_params_dropdown() -> html.Div:
    """GSA parameters dropdown component."""
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Select Parameter"),
                ],
            ),
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
            html.Div(
                [
                    html.Label("Select Result"),
                ],
            ),
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
    
def _get_heatmap_height(num_params: int) -> int:
    """Get the height of the heatmap based on the number of parameters."""
    height = num_params * 20
    if height < 800:
        return 800
    return height
    
def get_gsa_heatmap(data: dict[str, Any], nice_names: bool = True) -> plotly.graph_objects.Figure:
    """GSA heatmap component."""
    
    if not data:
        logger.debug("No heatmap data found")
        return px.imshow(
            pd.DataFrame(),
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            aspect="auto",
        )
    
    df = pd.DataFrame(data).set_index("param")
    logger.debug(f"Heatmap data shape: {df.shape}")
    
    if nice_names:
        logger.debug("Applying nice names to heatmap")
        gsa_params = _unflatten_dropdown_options(GSA_PARM_OPTIONS)
        gsa_results = _unflatten_dropdown_options(GSA_RESULT_OPTIONS)
        df = df.rename(columns=gsa_results).rename(index=gsa_params)
    
    fig = px.imshow(
        df,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        aspect="auto",  # Automatically adjust aspect ratio
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

    