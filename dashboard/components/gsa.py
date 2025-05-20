"""Component to display GSA data."""

from typing import Any
from dash import dcc, html
from pathlib import Path
from .utils import (
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
)
from . import ids as ids

import pandas as pd
import plotly
import plotly.express as px

root = Path(__file__).parent.parent

GSA_PARM_OPTIONS = get_gsa_params_dropdown_options(root)
GSA_RESULT_OPTIONS = get_gsa_results_dropdown_options(root)

def gsa_params_dropdown() -> html.Div:
    """GSA parameters dropdown component."""
    return html.Div(
        [
            html.Label("Select Parameter"),
            dcc.Dropdown(
                id=ids.GSA_PARAM_DROPDOWN,
                options=GSA_PARM_OPTIONS,
                value=GSA_PARM_OPTIONS[0]["value"] if GSA_PARM_OPTIONS else None,
            ),
        ],
        className="dropdown-container",
    )

def gsa_results_dropdown() -> html.Div:
    """GSA results dropdown component."""
    return html.Div(
        [
            html.Label("Select Result"),
            dcc.Dropdown(
                id=ids.GSA_RESULTS_DROPDOWN,
                options=GSA_RESULT_OPTIONS,
                value=GSA_RESULT_OPTIONS[0]["value"] if GSA_RESULT_OPTIONS else None,
            ),
        ],
        className="dropdown-container",
    )
    
def get_gsa_heatmap(data: dict[str, Any]) -> plotly.graph_objects.Figure:
    """GSA heatmap component."""
    df = pd.DataFrame(data)
    
    fig = px.imshow(
        df,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        aspect="auto",  # Automatically adjust aspect ratio
        labels=dict(x="Parameters", y="Results", color="Sensitivity"),
    )
    
    fig.update_layout(
        title="Global Sensitivity Analysis",
        xaxis_title="Parameters",
        yaxis_title="Results",
        height=600,
        xaxis={'side': 'bottom'}
    )
    fig.update_xaxes(tickangle=45)
    
    return fig

    