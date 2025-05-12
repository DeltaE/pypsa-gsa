"""Component to display GSA data."""

from dash import dcc, html
from pathlib import Path
from .utils import (
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
)
from . import ids as ids

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
    
def plot_gsa_heatmap() -> html.Div:
    """GSA heatmap component."""
    return html.Div(
        [
            dcc.Graph(
                id=ids.GSA_HEATAMP,
                config={"displayModeBar": False},
                style={"height": "100%"},
            ),
        ],
        className="graph-container",
    )