"""Component to display uncertainity data."""

from dash import dcc, html
from pathlib import Path
from .utils import (
    get_ua_params_dropdown_options,
    get_ua_results_dropdown_options,
)
from . import ids as ids

root = Path(__file__).parent.parent

UA_PARM_OPTIONS = get_ua_params_dropdown_options(root)
UA_RESULT_OPTIONS = get_ua_results_dropdown_options(root)


def ua_options_block() -> html.Div:
    """UA options block component."""
    return html.Div(
        [
            html.H6("Select Parameter"),
            ua_params_dropdown(),
            html.H6("Select Result"),
            ua_results_dropdown(),
        ],
    )


def ua_params_dropdown() -> html.Div:
    """UA parameters dropdown component."""
    return html.Div(
        [
            html.H6("Select Parameter"),
            dcc.Dropdown(
                id=ids.UA_PARAM_DROPDOWN,
                options=UA_PARM_OPTIONS,
                value=UA_PARM_OPTIONS[0]["value"] if UA_PARM_OPTIONS else None,
            ),
        ],
        className="dropdown-container",
    )


def ua_results_dropdown() -> html.Div:
    """UA results dropdown component."""
    return html.Div(
        [
            html.H6("Select Result"),
            dcc.Dropdown(
                id=ids.UA_RESULTS_DROPDOWN,
                options=UA_RESULT_OPTIONS,
                value=UA_RESULT_OPTIONS[0]["value"] if UA_RESULT_OPTIONS else None,
            ),
        ],
        className="dropdown-container",
    )
