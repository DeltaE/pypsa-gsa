"""Component to display uncertainity data for the full system."""

from dash import dcc, html

from .data import (
    RESULT_SUMMARY_TYPE_DROPDOWN_OPTIONS,
)


from . import ids as ids
# from .utils import _unflatten_dropdown_options
# import plotly.graph_objects as go
# import plotly.express as px

import logging

logger = logging.getLogger(__name__)


def ua2_options_block() -> html.Div:
    """UA system options block component."""
    return html.Div(
        [
            ua2_result_type_dropdown(),
            ua2_result_dropdown(),
            ua2_percentile_interval_slider(),
        ],
    )


def ua2_result_type_dropdown() -> html.Div:
    """UA2 result type dropdown component."""
    return html.Div(
        [
            html.H6("Select Result Type"),
            dcc.Dropdown(
                id=ids.UA2_RESULTS_TYPE_DROPDOWN,
                options=RESULT_SUMMARY_TYPE_DROPDOWN_OPTIONS,
                value="cost",
            ),
        ],
    )


def ua2_result_dropdown() -> html.Div:
    """UA result type dropdown component."""
    return html.Div(
        [
            html.H6("Select Result"),
            dcc.Dropdown(
                id=ids.UA2_RESULTS_DROPDOWN,
                options=[],
                value="objective_cost",
            ),
        ],
    )


def ua2_percentile_interval_slider() -> html.Div:
    """UA system slider component."""

    return html.Div(
        [
            html.H6("Percentile Interval:"),
            dcc.RangeSlider(
                id=ids.UA2_INTERVAL_SLIDER,
                min=0,
                max=100,
                value=[0, 100],
                step=1,
                included=False,
                marks={x: str(x) for x in range(0, 101, 25)},
                tooltip={
                    "placement": "bottom",
                    "always_visible": False,
                    "template": "{value}%",
                },
            ),
        ],
    )
