"""Shared components for the dashboard."""

from dash import html, dcc
from . import ids as ids
from .utils import get_iso_dropdown_options

ISO_OPTIONS = get_iso_dropdown_options()

print([x["value"] for x in ISO_OPTIONS] if ISO_OPTIONS else None)

def iso_dropdown() -> html.Div:
    """ISO dropdown component."""
    return html.Div(
        [
            html.Label("ISO"),
            dcc.Dropdown(
                id=ids.ISO_DROPDOWN,
                options=ISO_OPTIONS,
                value=[x["value"] for x in ISO_OPTIONS] if ISO_OPTIONS else None,
                multi=True
            ),
        ],
        className="dropdown-container",
    )