"""Main app interface."""

from typing import Any
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

from components.data import (
    EMISSIONS,
    METADATA,
    RAW_GSA,
    RAW_UA,
    RAW_PARAMS,
    ISO_SHAPE,
    GSA_PARM_OPTIONS,
    GSA_RESULT_OPTIONS,
    SECTOR_DROPDOWN_OPTIONS_ALL,
    SECTOR_DROPDOWN_OPTIONS_SYSTEM,
    SECTOR_DROPDOWN_OPTIONS_IDV,
    CR_PARAM_OPTIONS,
    CR_DATA,
    SECTOR_DROPDOWN_OPTIONS_SYSTEM_POWER_NG,
    SECTOR_DROPDOWN_OPTIONS_TRADE,
)
import components.ids as ids
from components.utils import (
    DEFAULT_PLOTLY_THEME,
    get_continuous_color_scale_options,
    get_cr_result_types_dropdown_options,
    get_cr_results_dropdown_options,
    get_discrete_color_scale_options,
    DEFAULT_CONTINOUS_COLOR_SCALE,
    DEFAULT_DISCRETE_COLOR_SCALE,
    get_plotly_plotting_themes,
)
from components.gsa import (
    GSA_RB_OPTIONS,
    filter_gsa_data,
    filter_gsa_data_for_map,
    get_gsa_map,
    gsa_options_block,
    get_gsa_heatmap,
    get_gsa_data_table,
    get_gsa_barchart,
    normalize_mu_star_data,
)
from components.shared import iso_options_block, plotting_options_block
from components.ua import (
    filter_ua_on_result_sector_and_type,
    get_ua_box_whisker,
    get_ua_data_table,
    get_ua_histogram,
    get_ua_scatter_plot,
    get_ua_violin_plot,
    remove_ua_outliers,
    ua_options_block,
    get_ua_barchart,
)
from components.input_data import (
    get_input_data_barchart,
    get_inputs_data_table,
    input_data_options_block,
)
from components.cr import (
    cr_options_block,
    get_cr_data_table,
    get_cr_scatter_plot,
)

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

SIDEBAR_WIDTH = 3
CONTENT_WIDTH = int(12 - SIDEBAR_WIDTH)

OPTIONS_BLOCK_CLASS = "py-1"


########
# layout
########

app.layout = html.Div(
    [
        dbc.NavbarSimple(
            brand="PyPSA-USA Sector Uncertainity Analysis",
            brand_href="#",
            color="primary",
            dark=True,
            id=ids.NAVBAR,
            className="mb-3",
        ),
        dbc.Container(
            [
                # html.H1("PyPSA-USA Uncertainity Analysis", className="my-4"),
                dbc.Row(
                    [
                        # not using collapsable cards as when you collapse the cards
                        # padding is retained and it looks weird. lol.
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Data Viewing Options",
                                                            className="card-title",
                                                        ),
                                                        plotting_options_block(),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    id=ids.PLOTTING_OPTIONS_BLOCK,
                                    className=OPTIONS_BLOCK_CLASS,
                                ),
                                html.Div(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Spatial Options",
                                                            className="card-title",
                                                        ),
                                                        iso_options_block(
                                                            RAW_GSA, RAW_UA
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    id=ids.ISO_OPTIONS_BLOCK,
                                    className=OPTIONS_BLOCK_CLASS,
                                ),
                                html.Div(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "GSA Options",
                                                            className="card-title",
                                                        ),
                                                        gsa_options_block(),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    id=ids.GSA_OPTIONS_BLOCK,
                                    className=OPTIONS_BLOCK_CLASS,
                                ),
                                html.Div(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Input Data Options",
                                                            className="card-title",
                                                        ),
                                                        input_data_options_block(),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    id=ids.INPUT_DATA_OPTIONS_BLOCK,
                                    className=OPTIONS_BLOCK_CLASS,
                                ),
                                html.Div(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Uncertaintiy Options",
                                                            className="card-title",
                                                        ),
                                                        ua_options_block(),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    id=ids.UA_OPTIONS_BLOCK,
                                    className=OPTIONS_BLOCK_CLASS,
                                ),
                                html.Div(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Custom Result Options",
                                                            className="card-title",
                                                        ),
                                                        cr_options_block(),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    id=ids.CR_OPTIONS_BLOCK,
                                    className=OPTIONS_BLOCK_CLASS,
                                ),
                            ],
                            md=SIDEBAR_WIDTH,
                        ),
                        dbc.Col(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            label="Input Data",
                                            tab_id=ids.DATA_TAB,
                                        ),
                                        dbc.Tab(
                                            label="Sensitivity Analysis",
                                            tab_id=ids.SA_TAB,
                                        ),
                                        dbc.Tab(
                                            label="Uncertainty Analysis",
                                            tab_id=ids.UA_TAB,
                                        ),
                                        dbc.Tab(
                                            label="Custom Result",
                                            tab_id=ids.CR_TAB,
                                        ),
                                    ],
                                    id=ids.TABS,
                                    active_tab=ids.SA_TAB,
                                    className="mb-4",
                                ),
                                html.Div(id=ids.TAB_CONTENT),
                            ],
                            md=CONTENT_WIDTH,
                        ),
                    ]
                ),
                # Store components to share data between callbacks
                dcc.Store(id=ids.GSA_ISO_DATA),
                dcc.Store(id=ids.GSA_ISO_NORMED_DATA),
                dcc.Store(id=ids.GSA_HM_DATA),
                dcc.Store(id=ids.GSA_BAR_DATA),
                dcc.Store(id=ids.GSA_MAP_DATA),
                dcc.Store(id=ids.UA_ISO_DATA),
                dcc.Store(id=ids.UA_RUN_DATA),
                dcc.Store(id=ids.CR_DATA),
                dcc.Store(id=ids.INPUTS_DATA),
                dcc.Store(id=ids.INPUTS_DATA_BY_ATTRIBUTE),
                dcc.Store(id=ids.INPUTS_DATA_BY_ATTRIBUTE_CARRIER),
                dcc.Store(id=ids.UA_EMISSIONS),
                dcc.Store(id=ids.CR_EMISSIONS),
                dcc.Store(id=ids.GSA_PARAM_BUTTON_STATE, data=""),
                dcc.Store(id=ids.GSA_RESULTS_BUTTON_STATE, data=""),
            ],
            fluid=True,
        ),
    ]
)

########################
# Update tab content
########################


@app.callback(
    Output(ids.TAB_CONTENT, "children"),
    [
        Input(ids.TABS, "active_tab"),
        Input(ids.GSA_HM_DATA, "data"),
        Input(ids.GSA_BAR_DATA, "data"),
        Input(ids.GSA_MAP_DATA, "data"),
        Input(ids.UA_RUN_DATA, "data"),
        Input(ids.CR_DATA, "data"),
        Input(ids.INPUTS_DATA_BY_ATTRIBUTE_CARRIER, "data"),
        Input(ids.UA_EMISSIONS, "data"),
        Input(ids.CR_EMISSIONS, "data"),
        Input(ids.PLOTTING_TYPE_DROPDOWN, "value"),
        Input(ids.COLOR_DROPDOWN, "value"),
    ],
    State(ids.UA_RESULTS_TYPE_DROPDOWN, "value"),
    State(ids.CR_RESULT_TYPE_DROPDOWN, "value"),
)
def render_tab_content(
    active_tab: str,
    gsa_hm_data: list[dict[str, Any]] | None,
    gsa_bar_data: list[dict[str, Any]] | None,
    gsa_map_data: list[dict[str, Any]] | None,
    ua_run_data: list[dict[str, Any]] | None,
    cr_data: list[dict[str, Any]] | None,
    inputs_data: list[dict[str, Any]] | None,
    ua_emissions: list[dict[str, Any]] | None,
    cr_emissions: list[dict[str, Any]] | None,
    plotting_type: str,
    color: str,
    ua_result_type: str,
    cr_result_type: str,
) -> html.Div:
    logger.debug(f"Rendering tab content for: {active_tab}")
    if active_tab == ids.DATA_TAB:
        if plotting_type == "data_table":
            view = get_inputs_data_table(inputs_data)
        elif plotting_type == "barchart":
            view = get_input_data_barchart(inputs_data, color_scale=color)
        else:
            view = html.Div([dbc.Alert("No plotting type selected", color="info")])
        return html.Div([dbc.Card([dbc.CardBody([view])])])
    elif active_tab == ids.SA_TAB:
        if plotting_type == "heatmap":
            view = dcc.Graph(
                id=ids.GSA_HEATMAP,
                figure=get_gsa_heatmap(gsa_hm_data, color_scale=color),
            )
        elif plotting_type == "data_table":
            view = get_gsa_data_table(gsa_hm_data)
        elif plotting_type == "barchart":
            view = dcc.Graph(
                id=ids.GSA_BAR_CHART,
                figure=get_gsa_barchart(gsa_bar_data, color_scale=color),
            )
        elif plotting_type == "map":
            view = get_gsa_map(gsa_map_data, ISO_SHAPE, color_scale=color)
        else:
            return html.Div([dbc.Alert("No plotting type selected", color="info")])
        return html.Div([dbc.Card([dbc.CardBody([view])])])
    elif active_tab == ids.UA_TAB:
        if plotting_type == "data_table":
            view = get_ua_data_table(ua_run_data)
        elif plotting_type == "barchart":
            view = dcc.Graph(
                id=ids.UA_BAR_CHART,
                figure=get_ua_barchart(
                    ua_run_data,
                    template=color,
                    result_type=ua_result_type,
                    emissions=ua_emissions if ua_result_type == "emissions" else None,
                ),
            )
        elif plotting_type == "violin":
            view = dcc.Graph(
                id=ids.UA_VIOLIN,
                figure=get_ua_violin_plot(
                    ua_run_data,
                    template=color,
                    result_type=ua_result_type,
                    emissions=ua_emissions if ua_result_type == "emissions" else None,
                ),
            )
        elif plotting_type == "scatter":
            view = dcc.Graph(
                id=ids.UA_SCATTER,
                figure=get_ua_scatter_plot(
                    ua_run_data,
                    template=color,
                    result_type=ua_result_type,
                    emissions=ua_emissions if ua_result_type == "emissions" else None,
                ),
            )
        elif plotting_type == "histogram":
            view = get_ua_histogram(
                ua_run_data, template=color, result_type=ua_result_type
            )
        elif plotting_type == "box_whisker":
            view = dcc.Graph(
                id=ids.UA_BOX_WHISKER,
                figure=get_ua_box_whisker(
                    ua_run_data,
                    template=color,
                    result_type=ua_result_type,
                    emissions=ua_emissions if ua_result_type == "emissions" else None,
                ),
            )
        else:
            return html.Div([dbc.Alert("No plotting type selected", color="info")])
        return html.Div([dbc.Card([dbc.CardBody([view])])])
    elif active_tab == ids.CR_TAB:
        if plotting_type == "data_table":
            view = get_cr_data_table(cr_data)
        elif plotting_type == "scatter":
            view = dcc.Graph(
                id=ids.UA_SCATTER,
                figure=get_cr_scatter_plot(
                    cr_data,
                    template=color,
                    marginal=None,
                    result_type=cr_result_type,
                    emissions=cr_emissions if cr_result_type == "emissions" else None,
                ),
            )
        elif plotting_type == "scatter-box":
            view = dcc.Graph(
                id=ids.UA_SCATTER,
                figure=get_cr_scatter_plot(
                    cr_data,
                    template=color,
                    marginal="box",
                    result_type=cr_result_type,
                    emissions=cr_emissions if cr_result_type == "emissions" else None,
                ),
            )
        elif plotting_type == "scatter-histogram":
            view = dcc.Graph(
                id=ids.UA_SCATTER,
                figure=get_cr_scatter_plot(
                    cr_data,
                    template=color,
                    marginal="histogram",
                    result_type=cr_result_type,
                    emissions=cr_emissions if cr_result_type == "emissions" else None,
                ),
            )
        elif plotting_type == "scatter-rug":
            view = dcc.Graph(
                id=ids.UA_SCATTER,
                figure=get_cr_scatter_plot(
                    cr_data,
                    template=color,
                    marginal="rug",
                    result_type=cr_result_type,
                    emissions=cr_emissions if cr_result_type == "emissions" else None,
                ),
            )
        elif plotting_type == "scatter-violin":
            view = dcc.Graph(
                id=ids.UA_SCATTER,
                figure=get_cr_scatter_plot(
                    cr_data,
                    template=color,
                    marginal="violin",
                    result_type=cr_result_type,
                    emissions=cr_emissions if cr_result_type == "emissions" else None,
                ),
            )
        else:
            return html.Div([dbc.Alert("Custom Result", color="info")])
        return html.Div([dbc.Card([dbc.CardBody([view])])])
    logger.error(f"No active tab selected: {active_tab}")
    return html.Div([dbc.Alert("No active tab selected", color="info")])


@app.callback(
    [
        Output(ids.PLOTTING_TYPE_DROPDOWN, "options"),
        Output(ids.PLOTTING_TYPE_DROPDOWN, "value"),
    ],
    Input(ids.TABS, "active_tab"),
)
def update_plotting_type_dropdown_options(
    active_tab: str,
) -> tuple[list[dict[str, str]], str]:
    """Update the plotting type dropdown options based on the active tab."""
    logger.debug(f"Updating plotting type dropdown options for: {active_tab}")
    if active_tab == ids.DATA_TAB:
        return (
            [
                {"label": "Bar Chart", "value": "barchart"},
                {"label": "Data Table", "value": "data_table"},
            ],
            "data_table",
        )
    elif active_tab == ids.UA_TAB:
        return (
            [
                {"label": "Bar Chart", "value": "barchart"},
                {"label": "Box Whisker", "value": "box_whisker"},
                {"label": "Data Table", "value": "data_table"},
                {"label": "Histogram", "value": "histogram"},
                {"label": "Scatter Plot", "value": "scatter"},
                {"label": "Violin Plot", "value": "violin"},
            ],
            "scatter",
        )
    elif active_tab == ids.SA_TAB:
        return (
            [
                {"label": "Bar Chart", "value": "barchart"},
                {"label": "Data Table", "value": "data_table"},
                {"label": "Heatmap", "value": "heatmap"},
                {"label": "Map", "value": "map"},
            ],
            "heatmap",
        )
    elif active_tab == ids.CR_TAB:
        return (
            [
                {"label": "Data Table", "value": "data_table"},
                {"label": "Scatter", "value": "scatter"},
                {"label": "Scatter (Box)", "value": "scatter-box"},
                {"label": "Scatter (Histogram)", "value": "scatter-histogram"},
                {"label": "Scatter (Rug)", "value": "scatter-rug"},
                {"label": "Scatter (Violin)", "value": "scatter-violin"},
            ],
            "scatter",
        )
    else:
        logger.debug(f"Invalid active tab for plotting type dropdown: {active_tab}")
        return ([{"label": "Data Table", "value": "data_table"}], "data_table")


#############################
# Enable/disable option cards
#############################


@app.callback(
    [
        Output(ids.PLOTTING_OPTIONS_BLOCK, "style"),
        Output(ids.ISO_OPTIONS_BLOCK, "style"),
        Output(ids.INPUT_DATA_OPTIONS_BLOCK, "style"),
        Output(ids.GSA_OPTIONS_BLOCK, "style"),
        Output(ids.UA_OPTIONS_BLOCK, "style"),
        Output(ids.CR_OPTIONS_BLOCK, "style"),
    ],
    Input(ids.TABS, "active_tab"),
)
def callback_show_hide_option_blocks(
    active_tab: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Show/hide option blocks based on active tab."""
    # Default style to hide blocks
    hidden = {"display": "none"}
    visible = {"display": "block"}

    # Start with all hidden
    plotting_style = hidden
    iso_options = hidden
    input_data_options = hidden
    gsa_options = hidden
    ua_options = hidden
    cr_options = hidden

    plotting_style = visible

    if active_tab == ids.DATA_TAB:
        iso_options = visible
        input_data_options = visible
    elif active_tab == ids.SA_TAB:
        iso_options = visible
        gsa_options = visible
    elif active_tab == ids.UA_TAB:
        iso_options = visible
        ua_options = visible
    elif active_tab == ids.CR_TAB:
        cr_options = visible

    return (
        plotting_style,
        iso_options,
        input_data_options,
        gsa_options,
        ua_options,
        cr_options,
    )


###################################
# Enable/disable collapsable blocks
###################################


@app.callback(
    [
        Output(ids.GSA_PARAMS_SLIDER_COLLAPSE, "is_open"),
        Output(ids.GSA_PARAMS_RESULTS_COLLAPSE, "is_open"),
    ],
    Input(ids.GSA_PARAM_SELECTION_RB, "value"),
)
def callback_enable_disable_gsa_param_selection(value: str) -> tuple[bool, bool]:
    """Enable/disable GSA parameter selection collapse based on filtering mode."""
    params_slider_open = False
    params_dropdown_open = False
    if value == "rank":
        params_slider_open = True
    elif value == "name":
        params_dropdown_open = True
    else:
        logger.debug(f"Invalid GSA radio button value: {value}")
        params_slider_open = True
        params_dropdown_open = True
    return params_slider_open, params_dropdown_open


#################
# Get stored data
#################

#####################
# uncertainity inputs
#####################


@app.callback(
    Output(ids.UA_EMISSIONS, "data"),
    [
        Input(ids.ISO_DROPDOWN, "value"),
        Input(ids.UA_EMISSION_TARGET_RB, "value"),
    ],
)
def callback_update_ua_emissions(
    isos: list[str], emission_target: bool
) -> dict[str, dict[str, float]]:
    if emission_target:
        return {x: EMISSIONS[x] for x in isos}
    else:
        return {}


@app.callback(
    Output(ids.INPUTS_DATA, "data"),
    Input(ids.ISO_DROPDOWN, "value"),
)
def callback_update_inputs_data(isos: list[str]) -> list[dict[str, Any]]:
    logger.debug(f"ISO dropdown value: {isos}")
    if not isos:
        logger.debug("No ISOs selected from dropdown")
        return []
    isos.append("all")
    data = RAW_PARAMS[RAW_PARAMS.iso.isin(isos)].to_dict("records")
    return data


@app.callback(
    [
        Output(ids.INPUTS_DATA_BY_ATTRIBUTE, "data"),
        Output(ids.INPUT_DATA_SECTOR_DROPDOWN, "options"),
        Output(ids.INPUT_DATA_SECTOR_DROPDOWN, "disabled"),
    ],
    [
        Input(ids.INPUTS_DATA, "data"),
        Input(ids.INPUT_DATA_ATTRIBUTE_DROPDOWN, "value"),
    ],
)
def callback_update_inputs_data_attribute(
    data: list[dict[str, Any]], attribute: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]], bool]:
    logger.debug(f"Attribute value: {attribute}")
    if not data:
        logger.debug("No data provided")
        return [], [], True
    if not attribute:
        logger.debug("No attribute provided")
        return data, [], True
    df = pd.DataFrame(data)
    df = df[df.attribute == attribute]
    if df.empty:
        logger.debug("No data found for the given attribute")
        return [], [], True
    options = [{"label": sector, "value": sector} for sector in df.sector.unique()]
    return df.to_dict("records"), options, False


@app.callback(
    Output(ids.INPUTS_DATA_BY_ATTRIBUTE_CARRIER, "data"),
    [
        Input(ids.INPUTS_DATA_BY_ATTRIBUTE, "data"),
        Input(ids.INPUT_DATA_SECTOR_DROPDOWN, "value"),
    ],
)
def callback_update_inputs_data_attribute_sector(
    data: list[dict[str, Any]], sector: str
) -> list[dict[str, Any]]:
    logger.debug(f"Sector value: {sector}")
    if not data:
        logger.debug("No data provided")
        return []
    if not sector:
        logger.debug("No sector provided")
        return data
    df = pd.DataFrame(data)
    df = df[df.sector == sector]
    if df.empty:
        logger.debug("No data found for the given sector")
        return []
    return df.to_dict("records")


@app.callback(
    Output(ids.UA_EMISSION_TARGET_RB, "options"),
    Input(ids.UA_RESULTS_TYPE_DROPDOWN, "value"),
)
def disable_ua_emissions_target_rb(result_type: str) -> list[dict[str, str]]:
    if result_type == "emissions":
        return [
            {"label": "True", "value": True, "disabled": False},
            {"label": "False", "value": False, "disabled": False},
        ]
    else:
        return [
            {"label": "True", "value": True, "disabled": True},
            {"label": "False", "value": False, "disabled": True},
        ]


#####
# gsa
#####


@app.callback(
    Output(ids.GSA_ISO_DATA, "data"),
    Input(ids.ISO_DROPDOWN, "value"),
)
def callback_filter_gsa_on_iso(isos: list[str]) -> list[dict[str, Any]]:
    """Update the GSA store data based on the selected ISOs."""
    logger.debug(f"ISO dropdown value: {isos}")
    if not isos:
        logger.debug("No ISOs selected from dropdown")
        return []
    data = RAW_GSA[RAW_GSA.iso.isin(isos)].to_dict("records")
    return data


@app.callback(
    Output(ids.GSA_ISO_NORMED_DATA, "data"),
    Input(ids.GSA_ISO_DATA, "data"),
)
def callback_normalize_mu_star_data(
    gsa_iso_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize data for the mu/mu_max calculation.

    Treat this as a store so that it is not recalculated every time the data is filtered,
    as this is independent of the result selections.
    """
    df = pd.DataFrame(gsa_iso_data)
    logger.debug(f"Normalizing GSA data of shape: {df.shape}")
    return normalize_mu_star_data(df).to_dict("records")


@app.callback(
    Output(ids.GSA_HM_DATA, "data"),
    [
        Input(ids.GSA_ISO_DATA, "data"),
        Input(ids.GSA_PARAM_SELECTION_RB, "value"),
        Input(ids.GSA_PARAM_DROPDOWN, "value"),
        Input(ids.GSA_PARAMS_SLIDER, "value"),
        Input(ids.GSA_RESULTS_DROPDOWN, "value"),
    ],
    State(ids.PLOTTING_TYPE_DROPDOWN, "value"),
)
def callback_filter_gsa_data_for_heatmap(
    gsa_iso_data: list[dict[str, Any]] | None,
    param_option: str,
    params_dropdown: str | list[str],
    params_slider: int,
    results: str | list[str],
    plotting_type: str,
) -> list[dict[str, Any]]:
    """Update the GSA SI data for the data table."""
    if plotting_type == "map":  # map updates some dropdowns
        return dash.no_update
    df = filter_gsa_data(
        data=gsa_iso_data,
        param_option=param_option,
        params_dropdown=params_dropdown,
        params_slider=params_slider,
        results=results,
        keep_iso=False,
    )
    return df.reset_index().to_dict("records")


@app.callback(
    Output(ids.GSA_BAR_DATA, "data"),
    [
        Input(ids.GSA_ISO_NORMED_DATA, "data"),
        Input(ids.GSA_PARAM_SELECTION_RB, "value"),
        Input(ids.GSA_PARAM_DROPDOWN, "value"),
        Input(ids.GSA_PARAMS_SLIDER, "value"),
        Input(ids.GSA_RESULTS_DROPDOWN, "value"),
    ],
    State(ids.PLOTTING_TYPE_DROPDOWN, "value"),
)
def callback_filter_gsa_data_for_barchart(
    gsa_iso_normed_data: list[dict[str, Any]] | None,
    param_option: str,
    params_dropdown: str | list[str],
    params_slider: int,
    results: str | list[str],
    plotting_type: str,
) -> list[dict[str, Any]]:
    """Update the GSA barchart."""
    if plotting_type == "map":  # map updates some dropdowns
        return dash.no_update
    df = filter_gsa_data(
        data=gsa_iso_normed_data,
        param_option=param_option,
        params_dropdown=params_dropdown,
        params_slider=params_slider,
        results=results,
        keep_iso=False,
    )
    return df.reset_index().to_dict("records")


@app.callback(
    Output(ids.GSA_MAP_DATA, "data"),
    [
        Input(ids.GSA_ISO_DATA, "data"),
        Input(ids.GSA_PARAMS_SLIDER, "value"),
        Input(ids.GSA_RESULTS_DROPDOWN, "value"),
    ],
)
def callback_filter_gsa_data_for_map(
    gsa_iso_data: list[dict[str, Any]] | None,
    params_slider: int,
    result: str,  # only one result for map
    nice_names: bool = True,
) -> list[dict[str, Any]]:
    """Update the GSA barchart."""
    df = filter_gsa_data_for_map(
        data=gsa_iso_data,
        params_slider=params_slider,
        result=result,
    )
    if nice_names:
        logger.debug("Using nice names for GSA map")
        nn = {x["value"]: x["label"] for x in GSA_PARM_OPTIONS}
        df = df.replace(nn)
    return df.reset_index(names="iso").to_dict("records")


#####
# ua
#####


@app.callback(
    Output(ids.UA_ISO_DATA, "data"),
    Input(ids.ISO_DROPDOWN, "value"),
)
def callback_filter_ua_on_iso(isos: list[str]) -> list[dict[str, Any]]:
    """Update the UA store data based on the selected ISOs."""
    logger.debug(f"ISO dropdown value: {isos}")
    if not isos:
        logger.debug("No ISOs selected from dropdown")
        return []
    return RAW_UA[RAW_UA.iso.isin(isos)].to_dict("records")


@app.callback(
    Output(ids.UA_RUN_DATA, "data"),
    [
        Input(ids.UA_ISO_DATA, "data"),
        Input(ids.UA_RESULTS_TYPE_DROPDOWN, "value"),
        Input(ids.UA_RESULTS_SECTOR_DROPDOWN, "value"),
        Input(ids.UA_INTERVAL_SLIDER, "value"),
    ],
)
def callback_filter_ua_on_result_sector_and_type(
    data: list[dict[str, Any]],
    result_type: str,
    result_sector: str,
    interval: list[int],
) -> list[dict[str, Any]]:
    df = pd.DataFrame(data)
    df = filter_ua_on_result_sector_and_type(df, result_sector, result_type, METADATA)
    df = remove_ua_outliers(df, interval)
    return df.to_dict("records")


###########################
# Shared Options Callbacks
###########################


@app.callback(
    [
        Output(ids.COLOR_DROPDOWN, "value"),
        Output(ids.COLOR_DROPDOWN, "options"),
    ],
    [
        Input(ids.TABS, "active_tab"),
        Input(ids.PLOTTING_TYPE_DROPDOWN, "value"),
    ],
)
def callback_update_color_options(
    active_tab: str, plotting_type: str
) -> tuple[str, list[str]]:
    logger.debug(f"Updating color options for: {plotting_type} in {active_tab}")
    if active_tab == ids.DATA_TAB:
        return DEFAULT_PLOTLY_THEME, get_plotly_plotting_themes()
    elif active_tab == ids.UA_TAB:
        return DEFAULT_PLOTLY_THEME, get_plotly_plotting_themes()
    elif active_tab == ids.SA_TAB:
        if plotting_type in ["heatmap"]:
            logger.debug("Updating continuous color options")
            return DEFAULT_CONTINOUS_COLOR_SCALE, get_continuous_color_scale_options()
        else:
            return DEFAULT_DISCRETE_COLOR_SCALE, get_discrete_color_scale_options()
    elif active_tab == ids.CR_TAB:
        return DEFAULT_PLOTLY_THEME, get_plotly_plotting_themes()
    else:
        return "", []


######################
# Input Data Callbacks
######################


@app.callback(
    Output(ids.INPUT_DATA_ATTRIBUTE_DROPDOWN, "options"),
    Input(ids.INPUTS_DATA, "data"),
)
def callback_input_data_attribute_dropdown(
    data: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if not data:
        logger.debug("No input data provided")
        return {}
    return [
        {"label": attr[0], "value": attr[1]}
        for attr in sorted(
            set((x["attribute_nice_name"], x["attribute"]) for x in data),
            key=lambda x: x[0],  # sort based on nice_name
        )
    ]  # only unique elements


@app.callback(
    Output(ids.INPUT_DATA_SECTOR_DROPDOWN, "value", allow_duplicate=True),
    [
        Input(ids.INPUT_DATA_ATTRIBUTE_DROPDOWN, "value"),
        Input(ids.INPUT_DATA_SECTOR_DROPDOWN, "options"),
    ],
    prevent_initial_call=True,
)
def callback_update_input_data_attribute_dropdown(
    _: str, options: list[dict[str, str]]
) -> str:
    if not options:
        return ""
    return options[0]["value"]


@app.callback(
    [
        Output(ids.INPUT_DATA_ATTRIBUTE_DROPDOWN, "value"),
        Output(ids.INPUT_DATA_SECTOR_DROPDOWN, "value", allow_duplicate=True),
    ],
    Input(ids.INPUT_DATA_REMOVE_FILTERS, "n_clicks"),
    prevent_initial_call=True,
)
def callback_remove_input_data_filters(_: int) -> tuple[str, str]:
    return "", ""


########################
# GSA Options Callbacks
########################


@app.callback(
    Output(ids.GSA_PARAM_DROPDOWN, "value"),
    State(ids.PLOTTING_TYPE_DROPDOWN, "value"),
    [
        Input(ids.GSA_PARAM_SELECT_ALL, "n_clicks"),
        Input(ids.GSA_PARAM_REMOVE_ALL, "n_clicks"),
    ],
    prevent_initial_call=True,
)
def callback_select_remove_all_gsa_params(plotting_type: str, *args: Any) -> list[str]:
    """Select/remove all parameters when button is clicked."""
    if plotting_type == "map":
        return dash.no_update
    ctx = dash.callback_context  # to determine which button was clicked
    if not ctx.triggered:
        return dash.no_update

    # id of the button that was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    logger.debug(f"Triggered GSA param button ID of: {button_id}")

    if button_id == ids.GSA_PARAM_SELECT_ALL:
        return [option["value"] for option in GSA_PARM_OPTIONS]
    elif button_id == ids.GSA_PARAM_REMOVE_ALL:
        return []
    return dash.no_update


@app.callback(
    Output(ids.GSA_RESULTS_DROPDOWN, "value", allow_duplicate=True),
    State(ids.PLOTTING_TYPE_DROPDOWN, "value"),
    [
        Input(ids.GSA_RESULTS_SELECT_ALL, "n_clicks"),
        Input(ids.GSA_RESULTS_REMOVE_ALL, "n_clicks"),
    ],
    prevent_initial_call=True,
)
def callback_select_remove_all_gsa_results(plotting_type: str, *args: Any) -> list[str]:
    """Select/remove all results when button is clicked."""
    if plotting_type == "map":
        return dash.no_update
    ctx = dash.callback_context  # to determine which button was clicked
    if not ctx.triggered:
        return dash.no_update

    # id of the button that was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    logger.debug(f"Triggered GSA result button ID of: {button_id}")

    if button_id == ids.GSA_RESULTS_SELECT_ALL:
        return [option["value"] for option in GSA_RESULT_OPTIONS]
    elif button_id == ids.GSA_RESULTS_REMOVE_ALL:
        return []
    return dash.no_update


@app.callback(
    [
        Output(ids.GSA_RESULTS_DROPDOWN, "multi"),
        Output(ids.GSA_RESULTS_DROPDOWN, "value", allow_duplicate=True),
        Output(ids.GSA_RESULTS_SELECT_ALL, "disabled"),
        Output(ids.GSA_RESULTS_REMOVE_ALL, "disabled"),
    ],
    Input(ids.PLOTTING_TYPE_DROPDOWN, "value"),
    State(ids.TABS, "active_tab"),
    State(ids.GSA_RESULTS_DROPDOWN, "value"),
    prevent_initial_call="initial_duplicate",  # for the allow duplicates
)
def callback_update_gsa_results_dropdown(
    plotting_type: str, active_tab: str, current_results: list[str]
) -> tuple[bool, bool, bool]:
    if active_tab != ids.SA_TAB:
        return dash.no_update
    if plotting_type == "map":
        if isinstance(current_results, str):
            current_results = [current_results]
        return False, current_results[0], True, True
    else:
        return True, current_results, False, False


@app.callback(
    Output(ids.GSA_PARAM_SELECTION_RB, "options"),
    Input(ids.PLOTTING_TYPE_DROPDOWN, "value"),
    State(ids.TABS, "active_tab"),
)
def callback_update_gsa_rb(plotting_type: str, active_tab: str) -> tuple[str, bool]:
    if active_tab != ids.SA_TAB:
        return dash.no_update
    options = GSA_RB_OPTIONS.copy()
    if plotting_type == "map":
        for option in options:
            option["disabled"] = True
        return options
    else:
        for option in options:
            option["disabled"] = False
        return options


@app.callback(
    [
        Output(ids.GSA_PARAMS_SLIDER, "max"),
        Output(ids.GSA_PARAMS_SLIDER, "marks"),
        Output(ids.GSA_PARAMS_SLIDER, "value"),
    ],
    Input(ids.PLOTTING_TYPE_DROPDOWN, "value"),
)
def callback_update_gsa_top_n_range(plotting_type: str) -> tuple[int, dict[int, str]]:
    """Update the GSA map top n."""

    def _calc_marks(num_params: int) -> dict[int, str]:
        return {
            0: "0",
            num_params // 2: str(num_params // 2),
            num_params: str(num_params),
        }

    if plotting_type == "map":
        num_params = 10  # limit as maps are heavy to render
        top_n = 4
    else:
        num_params = len(GSA_PARM_OPTIONS)
        top_n = 6
    return num_params, _calc_marks(num_params), top_n


########################
# UA Options Callbacks
########################


@app.callback(
    [
        Output(ids.UA_RESULTS_SECTOR_DROPDOWN, "options"),
        Output(ids.UA_RESULTS_SECTOR_DROPDOWN, "value"),
    ],
    [
        Input(ids.UA_RESULTS_TYPE_DROPDOWN, "value"),
        Input(ids.UA_RESULTS_SECTOR_DROPDOWN, "value"),
    ],
)
def callback_update_ua_results_sector_dropdown_options(
    result_type: str, existing_value: str | None
) -> list[dict[str, str]]:
    logger.debug(f"Updating UA sector dropdown options for: {result_type}")
    if result_type == "cost":
        options = SECTOR_DROPDOWN_OPTIONS_SYSTEM
    elif result_type == "marginal_cost":
        options = SECTOR_DROPDOWN_OPTIONS_SYSTEM_POWER_NG
    elif result_type == "emissions":
        options = SECTOR_DROPDOWN_OPTIONS_SYSTEM
    elif result_type == "new_capacity":
        options = SECTOR_DROPDOWN_OPTIONS_IDV
    elif result_type == "total_capacity":
        options = SECTOR_DROPDOWN_OPTIONS_IDV
    elif result_type == "generation":
        options = SECTOR_DROPDOWN_OPTIONS_IDV
    elif result_type == "demand_response":
        options = SECTOR_DROPDOWN_OPTIONS_ALL
    elif result_type == "utilization":
        options = SECTOR_DROPDOWN_OPTIONS_IDV
    elif result_type == "trade":
        options = SECTOR_DROPDOWN_OPTIONS_TRADE
    else:
        logger.debug(f"Invalid result type for UA sector dropdown: {result_type}")
        options = SECTOR_DROPDOWN_OPTIONS_ALL

    # keep existing value, else try to set to power sector, else any value
    if not existing_value:
        existing_value = "power"
    if existing_value not in [x["value"] for x in options]:
        existing_value = "power"
    if any(x["value"] == existing_value for x in options):
        value = existing_value
    else:
        value = options[0]["value"]

    return options, value


########################
# Custom Result Callbacks
########################


@app.callback(
    Output(ids.CR_PARAMETER_DROPDOWN, "disabled"),
    Input(ids.CR_ISO_DROPDOWN, "value"),
)
def callback_enable_cr_parameter_dropdown(iso_value: str) -> bool:
    """Enable CR parameter dropdown only when ISO dropdown has a value."""
    return not bool(iso_value)


@app.callback(
    Output(ids.CR_INTERVAL_SLIDER, "disabled"),
    [
        Input(ids.CR_PARAMETER_DROPDOWN, "value"),
        Input(ids.CR_RESULT_DROPDOWN, "value"),
    ],
)
def callback_enable_cr_interval_slider(
    parameter_value: str, result_value: list[str]
) -> bool:
    """Enable CR interval slider only when both parameter and result dropdowns have values."""
    return not (bool(parameter_value) and bool(result_value))


@app.callback(
    Output(ids.CR_RESULT_TYPE_DROPDOWN, "disabled"),
    Input(ids.CR_SECTOR_DROPDOWN, "value"),
)
def callback_enable_cr_result_type_dropdown(sector_value: str) -> bool:
    """Enable CR result type dropdown only when sector dropdown has a value."""
    return not bool(sector_value)


@app.callback(
    Output(ids.CR_RESULT_DROPDOWN, "disabled"),
    [
        Input(ids.CR_SECTOR_DROPDOWN, "value"),
        Input(ids.CR_RESULT_TYPE_DROPDOWN, "value"),
    ],
)
def callback_enable_cr_result_dropdown(
    sector_value: str, result_type_value: str
) -> bool:
    """Enable CR result dropdown only when both sector and result type dropdowns have values."""
    return not (bool(sector_value) and bool(result_type_value))


@app.callback(
    Output(ids.CR_RESULT_TYPE_DROPDOWN, "options"),
    Input(ids.CR_SECTOR_DROPDOWN, "value"),
)
def callback_update_cr_result_type_dropdown_options(
    sector: str,
) -> list[dict[str, str]]:
    """Filter result options based on sector."""

    if sector not in ["system", "power", "industry", "service", "transport"]:
        logger.error(f"Invalid sector: {sector}")
        return {}

    return get_cr_result_types_dropdown_options(METADATA, sector)


@app.callback(
    Output(ids.CR_RESULT_DROPDOWN, "options"),
    Input(ids.CR_SECTOR_DROPDOWN, "value"),
    Input(ids.CR_RESULT_TYPE_DROPDOWN, "value"),
)
def callback_update_cr_result_dropdown_options(
    sector: str, result_type: str
) -> list[dict[str, str]]:
    """Filter result options based on sector."""

    if not sector or not result_type:
        return {}

    return get_cr_results_dropdown_options(METADATA, sector, result_type)


@app.callback(
    Output(ids.CR_PARAMETER_DROPDOWN, "options"),
    Input(ids.CR_ISO_DROPDOWN, "value"),
)
def callback_update_cr_parameter_dropdown_options(iso: str) -> list[dict[str, str]]:
    """Update CR parameter dropdown options based on ISO."""
    if not iso:
        return {}
    logger.info(f"CR data: ISO: {iso}")
    return CR_PARAM_OPTIONS[iso]


@app.callback(
    Output(ids.CR_DATA, "data"),
    [
        Input(ids.CR_ISO_DROPDOWN, "value"),
        Input(ids.CR_PARAMETER_DROPDOWN, "value"),
        Input(ids.CR_RESULT_DROPDOWN, "value"),
        Input(ids.CR_INTERVAL_SLIDER, "value"),
    ],
)
def callback_update_cr_data(
    iso: str, parameter: str, results: list[str], interval: list[int]
) -> list[dict[str, Any]]:
    """Update CR parameter dropdown value based on ISO."""
    if not iso:
        logger.debug("ISO not provided")
        return {}
    if not results:
        logger.debug("Results not provided")
        return {}
    if not parameter:
        logger.debug("Parameter not provided")
        return {}

    df = CR_DATA[iso]
    cols = []
    if parameter not in df.columns:
        logger.error(f"Parameter {parameter} not in CR data for {iso}")
        return {}
    cols.append(parameter)
    for result in results:
        if result not in df.columns:
            logger.error(f"Result {result} not in CR data for {iso}")
        else:
            cols.append(result)

    filtered = df[cols]
    filtered = remove_ua_outliers(filtered, interval)  # ua is same as cr

    return filtered.to_dict(orient="records")


@app.callback(
    Output(ids.CR_EMISSIONS, "data"),
    [
        Input(ids.CR_ISO_DROPDOWN, "value"),
        Input(ids.CR_EMISSION_TARGET_RB, "value"),
    ],
)
def callback_update_cr_emissions(
    iso: str | None = None, emission_target: bool = True
) -> dict[str, dict[str, float]]:
    if emission_target and iso:
        return {iso: EMISSIONS[iso]}
    else:
        return {}


@app.callback(
    Output(ids.CR_EMISSION_TARGET_RB, "options"),
    Input(ids.CR_RESULT_TYPE_DROPDOWN, "value"),
)
def disable_cr_emissions_target_rb(result_type: str) -> list[dict[str, str]]:
    if result_type == "emissions":
        return [
            {"label": "True", "value": True, "disabled": False},
            {"label": "False", "value": False, "disabled": False},
        ]
    else:
        return [
            {"label": "True", "value": True, "disabled": True},
            {"label": "False", "value": False, "disabled": True},
        ]


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
