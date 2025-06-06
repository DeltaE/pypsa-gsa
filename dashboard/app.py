from pathlib import Path
from typing import Any
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd

from components.utils import (
    DEFAULT_PLOTLY_THEME,
    get_continuous_color_scale_options,
    get_discrete_color_scale_options,
    get_gsa_params_dropdown_options,
    get_gsa_results_dropdown_options,
    DEFAULT_CONTINOUS_COLOR_SCALE,
    DEFAULT_DISCRETE_COLOR_SCALE,
    get_plotly_plotting_themes,
    get_ua_results_dropdown_options,
)
import components.ids as ids
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
    SECTOR_DROPDOWN_OPTIONS_ALL,
    SECTOR_DROPDOWN_OPTIONS_IDV,
    SECTOR_DROPDOWN_OPTIONS,
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
from components.input_data import get_inputs_data_table

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

RAW_GSA = pd.read_csv("data/sa.csv")
RAW_UA = pd.read_csv("data/ua_runs.csv")
RAW_PARAMS = pd.read_csv("data/parameters.csv")
ISO_SHAPE = gpd.read_file("data/iso.geojson")

root = Path(__file__).parent
GSA_PARM_OPTIONS = get_gsa_params_dropdown_options(root)
GSA_RESULT_OPTIONS = get_gsa_results_dropdown_options(root)
UA_RESULT_OPTIONS = get_ua_results_dropdown_options(root)

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
                dcc.Store(id=ids.INPUTS_DATA),
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
        Input(ids.INPUTS_DATA, "data"),
        Input(ids.PLOTTING_TYPE_DROPDOWN, "value"),
        Input(ids.COLOR_DROPDOWN, "value"),
    ],
    State(ids.UA_RESULTS_TYPE_DROPDOWN, "value"),
)
def render_tab_content(
    active_tab: str,
    gsa_hm_data: list[dict[str, Any]] | None,
    gsa_bar_data: list[dict[str, Any]] | None,
    gsa_map_data: list[dict[str, Any]] | None,
    ua_run_data: list[dict[str, Any]] | None,
    inputs_data: list[dict[str, Any]] | None,
    plotting_type: str,
    color: str,
    ua_result_type: str,
) -> html.Div:
    logger.debug(f"Rendering tab content for: {active_tab}")
    if active_tab == ids.DATA_TAB:
        view = get_inputs_data_table(inputs_data)
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
                    ua_run_data, template=color, result_type=ua_result_type
                ),
            )
        elif plotting_type == "violin":
            view = dcc.Graph(
                id=ids.UA_VIOLIN,
                figure=get_ua_violin_plot(
                    ua_run_data, template=color, result_type=ua_result_type
                ),
            )
        elif plotting_type == "scatter":
            view = dcc.Graph(
                id=ids.UA_SCATTER,
                figure=get_ua_scatter_plot(
                    ua_run_data, template=color, result_type=ua_result_type
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
                    ua_run_data, template=color, result_type=ua_result_type
                ),
            )
        else:
            return html.Div([dbc.Alert("No plotting type selected", color="info")])
        return html.Div([dbc.Card([dbc.CardBody([view])])])
    else:
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
        Output(ids.GSA_OPTIONS_BLOCK, "style"),
        Output(ids.UA_OPTIONS_BLOCK, "style"),
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
    iso_style = hidden
    gsa_style = hidden
    ua_style = hidden

    plotting_style = visible

    if active_tab == ids.SA_TAB:
        iso_style = visible
        gsa_style = visible
    elif active_tab == ids.UA_TAB:
        iso_style = visible
        ua_style = visible

    return plotting_style, iso_style, gsa_style, ua_style


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
    df = filter_ua_on_result_sector_and_type(df, result_sector, result_type)
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
    if active_tab == ids.UA_TAB:
        return DEFAULT_PLOTLY_THEME, get_plotly_plotting_themes()
    elif active_tab == ids.SA_TAB:
        if plotting_type in ["heatmap"]:
            logger.debug("Updating continuous color options")
            return DEFAULT_CONTINOUS_COLOR_SCALE, get_continuous_color_scale_options()
        else:
            return DEFAULT_DISCRETE_COLOR_SCALE, get_discrete_color_scale_options()
    else:
        return "", []


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
    Input(ids.UA_RESULTS_TYPE_DROPDOWN, "value"),
)
def callback_update_ua_results_sector_dropdown_options(
    result_type: str,
) -> list[dict[str, str]]:
    logger.debug(f"Updating UA sector dropdown options for: {result_type}")
    if result_type == "costs":
        options = SECTOR_DROPDOWN_OPTIONS_ALL
    elif result_type == "marginal_costs":
        options = SECTOR_DROPDOWN_OPTIONS
    elif result_type == "emissions":
        options = SECTOR_DROPDOWN_OPTIONS_ALL
    elif result_type == "new_capacity":
        options = SECTOR_DROPDOWN_OPTIONS_IDV
    elif result_type == "total_capacity":
        options = SECTOR_DROPDOWN_OPTIONS_IDV
    elif result_type == "generation":
        options = SECTOR_DROPDOWN_OPTIONS_IDV
    else:
        logger.debug(f"Invalid result type for UA sector dropdown: {result_type}")
        options = SECTOR_DROPDOWN_OPTIONS_ALL
    return options, options[0]["value"]


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
