"""Component to display uncertainity data."""

from typing import Any
from dash import dcc, html, dash_table
import pandas as pd

from .data import (
    UA_RESULT_OPTIONS,
    SECTOR_DROPDOWN_OPTIONS,
    RESULT_TYPE_DROPDOWN_OPTIONS,
)

from .styles import DATA_TABLE_STYLE
from .utils import (
    DEFAULT_PLOTLY_THEME,
    DEFAULT_HEIGHT,
    DEFAULT_LEGEND,
    DEFAULT_OPACITY,
    get_emission_limits,
    get_ua_param_result_mapper,
    get_ua_param_sector_mapper,
    get_y_label,
    DEFAULT_2005_EMISSION_LIMIT,
    DEFAULT_2030_EMISSION_LIMIT,
)
from . import ids as ids
from .utils import _unflatten_dropdown_options
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde


import logging

logger = logging.getLogger(__name__)


def get_stores_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts out stores that will have different units.

    This is ALSO SUUUUUUUPER hacky, but this just NEEDS to get done! :|
    """
    return df[
        (df.result.str.contains("Water Heater")) | (df.result.str.contains("Battery"))
    ]


def ua_options_block() -> html.Div:
    """UA options block component."""
    return html.Div(
        [
            ua_result_type_dropdown(),
            ua_result_sector_dropdown(),
            ua_percentile_interval_slider(),
            ua_emission_target_rb(),
        ],
    )


def ua_result_type_dropdown() -> html.Div:
    """UA result type dropdown component."""
    return html.Div(
        [
            html.H6("Select Result Type"),
            dcc.Dropdown(
                id=ids.UA_RESULTS_TYPE_DROPDOWN,
                options=RESULT_TYPE_DROPDOWN_OPTIONS,
                value="cost",
            ),
        ],
    )


def ua_result_sector_dropdown() -> html.Div:
    """UA result type dropdown component."""
    return html.Div(
        [
            html.H6("Select Sector"),
            dcc.Dropdown(
                id=ids.UA_RESULTS_SECTOR_DROPDOWN,
                options=SECTOR_DROPDOWN_OPTIONS,
                value="all",
            ),
        ],
    )


def ua_percentile_interval_slider() -> html.Div:
    """GSA slider component."""

    return html.Div(
        [
            html.H6("Percentile Interval:"),
            dcc.RangeSlider(
                id=ids.UA_INTERVAL_SLIDER,
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


def ua_emission_target_rb() -> html.Div:
    """UA emission target rb component."""
    return html.Div(
        [
            html.H6("Show Emission Target"),
            dcc.RadioItems(
                id=ids.UA_EMISSION_TARGET_RB,
                options=[
                    {"label": "True", "value": True},
                    {"label": "False", "value": False},
                ],
                value=True,
                inline=True,
                className="me-3",
                labelStyle={"marginRight": "20px"},
            ),
        ],
    )


def _filter_ua_on_result_sector(
    df: pd.DataFrame, sector: str, metadata: dict
) -> pd.DataFrame:
    """Filter UA data on result sector."""

    if sector == "all":
        return df

    sector_mapper = get_ua_param_sector_mapper(metadata)

    results = ["run", "iso"]
    for result in sector_mapper:
        if result["label"] == sector:
            results.append(result["value"])
    # results = [x["value"] for x in sector_mapper if x["label"] == sector]

    cols = [x for x in results if x in df]
    return df[cols]


def _filter_ua_on_result_type(
    df: pd.DataFrame, result_type: str, metadata: dict
) -> pd.DataFrame:
    """Filter UA data on result type."""

    result_mapper = get_ua_param_result_mapper(metadata)

    cols = list(set([x["value"] for x in result_mapper if x["label"] == result_type]))

    if not cols:
        logger.debug(f"No columns found for result type {result_type}")
        return pd.DataFrame(index=df.index)
    else:
        return df.set_index("run")[[x for x in cols if x in df]].reset_index()


def remove_ua_outliers(df: pd.DataFrame, interval: list[int]) -> pd.DataFrame:
    """Replace UA outliers with NaN values."""
    # intervals defined as ints, but pandas quantile expects [0,1]
    interval_low, interval_high = (
        round(min(interval) / 100, 2),
        round(max(interval) / 100, 2),
    )
    logger.debug(
        f"Removing UA outliers with interval {interval_low} and {interval_high}"
    )

    df_out = df.copy()
    for col in df.columns:
        if col == "run":  # shouldnt really happen, but just in case
            df_out[col] = df[col]
        else:
            lower_bound = df[col].quantile(interval_low)
            upper_bound = df[col].quantile(interval_high)
            # logger.debug(
            #     f"Removing UA outliers from {col} with interval {interval_low} and {interval_high}"
            # )
            df_out[col] = df[col].where(
                (df[col] >= lower_bound) & (df[col] <= upper_bound)
            )

    return df_out


def _read_serialized_ua_data(data: dict[str, Any]) -> pd.DataFrame:
    """Read serialized UA data from the dash store."""
    df = pd.DataFrame(data)
    # logger.debug(f"Searlized UA data read: {df}")
    df["run"] = df.run.astype(int)
    return df.set_index("run")


def _apply_nice_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply nice names."""
    logger.debug("Applying nice names")
    ua_results = _unflatten_dropdown_options(UA_RESULT_OPTIONS)
    return df.rename(columns=ua_results)


def _melt_results(df: pd.DataFrame) -> pd.DataFrame:
    """Melt results from UA data table."""
    return df.melt(id_vars=["run"], var_name="result", value_name="value")


def filter_ua_on_result_sector_and_type(
    df: pd.DataFrame, result_sector: str, result_type: str, metadata: dict
) -> pd.DataFrame:
    """Filter UA data on result sector and type."""
    filtered_on_sector = _filter_ua_on_result_sector(df, result_sector, metadata)
    filtered_on_sector_and_type = _filter_ua_on_result_type(
        filtered_on_sector, result_type, metadata
    )
    return filtered_on_sector_and_type


def get_ua_data_table(
    data: dict[str, Any], nice_names: bool = True
) -> dash_table.DataTable:
    """UA data table component."""
    if not data:
        logger.debug("No UA data table data found")
        return dash_table.DataTable(
            data=[],
            columns=[],
            style_table={"overflowX": "auto"},
        )

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    # Format columns for display
    columns = [
        {"name": "run", "id": "run", "type": "numeric", "format": {"specifier": ".0f"}}
    ] + [
        {"name": col, "id": col, "type": "numeric", "format": {"specifier": ".3f"}}
        for col in df.columns
        if col != "param"
    ]

    df = df.reset_index()

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        **DATA_TABLE_STYLE,
    )


def get_ua_scatter_plot(
    data: dict[str, Any],
    nice_names: bool = True,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA scatter plot component."""

    if not data:
        logger.debug("No UA scatter plot data found")
        return px.scatter(pd.DataFrame(columns=["run", "value"]), x="run", y="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    # drop outlier data
    df_melted = df_melted.dropna(subset=["value"])

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    fig = px.scatter(
        df_melted,
        x="run",
        y="value",
        color="result",
        labels=dict(run="Run Number", value=ylabel, result="Result Type"),
    )

    fig.update_layout(
        title="",
        xaxis_title="Run Number",
        yaxis_title=ylabel,
        height=DEFAULT_HEIGHT,
        showlegend=True,
        legend=DEFAULT_LEGEND,
        template=color_theme,
    )

    # SUUUUPER hacky way to account for stores being in different units
    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        scatter_fig = px.scatter(
            df_stores,
            x="run",
            y="value",
            color="result",
            labels=dict(run="Run Number", value=y_label_store, result="Result Type"),
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        # so the stores are clearly different
        scatter_fig.update_traces(marker_symbol="x")

        for trace in scatter_fig.data:
            trace.yaxis = "y2"
            fig.add_trace(trace)

        fig.update_layout(
            yaxis2=dict(
                title=y_label_store,
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

    if emissions:
        emissions_2005, emissions_2030 = get_emission_limits(emissions)

        fig.add_hline(
            y=emissions_2005,
            **DEFAULT_2005_EMISSION_LIMIT,
        )

        fig.add_hline(
            y=emissions_2030,
            **DEFAULT_2030_EMISSION_LIMIT,
        )

    return fig


def get_ua_barchart(
    data: dict[str, Any],
    nice_names: bool = True,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA barchart component showing mean values with 95% confidence intervals for each result category."""
    if not data:
        logger.debug("No UA barchart data found")
        return px.bar(pd.DataFrame(columns=["result", "value"]), x="result", y="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    df_melted = df_melted.dropna(subset=["value"])

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    # Calculate mean and std for each result category
    stats_df = (
        df_melted.groupby("result")
        .agg(value=("value", "mean"), std=("value", "std"))
        .reset_index()
    )

    fig = px.bar(
        stats_df,
        x="result",
        y="value",
        error_y="std",
        labels=dict(result="", value=ylabel),
        opacity=DEFAULT_OPACITY,
    )

    # Update layout
    fig.update_layout(
        title="(Mean ± 95% CI)",
        showlegend=False,
        height=DEFAULT_HEIGHT,
        bargap=0.2,
        xaxis=dict(tickangle=45),
        template=color_theme,
    )

    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        stats_df_stores = (
            df_stores.groupby("result")
            .agg(value=("value", "mean"), std=("value", "std"))
            .reset_index()
        )

        bar_fig = px.bar(
            stats_df_stores,
            x="result",
            y="value",
            error_y="std",
            labels=dict(result="", value=y_label_store),
            opacity=DEFAULT_OPACITY,
        )

        # vertical line to separate MWh from MW data
        fig.add_vline(
            x=len(stats_df) - 0.5,  # Position before the stores data
            line_dash="solid",
            line_color="black",
            line_width=2,
            opacity=0.8,
        )

        for trace in bar_fig.data:
            trace.yaxis = "y2"
            fig.add_trace(trace)

        fig.update_layout(
            yaxis2=dict(
                title=y_label_store,
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

    if emissions:
        emissions_2005, emissions_2030 = get_emission_limits(emissions)

        fig.add_hline(
            y=emissions_2005,
            **DEFAULT_2005_EMISSION_LIMIT,
        )

        fig.add_hline(
            y=emissions_2030,
            **DEFAULT_2030_EMISSION_LIMIT,
        )

    return fig


def get_ua_histogram(
    data: dict[str, Any], nice_names: bool = True, **kwargs
) -> html.Div:
    """UA histogram component with overlaid probability density functions"""
    if not data:
        logger.debug("No UA histogram data found")
        return html.Div(
            [dcc.Graph(figure=px.histogram(pd.DataFrame(columns=["value"]), x="value"))]
        )

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    ylabel = kwargs.get("result_type", None)

    result_types = df_melted["result"].unique()
    logger.debug(f"Historgram result types: {result_types}")
    fig_height = (
        int(DEFAULT_HEIGHT * (2 / 3)) if len(result_types) > 1 else DEFAULT_HEIGHT
    )

    figures = []

    for result_type in result_types:
        result_data = df_melted[df_melted["result"] == result_type]

        # Calculate number of bins using Freedman-Diaconis rule

        fig = px.histogram(
            result_data,
            x="value",
            labels=dict(value=ylabel),
            opacity=DEFAULT_OPACITY,
            title=f"{result_type}",
        )

        fig.update_layout(
            yaxis=dict(title="Count"),
            yaxis2=dict(
                # title="Probability Density",
                overlaying="y",
                side="right",
                showgrid=False,
                showticklabels=False,
            ),
        )

        # Add PDF overlay
        values = result_data["value"].dropna()

        # Check if we have enough data and variance for KDE
        # this will not be the case if all values are the same for all model runs
        if len(values) > 1 and values.var() > 1e-10:
            try:
                kde = gaussian_kde(values)
                x_range = np.linspace(
                    result_data["value"].min(), result_data["value"].max(), 100
                )
                y_range = kde(x_range)

                # Verify that the KDE integrates to approximately 1
                dx = x_range[1] - x_range[0]
                area = np.sum(y_range * dx)
                logger.debug(f"KDE area under curve with dx={dx} is: {area:.6f}")

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        name="PDF",
                        line=dict(width=2),
                        showlegend=True,
                        yaxis="y2",
                    )
                )
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.warning(f"Could not compute KDE for {result_type}: {e}")
        else:
            logger.debug(f"Insufficient variance in data for KDE in {result_type}")

        fig.update_layout(
            height=fig_height,
            showlegend=True,
            legend=DEFAULT_LEGEND,
            template=color_theme,
            margin=dict(t=50, l=50, r=50, b=50),
        )

        figures.append(dcc.Graph(figure=fig))

    if len(result_types) > 1:
        return html.Div(
            figures,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "20px",
                "padding": "20px",
            },
            id=ids.UA_HISTOGRAM,
        )
    else:
        return html.Div(
            figures[0],
            id=ids.UA_HISTOGRAM,
        )


def get_ua_violin_plot(
    data: dict[str, Any],
    nice_names: bool = True,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA violin plot component."""
    if not data:
        logger.debug("No UA violin plot data found")
        return px.violin(
            pd.DataFrame(columns=["result", "value"]), x="result", y="value"
        )

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    fig = px.violin(
        df_melted,
        x="result",
        y="value",
        labels=dict(result="", value=ylabel),
    )

    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title=ylabel,
        height=DEFAULT_HEIGHT,
        showlegend=True,
        legend=DEFAULT_LEGEND,
        template=color_theme,
    )

    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        violin_fig = px.violin(
            df_stores,
            x="result",
            y="value",
            labels=dict(result="", value=y_label_store),
        )

        # vertical line to separate MWh from MW data
        fig.add_vline(
            x=len(df_melted["result"].unique())
            - 0.5,  # Position before the stores data
            line_dash="solid",
            line_color="black",
            line_width=2,
            opacity=0.8,
        )

        for trace in violin_fig.data:
            trace.yaxis = "y2"
            fig.add_trace(trace)

        fig.update_layout(
            yaxis2=dict(
                title=y_label_store,
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

    if emissions:
        emissions_2005, emissions_2030 = get_emission_limits(emissions)

        fig.add_hline(
            y=emissions_2005,
            **DEFAULT_2005_EMISSION_LIMIT,
        )

        fig.add_hline(
            y=emissions_2030,
            **DEFAULT_2030_EMISSION_LIMIT,
        )

    return fig


def get_ua_box_whisker(
    data: dict[str, Any],
    nice_names: bool = True,
    emissions: list[dict[str, Any]] | None = None,
    **kwargs,
) -> go.Figure:
    """UA box whisker plot component."""
    if not data:
        logger.debug("No UA box whisker plot data found")
        return px.box(pd.DataFrame(columns=["result", "value"]), x="result", y="value")

    df = _read_serialized_ua_data(data)

    if nice_names:
        df = _apply_nice_names(df)

    df = df.reset_index()
    df_melted = _melt_results(df)

    color_theme = kwargs.get("template", DEFAULT_PLOTLY_THEME)
    result_type = kwargs.get("result_type", None)
    ylabel = get_y_label(df_melted, result_type)

    df_stores = get_stores_df(df_melted)
    if not df_stores.empty:
        df_melted = df_melted[~df_melted.result.isin(df_stores.result)].copy()

    fig = px.box(
        df_melted,
        x="result",
        y="value",
        labels=dict(result="", value=ylabel),
    )

    # Update layout
    fig.update_layout(
        title="",
        showlegend=False,
        height=DEFAULT_HEIGHT,
        xaxis=dict(tickangle=45),
        template=color_theme,
    )

    if not df_stores.empty:
        y_label_store = ylabel.replace("(MW)", "(MWh)")

        box_fig = px.box(
            df_stores,
            x="result",
            y="value",
            labels=dict(result="", value=y_label_store),
        )

        # vertical line to separate MWh from MW data
        fig.add_vline(
            x=len(df_melted["result"].unique())
            - 0.5,  # Position before the stores data
            line_dash="solid",
            line_color="black",
            line_width=2,
            opacity=0.8,
        )

        for trace in box_fig.data:
            trace.yaxis = "y2"
            fig.add_trace(trace)

        fig.update_layout(
            yaxis2=dict(
                title=y_label_store,
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

    if emissions:
        emissions_2005, emissions_2030 = get_emission_limits(emissions)

        fig.add_hline(
            y=emissions_2005,
            **DEFAULT_2005_EMISSION_LIMIT,
        )

        fig.add_hline(
            y=emissions_2030,
            **DEFAULT_2030_EMISSION_LIMIT,
        )

    return fig
