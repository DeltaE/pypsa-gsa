"""Styles for the dashboard."""

BUTTON_STYLE = dict(
    style={
        "marginTop": "10px",
        "marginBottom": "5px",
        "marginRight": "10px",
    },
    size="sm",
    color="secondary",
    outline=True,
)

DATA_TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={
        "textAlign": "left",
        "padding": "10px",
        "whiteSpace": "normal",
        "height": "auto",
    },
    style_header={
        "backgroundColor": "rgb(230, 230, 230)",
        "fontWeight": "bold",
        "border": "1px solid black",
    },
    style_data={"border": "1px solid lightgrey"},
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"}
    ],
    page_size=50,
    sort_action="native",
    filter_action="native",
    sort_mode="multi",
    export_format="csv",
    export_headers="display",
)
