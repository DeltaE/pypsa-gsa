"""ID Names."""

# top level
NAVBAR = "navbar"

# tabs
TABS = "tab"
TAB_CONTENT = "tab-content"
DATA_TAB = "data-tab"
SA_TAB = "sa-tab"
UA_TAB = "ua-tab"
CR_TAB = "cr-tab"

# dropdowns
STATE_DROPDOWN = "state-dropdown"
GSA_PARAM_DROPDOWN = "gsa-param-dropdown"
GSA_RESULTS_DROPDOWN = "gsa-results-dropdown"
UA_PARAM_DROPDOWN = "ua-param-dropdown"
UA_RESULTS_DROPDOWN = "ua-results-dropdown"
UA_RESULTS_TYPE_DROPDOWN = "ua-results-type-dropdown"
UA_RESULTS_SECTOR_DROPDOWN = "ua-results-sector-dropdown"
CR_SECTOR_DROPDOWN = "cr-sector-dropdown"
CR_STATE_DROPDOWN = "cr-state-dropdown"
CR_PARAMETER_DROPDOWN = "cr-parameter-dropdown"
CR_RESULT_DROPDOWN = "cr-result-dropdown"
CR_RESULT_TYPE_DROPDOWN = "cr-result-type-dropdown"
CR_INTERVAL_SLIDER = "cr-interval-slider"
COLOR_DROPDOWN = "color-scale-dropdown"
PLOTTING_TYPE_DROPDOWN = "plotting-type-dropdown"
INPUT_DATA_ATTRIBUTE_DROPDOWN = "inputs-attribute-dropdown"
INPUT_DATA_SECTOR_DROPDOWN = "inputs-sector-dropdown"

# buttons
GSA_PARAM_SELECT_ALL = "gsa-param-select-all"
GSA_PARAM_REMOVE_ALL = "gsa-param-remove-all"
GSA_RESULTS_SELECT_ALL = "gsa-results-select-all"
GSA_RESULTS_REMOVE_ALL = "gsa-results-remove-all"
INPUT_DATA_REMOVE_FILTERS = "input-data-remove-filters"
STATES_SELECT_ALL = "states-select-all"
STATES_REMOVE_ALL = "states-remove-all"

# button states
GSA_PARAM_BUTTON_STATE = "gsa-param-button-state"
GSA_RESULTS_BUTTON_STATE = "gsa-results-button-state"

# radio buttons
GSA_PARAM_SELECTION_RB = "gsa-param-selection-rb"
UA_EMISSION_TARGET_RB = "ua-emission-target-rb"
CR_EMISSION_TARGET_RB = "cr-emission-target-rb"

# sliders
GSA_PARAMS_SLIDER = "gsa-range-slider"
UA_INTERVAL_SLIDER = "ua-interval-slider"

# collapsable blocks
PLOTTING_OPTIONS_BLOCK = "plotting-options-block"
STATE_OPTIONS_BLOCK = "state-options-block"
GSA_OPTIONS_BLOCK = "gsa-options-block"
GSA_PARAMS_RESULTS_COLLAPSE = "gsa-params-results-collapse"
GSA_PARAMS_SLIDER_COLLAPSE = "gsa-range-slider-collapse"
UA_OPTIONS_BLOCK = "ua-options-block"
INPUT_DATA_OPTIONS_BLOCK = "input-data-options-block"
CR_OPTIONS_BLOCK = "cr-options-block"

# gsa stores
GSA_STATE_DATA = "gsa-state-data"  # other stores use this
GSA_STATE_NORMED_DATA = "gsa-state-normed-data"
GSA_HM_DATA = "gsa-hm-data"
GSA_BAR_DATA = "gsa-bar-data"
GSA_MAP_DATA = "gsa-map-data"
GSA_NORMED = "gsa-normed"  # for mu/mu_max clac
GSA_DATA_TABLE_DATA = "gsa-data-table-data"

UA_STATE_DATA = "ua-state-data"  # filtered by STATE
UA_RUN_DATA = "ua-run-data"  # filtered by result

CR_DATA = "cr-data"  # dict of dfs

INPUTS_DATA = "inputs-data"
INPUTS_DATA_BY_ATTRIBUTE = "inputs-data-by-attribute"
INPUTS_DATA_BY_ATTRIBUTE_CARRIER = "inputs-data-by-attribute-carrier"

# data tables (filtered for params/results)
UA_DATA_TABLE = "ua-data-table"
GSA_NORMED_DATA_TABLE = "gsa-normed-data-table"
GSA_RUNS_DATA_TABLE = "gsa-runs-data-table"
INPUTS_DATA_TABLE = "inputs-data-table"
CR_DATA_TABLE = "cr-data-table"

# input data plots
INPUT_DATA_BAR_CHART = "input-data-bar-chart"

# gsa plots
GSA_HEATMAP = "gsa-heatmap"
GSA_BAR_CHART = "gsa-bar-chart"
GSA_MAP = "gsa-map"
GSA_DATA_TABLE = "gsa-data-table"

# ua plots
UA_BAR_CHART = "ua-bar-chart"
UA_SCATTER = "ua-scatter"
UA_VIOLIN = "ua-violin"
UA_HISTOGRAM = "ua-histogram"
UA_BOX_WHISKER = "ua-box-whisker"

# custom result plots
CR_SCATTER = "cr-scatter"
CR_HISTOGRAM = "cr-histogram"

# for drawing emission limit lines on charts
UA_EMISSIONS = "ua-emissions"
CR_EMISSIONS = "cr-emissions"
