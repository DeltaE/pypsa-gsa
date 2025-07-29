"""Constants"""

# energy conversion of MWh to MMCF
NG_MWH_2_MMCF = 305

# 3.412 MMBTU = 1 MWH
MMBTU_2_MWH = 3.412

GSA_COLUMNS = [
    "name",
    "group",
    "nice_name",
    "component",
    "carrier",
    "attribute",
    "range",
    "unit",
    "min_value",
    "max_value",
    "source",
    "notes",
]

CONVENTIONAL_CARRIERS = [
    "nuclear",
    "oil",
    "OCGT",
    "CCGT",
    "coal",
    "geothermal",
    "biomass",
    "waste",
]

RPS_CARRIERS = [
    "onwind",
    "offwind",
    "offwind_floating",
    "solar",
    "hydro",
    "geothermal",
    "biomass",
    "EGS",
]

CES_CARRIERS = [
    "onwind",
    "offwind",
    "offwind_floating",
    "solar",
    "hydro",
    "geothermal",
    "EGS",
    "biomass",
    "nuclear",
]

ADDITIONAL_VALID_ATTRIBUTES = {
    "links": [
        "nat_gas_import",  # constraint
        "nat_gas_export",  # constraint
        "lv",  # constraint
        "gshp",  # constraint
        "tct",  # constraint
        "ev_policy",  # constraint
        "discount_rate",
        "fixed_cost",
        "occ",
        "vmt_per_year",
        "itc",
        "leakage",
        "gwp",
        "elec_trade",  # constraint
    ],
    "generators": [
        "tct",
        "discount_rate",
        "fixed_cost",
        "occ",
        "itc",
        "rps",
        "ces",
        "landuse",
    ],
    "stores": ["co2L"],
    "storage_units": ["tct", "discount_rate", "fixed_cost", "occ", "itc"],
    "lines": [],
}

CACHED_ATTRS = [
    "capital_cost",
    "discount_rate",
    "fixed_cost",
    "occ",
    "vmt_per_year",
    "lifetime",
    "itc",
    "gwp",
    "leakage",
]

CONSTRAINT_ATTRS = [
    "nat_gas_import",
    "nat_gas_export",
    "lv",
    "gshp",
    "tct",
    "co2L",
    "ev_policy",
    "rps",
    "ces",
    "elec_trade",
]

VALID_RANGES = ["percent", "absolute"]

VALID_UNITS = [
    "mw",
    "percent",
    "per_unit",
    "usd",
    "usd/mw",
    "usd/mwh",
    "usd/kvmt",
    "usd/T",
    "kvmt/year",
    "kvmt/mwh",
    "years",
    "mmt",
]

VALID_RESULTS = {
    "generators": ["p_nom_opt", "p_nom_new"],
    "generators_t": ["p"],
    "links": ["p_nom_opt", "p_nom_new"],
    "links_t": ["p0", "p1", "p2"],
    "buses_t": ["marginal_price"],
    "system": ["cost"],
    "stores": ["e_nom_opt"],
}

VALID_UA_PLOTS = ["scatter", "bar"]

# hard codes where gas can enter/exit the states
# if multiple POEs exist, the larger pipeline is used as the POE
# https://atlas.eia.gov/datasets/eia::border-crossings-natural-gas/explore?location=48.411182%2C-90.296487%2C5.24
POINTS_OF_ENTRY = {
    "AZ": "MX",  # Arizona - Mexico
    "CA": "MX",  # California - Mexico
    "ID": "BC",  # Idaho - BC
    "ME": "NB",  # Maine - New Brunswick
    "MI": "ON",  # Michigan - Ontario
    "MN": "MB",  # Minnesota - Manitoba
    "MT": "SK",  # Montana - Saskatchewan
    "ND": "SK",  # North Dakota - Saskatchewan
    "NH": "QC",  # New Hampshire - Quebec
    "NY": "ON",  # New York - Ontario
    "TX": "MX",  # Texas - Mexico
    "VT": "QC",  # Vermont - Mexico
    "WA": "BC",  # Washington - BC
}

STATES_TO_EXCLUDE = ["AK", "AS", "GU", "HI", "MP", "PR", "TT"]

STATE_2_CODE = {
    # United States
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "American Samoa": "AS",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Guam": "GU",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Northern Mariana Islands": "MP",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Puerto Rico": "PR",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Trust Territories": "TT",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Virgin Islands": "VI",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    # Canada
    "Alberta": "AB",
    "British Columbia": "BC",
    "Manitoba": "MB",
    "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Northwest Territories": "NT",
    "Nova Scotia": "NS",
    "Nunavut": "NU",
    "Ontario": "ON",
    "Prince Edward Island": "PE",
    "Quebec": "QC",
    "Saskatchewan": "SK",
    "Yukon": "YT",
    # Mexico
    "Mexico": "MX",
}

# See ./dashboard/components/shared.py for same mappings.
ISO_STATES = {
    "caiso": ["CA"],
    "ercot": ["TX"],
    "isone": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "miso": ["AR", "IL", "IN", "IA", "LA", "MI", "MN", "MO", "MS", "WI"],
    "nyiso": ["NY"],
    "pjm": ["DE", "KY", "MD", "NJ", "OH", "PA", "VA", "WV"],
    "spp": ["KS", "ND", "NE", "OK", "SD"],
    "northwest": ["ID", "MT", "OR", "WA", "WY"],
    "southeast": ["AL", "FL", "GA", "NC", "SC", "TN"],
    "southwest": ["AZ", "CO", "NM", "NV", "UT"],
    "mexico": ["MX"],
    "canada": ["BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "NL", "NFI", "PEI"],
}

# manually mapped to match EIA regions to ISOs
REGION_2_ISO = {
    "California": "caiso",
    "Canada": "canada",
    "Carolinas": "southeast",
    "Central": "spp",
    "Florida": "southeast",
    "Mexico": "mexico",
    "Mid-Atlantic": "pjm",
    "Midwest": "miso",
    "New England": "isone",
    "New York": "nyiso",
    "Northwest": "northwest",
    "Southeast": "southeast",
    "Southwest": "southwest",
    "Tennessee": "southeast",
    "Texas": "ercot",
}
