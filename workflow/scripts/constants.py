"""Constants"""

STATES_TO_EXCLUDE = [
    "AK",
    "AS",
    "GU",
    "HI",
    "MP",
    "PR",
    "TT"
]

CONVENTIONAL_CARRIERS = [
    "nuclear", 
    "oil", 
    "OCGT", 
    "CCGT", 
    "coal", 
    "geothermal", 
    "biomass", 
    "waste"
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
        "nat_gas_trade",  # constraint
        "tx",  # constraint
        "gshp",  # constraint
        "tct",  # constraint
        "discount_rate",
        "fixed_cost",
        "occ",
        "vmt_per_year",
        "efficiency2",
        "itc",
    ],
    "generators": ["tct", "rps", "discount_rate", "fixed_cost", "occ", "itc"],
    "stores": ["co2L"],
    "storage_units": ["tct", "discount_rate", "fixed_cost", "occ", "itc"],
    "lines": ["tx"],
}

CACHED_ATTRS = [
    "capital_cost",
    "discount_rate",
    "fixed_cost",
    "occ",
    "vmt_per_year",
    "lifetime",
    "itc",
]

CONSTRAINT_ATTRS = ["nat_gas_trade", "tx", "gshp", "tct", "co2L"]

VALID_RANGES = ["percent", "absolute"]

VALID_UNITS = [
    "mw",
    "percent",
    "per_unit",
    "usd",
    "usd/mw",
    "usd/mwh",
    "usd/kvmt",
    "kvmt/year",
    "kvmt/mwh",
    "years",
]

VALID_RESULTS = {
    "generators":["p_nom_opt"],
    "generators_t":["p"],
    "links":["p_nom_opt"],
    "links_t":["p0","p1","p2"],
    "buses_t":["marginal_price"],
    "system":["cost"],
    "stores":["e_nom_opt"],
    
}

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
