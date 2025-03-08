"""Constants"""

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
        "constraint",
        "discount_rate",
        "fixed_cost",
        "occ",
        "vmt_per_year",
        "efficiency2",
    ],
    "generators": ["constraint", "discount_rate", "fixed_cost", "occ"],
    "stores": ["constraint"],
    "storage_units": ["constraint", "discount_rate", "fixed_cost", "occ"],
    "lines": ["constraint"],
}

VALID_RANGES = ["percent", "absolute"]

VALID_UNITS = [
    "mw",
    "percent",
    "per_unit",
    "usd",
    "usd/mw",
    "usd/mwh",
    "usd/vmt",
    "vmt/year",
    "vmt/mwh",
    "years",
]

