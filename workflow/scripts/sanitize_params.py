"""Sanitizes parameters file.

Ensures unit alignment and basic validation checks."""

import pandas as pd
import pypsa
from constants import (
    ADDITIONAL_VALID_ATTRIBUTES,
    VALID_RANGES,
    VALID_UNITS,
    CONSTRAINT_ATTRS,
)

###
# Sanitize names
###


def is_valid_attributes(
    params: pd.DataFrame, n: pypsa.Network | None = None
) -> bool:
    """Confirm attributes are valid.

    Assumes component names are valid.
    """

    def _check_attribute(c: str, attr: str) -> None:
        if c == "links":
            valid = attr in VALID_LINK_ATTRS
        elif c == "links_t":
            valid = attr in VALID_LINK_T_ATTRS
        elif c == "lines":
            valid = attr in VALID_LINE_ATTRS
        elif c == "lines_t":
            valid = attr in VALID_LINE_T_ATTRS
        elif c == "generators":
            valid = attr in VALID_GENERATOR_ATTRS
        elif c == "generators_t":
            valid = attr in VALID_GENERATOR_T_ATTRS
        elif c == "stores":
            valid = attr in VALID_STORE_ATTRS
        elif c == "stores_t":
            valid = attr in VALID_STORE_T_ATTRS
        elif c == "storage_units":
            valid = attr in VALID_STORAGEUNIT_ATTRS
        elif c == "storage_units_t":
            valid = attr in VALID_STORAGEUNIT_T_ATTRS
        elif c == "loads":
            valid = attr in VALID_LOAD_ATTRS
        elif c == "loads_t":
            valid = attr in VALID_LOAD_T_ATTRS
        else:
            raise ValueError(f"Component {c} not valid")
        
        if not valid:
            try:
                valid = attr in ADDITIONAL_VALID_ATTRIBUTES[c]
            except KeyError:
                valid = False

        if not valid:
            raise ValueError(f"Attribute of {attr} for component {c} is not valid")

    if not n:
        n = pypsa.Network()

    VALID_LINK_ATTRS = n.component_attrs["Link"].index
    VALID_LINK_T_ATTRS = list(n.pnl("Link").keys())
    VALID_LINE_ATTRS = n.component_attrs["Line"].index
    VALID_LINE_T_ATTRS = list(n.pnl("Line").keys())
    VALID_GENERATOR_ATTRS = n.component_attrs["Generator"].index
    VALID_GENERATOR_T_ATTRS = list(n.pnl("Generator").keys())
    VALID_STORE_ATTRS = n.component_attrs["Store"].index
    VALID_STORE_T_ATTRS = list(n.pnl("Store").keys())
    VALID_LOAD_ATTRS = n.component_attrs["Load"].index
    VALID_LOAD_T_ATTRS = list(n.pnl("Load").keys())
    VALID_STORAGEUNIT_ATTRS = n.component_attrs["StorageUnit"].index
    VALID_STORAGEUNIT_T_ATTRS = list(n.pnl("StorageUnit").keys())

    df = params.copy()
    df.apply(
        lambda row: _check_attribute(row["component"], row["attribute"]), axis=1
    )
    return True


def sanitize_component_name(params: pd.DataFrame) -> pd.DataFrame:
    """Confirm components are valid."""

    def _sanitize_component_name(c: str) -> str:
        match c.lower():
            case "link" | "links":
                return "links"
            case "generator" | "generators":
                return "generators"
            case "store" | "stores":
                return "stores"
            case "storageunit" | "storageunits":
                return "storage_units"
            case "store" | "stores":
                return "stores"
            case "links_t" | "link_t":
                return "links_t"
            case "loads" | "load":
                return "loads"
            case "loads_t" | "load_t":
                return "loads_t"
            case "line" | "lines":
                return "lines"
            case "lines_t" | "line_t":
                return "lines_t"
            case "bus" | "buses":
                return "buses"
            case "bus_t" | "buses_t":
                return "buses_t"
            case "system" | "network": # results processing
                return "system"
            case _:
                raise KeyError(c)

    df = params.copy()
    df["component"] = df.component.map(lambda x: _sanitize_component_name(x))
    return df


###
# Sanitize units
###

def correct_usd(params: pd.DataFrame) -> pd.DataFrame:
    """Ensures all 'usd' references are lowercase"""
    
    df = params.copy()
    df["unit"] = df["unit"].str.replace("USD", "usd")
    return df

def correct_water_heater_units(params: pd.DataFrame) -> pd.DataFrame:
    """Takes same assumptions from PyPSA-USA

    USD/gal -> USD/MWh (water storage)
      assume cp = 4.186 kJ/kg/C
      (USD / gal) * (1 gal / 3.75 liter) * (1L / 1 kg H2O) = 0.267 USD / kg water
      (0.267 USD / kg) * (1 / 4.186 kJ/kg/C) * (1 / 1C) = 0.0637 USD / kJ
      (0.0637 USD / kJ) * (1000 kJ / 1 MJ) * (3600sec / 1hr) = 229335 USD / MWh
    """
    df = params.copy()
    df.loc[
        ((df.unit == "usd/gal") & (df.carrier.str.startswith(("res", "com")))),
        "min_value",
    ] *= 229335
    df.loc[
        ((df.unit == "usd/gal") & (df.carrier.str.startswith(("res", "com")))),
        "max_value",
    ] *= 229335
    df.loc[
        ((df.unit == "usd/gal") & (df.carrier.str.startswith(("res", "com")))), "unit"
    ] = "usd/mwh"
    return df


def correct_kBtu_units(params: pd.DataFrame) -> pd.DataFrame:
    """Takes same assumptions from PyPSA-USA

    MMBTU_MWHthemal = 3.4129  # MMBTU to MWh_thermal
    """

    mmbtu_2_mwh = 3.4129

    df = params.copy()
    df.loc[
        (
            (df.unit.str.contains("/kBtu"))
            & (df.attribute.isin(("occ", "capital_cost", "fixed_cost", "constraint")))
        ),
        "min_value",
    ] *= 1000 * mmbtu_2_mwh
    df.loc[
        (
            (df.unit.str.contains("/kBtu"))
            & (df.attribute.isin(("occ", "capital_cost", "fixed_cost", "constraint")))
        ),
        "max_value",
    ] *= 1000 * mmbtu_2_mwh
    df.loc[
        (
            (df.unit.str.contains("/kBtu"))
            & (df.attribute.isin(("occ", "capital_cost", "fixed_cost", "constraint")))
        ),
        "unit",
    ] = "usd/mw"

    return df


def correct_mpge_units(params: pd.DataFrame) -> pd.DataFrame:
    """Takes same assumptions from PyPSA-USA

    Assumption from https://www.nrel.gov/docs/fy18osti/70485.pdf
    wh_per_gallon = 33700  # footnote 24
    """

    wh_per_gallon = 33700

    df = params.copy()
    df.loc[(df.unit == "mpge"), "min_value"] *= 1e6 / wh_per_gallon
    df.loc[(df.unit == "mpge"), "max_value"] *= 1e6 / wh_per_gallon
    df.loc[(df.unit == "mpge"), "unit"] = "vmt/mwh"

    return df

def correct_miles(params: pd.DataFrame) -> pd.DataFrame:
    """Ensures all miles are labeled as 'vmt'."""
    
    df = params.copy()
    df["unit"] = df["unit"].str.replace("miles", "vmt")
    return df

def correct_vmt_units(params: pd.DataFrame) -> pd.DataFrame:
    """Convert vmt units to kvmt."""

    df = params.copy()
    df.loc[(df.unit.str.startswith("vmt")), "min_value"] *= 1e-3
    df.loc[(df.unit.str.startswith("vmt")), "max_value"] *= 1e-3

    df.loc[(df.unit.str.endswith("/vmt")), "min_value"] *= 1e3
    df.loc[(df.unit.str.endswith("/vmt")), "max_value"] *= 1e3

    df.unit = df.unit.str.replace("vmt", "kvmt")

    return df

def correct_mmbtu_units(params: pd.DataFrame) -> pd.DataFrame:
    """Takes same assumptions from PyPSA-USA

    MMBTU_MWHthemal = 3.4129  # MMBTU to MWh_thermal
    """

    mmbtu_2_mwh = 3.4129

    df = params.copy()
    df.loc[
        (
            (df.unit.str.contains("mmbtu/mwh"))
            & (df.attribute.isin(("efficiency", "efficiency2")))
        ),
        "min_value",
    ] *= mmbtu_2_mwh
    df.loc[
        (
            (df.unit.str.contains("mmbtu/mwh"))
            & (df.attribute.isin(("efficiency", "efficiency2")))
        ),
        "max_value",
    ] *= mmbtu_2_mwh
    df.loc[
        (
            (df.unit.str.contains("mmbtu/mwh"))
            & (df.attribute.isin(("efficiency", "efficiency2")))
        ),
        "unit",
    ] = "per_unit"

    return df


def correct_tonnes_units(params: pd.DataFrame) -> pd.DataFrame:
    """Converts tonens to mmt"""
    df = params.copy()
    df.loc[(df.unit == "tonnes") & (df.attribute == "co2"), "min_value"] *= 1 / 1e6
    df.loc[(df.unit == "tonnes") & (df.attribute == "co2"), "max_value"] *= 1 / 1e6
    df.loc[(df.unit == "tonnes") & (df.attribute == "co2"), "unit"] = "mmt"
    return df


def correct_kw_units(params: pd.DataFrame) -> pd.DataFrame:
    """Converts kw to mw"""
    df = params.copy()
    df.loc[
        (df.unit.str.endswith("/kw"))
        & (df.attribute.isin(("occ", "capital_cost", "fixed_cost"))),
        "min_value",
    ] *= 1e3
    df.loc[
        (df.unit.str.endswith("/kw"))
        & (df.attribute.isin(("occ", "capital_cost", "fixed_cost"))),
        "max_value",
    ] *= 1e3
    df["unit"] = df.unit.str.replace("/kw", "/mw")
    return df


def correct_percent_units(params: pd.DataFrame) -> pd.DataFrame:
    """Converts any absolute values given in percent to per_unit."""

    df = params.copy()
    df.loc[(df.range == "absolute") & (df.unit == "percent"), "min_value"] *= 1 / 1e2
    df.loc[(df.range == "absolute") & (df.unit == "percent"), "max_value"] *= 1 / 1e2
    df.loc[(df.range == "absolute") & (df.unit == "percent"), "unit"] = "per_unit"
    return df

def strip_whitespace(params: pd.DataFrame) -> pd.DataFrame:
    """Strips any leading/trailing whitespace from naming columns."""
    
    df = params.copy()
    df["name"] = df.name.str.strip()
    df["group"] = df.group.str.strip()
    df["nice_name"] = df.nice_name.str.strip()
    df["component"] = df.component.str.strip()
    df["attribute"] = df.attribute.str.strip()
    return df

###
# Input checks
###


def is_valid_min_max(params: pd.DataFrame) -> bool:
    """Ensures all min values are leq to max values"""

    df = params.copy()

    if all(df["min_value"] <= df["max_value"]):
        return True
    else:
        error = df[df["min_value"] > df["max_value"]]
        print(f"Min values greater than max valuse for {error.name.to_list()}")
        return False


def is_valid_fom_units(params: pd.DataFrame) -> bool:
    """Ensures fom units match occ units."""

    df = params.copy()

    fom = df[df.attribute == "fixed_cost"].set_index("carrier")["unit"].to_frame(name="fom")
    occ = df[df.attribute == "occ"].set_index("carrier")["unit"].to_frame(name="occ")

    units = occ.join(fom)

    # transport units will be different as there are two uncertain parameters
    # occ in usd
    # fom in usd/vmt
    units = units[~units.index.str.startswith(("trn-elec-veh", "trn-lpg-veh"))]

    if all(units["fom"] == units["occ"]):
        return True
    else:
        error = units[units["fom"] != units["occ"]]
        print(f"FOM and OCC units for {error.index.to_list()} do not match")
        return False


def is_valid_range(params: pd.DataFrame) -> bool:
    """Ensures range inputs are allowed."""

    df = params.copy()

    if all(df["range"].isin(VALID_RANGES)):
        return True
    else:
        error = df[~df["range"].isin(VALID_RANGES)]
        print(f"{error.name.to_list()} do not have valid ranges of {VALID_RANGES}")
        return False

def is_valid_units(params: pd.DataFrame) -> bool:
    """Ensures converted units are valid."""

    df = params.copy()

    if all(df["unit"].isin(VALID_UNITS)):
        return True
    else:
        error = df[~df["unit"].isin(VALID_UNITS)]
        print(f"{error.name.to_list()} do not have valid units of {VALID_UNITS}")
        return False
    
def no_empty_values(params: pd.DataFrame) -> bool:
    """Ensures all required columns have data."""
    
    df = params.copy()
    
    req_cols = [x for x in df.columns if x not in ("source", "notes")]
    
    for col in req_cols:
        if any(df[col].isna()):
            error = df[df[col].isna()]
            print(f"{error.name.to_list()} in {col} column do not have values")
            return False
    
    return True
    
def is_valid_nice_name(params: pd.DataFrame) -> bool:
    """Ensures all group and nice_names are consistent."""
    
    df = params.copy()
    
    for group in df.group.unique():
        temp = df[df.group == group]
        if len(temp.nice_name.unique()) != 1:
            print(f"Inconsistent nice_names for group {group}")
            return False
    return True
    
def is_valid_capital_costs(params: pd.DataFrame) -> bool:
    """Capital costs are an intermediate calculation."""

    df = params.copy()

    temp = df[df.attribute == "capital_cost"]
    if not temp.empty:
        return False

    for attr in ["occ", "lifetime", "discount_rate", "vmt_per_year"]:
        temp = df[df.attribute == attr]
        if not all(temp.range == "absolute"):
            return False

    return True

def is_no_duplicates(params: pd.DataFrame) -> bool:
    """No duplicate parameter names"""

    df = params.copy()

    df = df[df.duplicated("name")]
    if not df.empty:
        duplicates = set(df.name.to_list())
        print(f"Duplicate definitions of {duplicates}")
        return False
    else:
        return True

def is_constraints_abs(params: pd.DataFrame) -> bool:
    """Constraints must be in absolute terms."""

    df = params.copy()

    df = df[(df.attribute.isin(CONSTRAINT_ATTRS)) & ~(df.range == "absolute")]

    if not df.empty:
        errors = set(df.name.to_list())
        print(f"{errors} must have absolute ranges")
        return False
    else:
        return True


if __name__ == "__main__":

    if "snakemake" in globals():
        in_params = snakemake.params.parameters
        out_params = snakemake.output.parameters
    else:
        in_params = "config/parameters.csv"
        out_params = "results/Test/parameters.csv"
    
    df = pd.read_csv(in_params, dtype={"min_value": float, "max_value": float})
    
    # top level sanitize 
    df = sanitize_component_name(df)
    df = strip_whitespace(df)

    # units 
    df = correct_usd(df)
    df = correct_miles(df)
    df = correct_kw_units(df)
    df = correct_kBtu_units(df)
    df = correct_mmbtu_units(df)
    df = correct_mpge_units(df)
    df = correct_percent_units(df)
    df = correct_tonnes_units(df)
    df = correct_water_heater_units(df)
    df = correct_vmt_units(df)

    
    # validation of data 
    # note, does not check carrier
    assert no_empty_values(df), "empty values exist"
    assert is_valid_attributes(df), "invalid attributes"
    assert is_valid_units(df), "invalid units"
    assert is_valid_fom_units(df), "invalid fom units"
    assert is_valid_range(df), "invalid range"
    assert is_valid_min_max(df), "invalid min/max values"
    assert is_valid_nice_name(df), "invalid nice_name"
    assert is_no_duplicates(df), "duplicate names"
    assert is_constraints_abs(df), "constraints must be absolute"

    df.to_csv(out_params, index=False)
    