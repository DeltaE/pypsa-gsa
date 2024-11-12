"""Applies sampled value to the base network"""

import pandas as pd
import pypsa
from typing import Any, Optional
from pathlib import Path
from constants import (
    DEFAULT_LINK_ATTRS,
    DEFAULT_GENERATOR_ATTRS,
    DEFAULT_STORE_ATTRS,
    DEFAULT_LINK_T_ATTRS,
    DEFAULT_LOAD_T_ATTRS,
)
import yaml


def create_directory(d: str | Path, del_existing: bool = True) -> None:
    """Removes exising data and creates empty directory"""

    if isinstance(d, str):
        d = Path(d)

    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        return

    existing_files = [x.name for x in d.iterdir()]

    if not del_existing and existing_files:
        raise ValueError(f"Delete files {existing_files} in {str(d)} and rerun")
    elif del_existing and existing_files:
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    return


def sanitzie_params(
    params: pd.DataFrame, n: Optional[pypsa.Network] = None
) -> pd.DataFrame:

    if isinstance(n, pypsa.Network):
        VALID_LINK_ATTRS = n.component_attrs["Link"].index
        VALID_GENERATOR_ATTRS = n.component_attrs["Generator"].index
        VALID_STORE_ATTRS = n.component_attrs["Store"].index
    else:
        VALID_LINK_ATTRS = DEFAULT_LINK_ATTRS
        VALID_LINK_T_ATTRS = DEFAULT_LINK_T_ATTRS
        VALID_GENERATOR_ATTRS = DEFAULT_GENERATOR_ATTRS
        VALID_STORE_ATTRS = DEFAULT_STORE_ATTRS
        VALID_LOAD_T_ATTRS = DEFAULT_LOAD_T_ATTRS

    def _sanitize_component_name(c: str) -> str:

        c = c.lower()

        match c:
            case "link" | "links":
                return "links"
            case "generator" | "generators":
                return "generators"
            case "store" | "stores":
                return "stores"
            case "links_t" | "link_t":
                return "links_t"
            case "loads_t" | "load_t":
                return "loads_t"
            case _:
                raise NotImplementedError

    def _check_attribute(c: str, attr: str):

        if c == "links":
            assert attr in VALID_LINK_ATTRS
        elif c == "links_t":
            assert attr in VALID_LINK_T_ATTRS
        elif c == "generators":
            assert attr in VALID_GENERATOR_ATTRS
        elif c == "stores":
            assert attr in VALID_STORE_ATTRS
        elif c == "loads_t":
            assert attr in VALID_LOAD_T_ATTRS
        else:
            raise NotImplementedError

    df = params.copy()

    df["component"] = df.component.map(lambda x: _sanitize_component_name(x))

    df.apply(lambda row: _check_attribute(row["component"], row["attribute"]), axis=1)

    return df


def get_sample_data(
    params: pd.DataFrame, sample: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Gets data structure for applying data"""

    data = {}
    p = params.set_index("name")
    for name in p.index:
        data[name] = {}
        data[name]["component"] = p.at[name, "component"]
        data[name]["carrier"] = p.at[name, "carrier"]
        data[name]["attribute"] = p.at[name, "attribute"]
        data[name]["range"] = p.at[name, "range"]  # absolute | percent
        data[name]["value"] = sample[name].to_dict()  # run: value
    return data


def _apply_static_sample(
    n: pypsa.Network, c: str, car: str, attr: str, value: int | float, absolute: bool
):
    df = getattr(n, c)
    slicer = df[df.carrier == car].index
    if absolute:
        getattr(n, c).loc[slicer, attr] = value
    else:
        ref = getattr(n, c).loc[slicer, attr]
        multiplier = value / 100  # can be positive or negative
        getattr(n, c).loc[slicer, attr] = ref + ref.mul(multiplier)
        print("")


def _apply_dynamic_sample(
    n: pypsa.Network, c: str, car: str, attr: str, value: int | float
):
    df_t = getattr(n, c)[attr]
    df_static = getattr(n, c.split("_t")[0])

    name_to_carrier = df_static.carrier.to_dict()
    name_carrier_map = {x: name_to_carrier[x] for x in df_t.columns}
    names = [x for x, y in name_carrier_map.items() if y == car]

    ref = getattr(n, c)[attr].loc[:, names]  # same as df_t.loc[...]
    multiplier = value / 100  # can be positive or negative
    getattr(n, c)[attr].loc[:, names] = ref + ref.mul(multiplier)


def apply_sample(
    n: pypsa.Network, sample: dict[str, dict[str, Any]], run: int
) -> dict[dict[str, str | float]]:
    """Applies a sample to a network for a single model run

    This will modify the network! Pass a copy of a network if you dont want to modify the
    reference network.
    """

    meta = {}

    for name, data in sample.items():
        c = data["component"]
        car = data["carrier"]
        attr = data["attribute"]
        absolute = True if data["range"] == "absolute" else False
        value = round(data["value"][run], 2)

        if c.endswith("_t"):
            assert not absolute
            _apply_dynamic_sample(n, c, car, attr, value)
        else:
            _apply_static_sample(n, c, car, attr, value, absolute)

        meta[str(name)] = {
            "component": c,
            "carrier": car,
            "attribute": attr,
            "range": data["range"],
            "value": value,
        }

    return meta


if __name__ == "__main__":

    if "snakemake" in globals():
        param_file = snakemake.params.parameters
        sample_file = snakemake.input.sample_file
        base_network_file = snakemake.input.network
        root_dir = Path(snakemake.params.root_dir)
    else:
        # param_file = sys.argv[1]
        # replicates = int(sys.argv[2])
        # sample_file = sys.argv[3]
        param_file = "config/parameters.csv"
        sample_file = "results/California/sample.csv"
        base_network_file = "resources/elec_s33_c4m_ec_lv1.0_2190SEG_E-G.nc"
        root_dir = Path("results/model_runs/California/")

    params = pd.read_csv(param_file)
    sample = pd.read_csv(sample_file)
    base_n = pypsa.Network(base_network_file)

    params = sanitzie_params(params)

    sample_data = get_sample_data(params, sample)

    for run in range(len(sample)):

        n = base_n.copy()

        meta = apply_sample(n, sample_data, run)

        create_directory(Path(root_dir, str(run)))

        n_save_name = Path(root_dir, str(run), "n.nc")

        meta_save_name = Path(root_dir, str(run), "meta.yaml")

        n.export_to_netcdf(n_save_name)

        with open(meta_save_name, "w") as f:
            yaml.dump(meta, f)
