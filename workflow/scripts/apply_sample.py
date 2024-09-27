"""Applies sampled value to the base network"""

import pandas as pd
import pypsa
from typing import Any, Optional
from pathlib import Path
from constants import DEFAULT_LINK_ATTRS, DEFAULT_GENERATOR_ATTRS, DEFAULT_STORE_ATTRS
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
        VALID_GENERATOR_ATTRS = DEFAULT_GENERATOR_ATTRS
        VALID_STORE_ATTRS = DEFAULT_STORE_ATTRS

    def _sanitize_component_name(c: str) -> str:

        c = c.lower()

        match c:
            case "link" | "links":
                return "links"
            case "generator" | "generators":
                return "generators"
            case "store" | "stores":
                return "stores"
            case _:
                raise NotImplementedError

    def _check_attribute(c: str, attr: str):

        if c == "links":
            assert attr in VALID_LINK_ATTRS
        elif c == "generators":
            assert attr in VALID_GENERATOR_ATTRS
        elif c == "stores":
            assert attr in VALID_STORE_ATTRS
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
        data[name]["value"] = sample[name].to_dict()  # run: value
    return data


def apply_sample(
    n: pypsa.Network, sample: dict[str, dict[str, Any]], run: int
) -> dict[dict[str, str | float]]:
    """Applies a sample to a network for a single model run

    This will modify the network! Pass a copy of a network if you dont want to modeify the
    reference network.
    """

    meta = {}

    for name, data in sample.items():
        c = data["component"]
        car = data["carrier"]
        attr = data["attribute"]
        value = round(data["value"][run], 2)

        df = getattr(n, c)
        slicer = df[df.carrier == car].index

        getattr(n, c).loc[slicer, attr] = value

        meta[str(name)] = {
            "component": c,
            "carrier": car,
            "attribute": attr,
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
        sample_file = "resources/sample.csv"
        base_network_file = "resources/elec_s50_c35_ec_lv1.0_48SEG_E-G.nc"
        root_dir = Path("results/model_runs/Western/")

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
