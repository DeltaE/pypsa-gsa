"""Applies sampled value to the base network"""

import pandas as pd
import pypsa
from typing import Any
from pathlib import Path
import yaml
from dataclasses import dataclass
from utils import calculate_annuity
from constants import CACHED_ATTRS

from logging import getLogger

logger = getLogger(__name__)


@dataclass
class CapitalCostCache:
    """Perform intermediate capital cost calculation."""

    component: str
    capital_cost: float = None
    lifetime: int = None
    discount_rate: float = None
    fixed_cost: float = None
    occ: float = None
    itc: float = None
    vmt_per_year: float = None

    def is_valid_data(self) -> bool:
        """Checks that all data is present before applying sampel"""
        if self.capital_cost:
            return True
        if not self.occ:
            raise ValueError("occ")
        elif not self.fixed_cost:
            raise ValueError("fom")
        elif not self.discount_rate:
            raise ValueError("discount rate")
        elif not (self.lifetime or self.vmt_per_year):
            raise ValueError("lifetime or vmt_per_year")
        else:
            return True

    def calculate_capex(self, transport: bool = False) -> float:
        """Capex is an intermediate calcualtion.

        Fixed cost is given in same units as occ.
        """
        assert self.is_valid_data()
        if self.capital_cost:
            logger.info("Returning pre-defined capital cost.")
            return round(self.capital_cost, 2)
        elif transport:
            return self._calc_transport_capex()
        else:
            return self._calc_standard_capex()

    def _calc_standard_capex(self) -> float:
        assert self.discount_rate < 1  # ensures is per_unit
        annuity = calculate_annuity(self.lifetime, self.discount_rate)
        capex = (self.occ + self.fixed_cost) * annuity

        if self.itc:
            assert self.itc < 1  # ensures is per_unit
            return round(capex * (1 - self.itc), 2)
        else:
            return round(capex, 2)

    def _calc_transport_capex(self) -> float:
        """OCC comes as 'usd' and needs to be converted to usd/kvmt."""
        assert self.discount_rate < 1  # ensures is per_unit
        annuity = calculate_annuity(self.lifetime, self.discount_rate)

        assert self.vmt_per_year
        capex = ((self.occ / self.vmt_per_year) + self.fixed_cost) * annuity

        return round(capex, 2)


def is_valid_carrier(n: pypsa.Network, params: pd.DataFrame) -> bool:
    """Check all defined carriers are in the network."""

    df = params.copy()

    sa_cars = df.carrier.unique()
    n_cars = n.carriers.index.to_list()

    errors = []

    for car in sa_cars:
        if car not in n_cars:
            errors.append(car)

    if errors:
        logger.error(f"{errors} are not defined in network.")
        return False
    else:
        return True

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
    """Applies a sample to a network for a single model run.

    As there are some intermediate calculations, some data is cached and applied at the end.

    This will modify the network! Pass a copy of a network if you dont want to modify the
    reference network.
    """

    meta = {}
    cached = {}  # for internediate calcualtions

    for name, data in sample.items():
        c = data["component"]
        car = data["carrier"]
        attr = data["attribute"]
        absolute = True if data["range"] == "absolute" else False
        value = round(data["value"][run], 2)

        if attr in CACHED_ATTRS and absolute:
            if car not in cached:
                cached[car] = {"component": c}
            cached[car][attr] = value
        elif c.endswith("_t"):
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

    for car, data in cached.items():
        try:
            cache = CapitalCostCache(**data)
            capex = cache.calculate_capex()
        except ValueError as ex:
            logger.error(f"Capital cost error with {car}")
            raise ValueError(ex)
        if car.endswith("battery_storage"):
            pass
        # logger.info(f"Applying capex to {car}")
        _apply_static_sample(n, cache.component, car, "capital_cost", capex, "absolute")

    return meta


if __name__ == "__main__":
    if "snakemake" in globals():
        param_file = snakemake.input.parameters
        sample_file = snakemake.input.sample_file
        base_network_file = snakemake.input.network
        root_dir = Path(snakemake.params.root_dir)
    else:
        param_file = "results/Testing/parameters.csv"
        sample_file = "results/Testing/sample.csv"
        base_network_file = "results/Testing/base.nc"
        root_dir = Path("results/Testing/modelruns/")

    params = pd.read_csv(param_file)
    sample = pd.read_csv(sample_file)
    base_n = pypsa.Network(base_network_file)

    # check carrier here as it requires reading in network
    assert is_valid_carrier(base_n, params)

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
