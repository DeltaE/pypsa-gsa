"""Independent script to collect result data into a database."""

import pandas as pd
from pathlib import Path
import json

from components.utils import ISOS

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

root = Path(__file__).parent.parent

ROUND_TO = 5


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    """Converts small values to zero."""
    return df.mask(df.abs() < 1e-8, 0).round(ROUND_TO)


def get_param_names(root: Path, iso: str, mode: str) -> dict[str, str]:
    """Gets names and nice_names for the parameter data."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_f = Path(root, "results", iso, mode, "parameters.csv")
    df = pd.read_csv(data_f, index_col=0)[["group", "nice_name"]]
    logger.debug(f"Collecting nice names from {data_f}.")
    df = df.drop_duplicates()
    return df.set_index("group")["nice_name"].to_dict()


def get_result_names(root: Path, iso: str, mode: str) -> dict[str, str]:
    """Gets names and nice_names for the result data."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_f = Path(root, "results", iso, mode, "results.csv")
    df = pd.read_csv(data_f, index_col=0)
    logger.debug(f"Collecting nice names from {data_f}.")
    return df.nice_name.to_dict()


def collect_sa(root: Path, iso: str, results: list[str]) -> pd.DataFrame:
    """Collects the SA data from the results directory."""

    data_dir = Path(root, "results", iso, "gsa", "SA")
    logger.debug(f"Collecting SA data for {iso} from {data_dir}.")

    inputs = [Path(data_dir, f"{x}.csv") for x in results]

    dfs = []
    for f in inputs:
        name = f.stem
        df = pd.read_csv(f, index_col=0).mu_star
        df.name = name
        dfs.append(df)
    df = _round_df(pd.concat(dfs, axis=1))
    df["iso"] = iso
    df = df.reset_index(names=["param"])
    return df.set_index(["param", "iso"])


def collect_runs(root: Path, iso: str, mode: str, results: list[str]) -> pd.DataFrame:
    """Collects the model run results data from the results directory."""

    assert mode in ("gsa", "ua"), f"Invalid mode: {mode}. Must be 'gsa' or 'ua'."

    data_dir = Path(root, "results", iso, mode, "results")
    logger.debug(f"Collecting model run results data for {iso} from {data_dir}.")

    inputs = [Path(data_dir, f"{x}.csv") for x in results]

    dfs = []
    for f in inputs:
        name = f.stem
        try:
            df = pd.read_csv(f, index_col=1)
        except FileNotFoundError:
            logger.warning(f"No file for for {f}.")
            continue
        df = df.rename(columns={"value": name})
        dfs.append(df)
    df = _round_df(pd.concat(dfs, axis=1))
    df["iso"] = iso
    df = df.reset_index(names=["run"])
    return df.set_index(["run", "iso"])


def get_empty_sa() -> pd.DataFrame:
    """Returns an empty dataframe for the SA data."""
    return pd.DataFrame(columns=["param", "iso"]).set_index(["param", "iso"])


def get_empty_run() -> pd.DataFrame:
    """Returns an empty dataframe for the run data."""
    return pd.DataFrame(columns=["param", "iso"]).set_index(["param", "iso"])


if __name__ == "__main__":
    # sensitivity measures

    dfs = []
    for iso in ISOS:
        iso_data = Path(root, "results", iso, "gsa", "SA")
        if not iso_data.exists():
            logger.warning(f"No gsa sa data for '{iso}'")
            dfs.append(get_empty_sa())
        else:
            sa_names = get_result_names(root, iso, "gsa")
            df = collect_sa(root, iso, sa_names.keys())
            dfs.append(df)

    if not dfs:
        logger.error("No data found.")
        raise ValueError("No ISO data found.")

    sa = pd.concat(dfs, axis=0)
    sa.to_csv(Path(root, "dashboard", "data", "sa.csv"), index=True)

    # model run results gsa

    dfs = []
    for iso in ISOS:
        iso_data = Path(root, "results", iso, "gsa", "results")
        if not iso_data.exists():
            logger.warning(f"No gsa model run data for '{iso}'")
            dfs.append(get_empty_run())
        else:
            result_names = get_result_names(root, iso, "gsa")
            df = collect_runs(root, iso, "gsa", result_names.keys())
            dfs.append(df)

    if not dfs:
        logger.error("No data found.")
        raise ValueError("No ISO data found.")

    sa = pd.concat(dfs, axis=0)
    sa.to_csv(Path(root, "dashboard", "data", "gsa_runs.csv"), index=True)

    # model run results ua

    dfs = []
    for iso in ISOS:
        iso_data = Path(root, "results", iso, "ua", "results")
        if not iso_data.exists():
            logger.warning(f"No ua model run data for '{iso}'")
            dfs.append(get_empty_run())
        else:
            result_names = get_result_names(root, iso, "ua")
            df = collect_runs(root, iso, "ua", result_names.keys())
            dfs.append(df)
    if not dfs:
        logger.error("No data found.")
        raise ValueError("No ISO data found.")

    sa = pd.concat(dfs, axis=0)
    sa.to_csv(Path(root, "dashboard", "data", "ua_runs.csv"), index=True)

    # get nice names

    for iso in ISOS:
        try:
            sa_params = get_param_names(root, iso, "gsa")
            sa_results = get_result_names(root, iso, "gsa")
        except FileNotFoundError:
            logger.debug(f"No gsa nice names for {iso}.")
            pass
        else:
            continue

    with open(Path(root, "dashboard", "data", "sa_params.json"), "w") as f:
        json.dump(sa_params, f, indent=4)

    with open(Path(root, "dashboard", "data", "sa_results.json"), "w") as f:
        json.dump(sa_results, f, indent=4)

    for iso in ISOS:
        try:
            ua_params = get_param_names(root, iso, "ua")
            ua_results = get_result_names(root, iso, "ua")
        except FileNotFoundError:
            logger.debug(f"No ua nice names for {iso}.")
            pass
        else:
            continue

    with open(Path(root, "dashboard", "data", "ua_params.json"), "w") as f:
        json.dump(ua_params, f, indent=4)

    with open(Path(root, "dashboard", "data", "ua_results.json"), "w") as f:
        json.dump(ua_results, f, indent=4)
