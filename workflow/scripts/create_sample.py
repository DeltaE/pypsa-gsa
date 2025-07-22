"""Generates a sample from a list of parameters."""

from SALib.sample import morris, latin, sobol
import pandas as pd
from utils import create_salib_problem
from utils import configure_logging


import logging
logger = logging.getLogger(__name__)


def main(parameters: pd.DataFrame, method: str, replicates: int):
    """Creates sample using SALib."""

    problem = create_salib_problem(parameters, method)

    if method == "morris":
        sample = morris.sample(
            problem,
            N=100,
            optimal_trajectories=replicates,
            local_optimization=True,
            seed=42,
        )
    elif method == "lhs":
        sample = latin.sample(problem, N=replicates, seed=42)
    elif method == "sobol":
        sample = sobol.sample(
            problem, N=(2**replicates), calc_second_order=False, seed=42
        )
    else:
        raise ValueError(
            f"{method} is not a supported sampling method. Choose from ['morris', 'lhs', 'sobol']"
        )

    return pd.DataFrame(sample, columns=problem["names"]).round(5)


if __name__ == "__main__":

    if "snakemake" in globals():
        param_file = snakemake.input.parameters
        method = snakemake.params.method
        replicates = int(snakemake.params.replicates)
        sample_file = snakemake.output.sample_file
        configure_logging(snakemake)
    else:
        param_file = "results/caiso/gsa/parameters.csv"
        replicates = 10
        sample_file = "results/caiso/gsa/sample.csv"
        method = "morris"

    parameters = pd.read_csv(param_file)

    df = main(parameters, method, replicates)

    df.to_csv(sample_file, index=False)
