"""Generates a sample from a list of parameters."""

from SALib.sample import morris
import pandas as pd
from utils import create_salib_problem

from logging import getLogger

logger = getLogger(__name__)


def main(parameters: pd.DataFrame, replicates: int):
    problem = create_salib_problem(parameters)

    sample = morris.sample(
        problem,
        N=100,
        optimal_trajectories=replicates,
        local_optimization=True,
        seed=42,
    )

    return pd.DataFrame(sample, columns=problem["names"]).round(5)


if __name__ == "__main__":

    if "snakemake" in globals():
        param_file = snakemake.input.parameters
        replicates = int(snakemake.params.replicates)
        sample_file = snakemake.output.sample_file
    else:
        param_file = "../../config/parameters.csv"
        replicates = 10
        sample_file = "sample.csv"

    parameters = pd.read_csv(param_file)

    df = main(parameters, replicates)

    df.to_csv(sample_file, index=False)
