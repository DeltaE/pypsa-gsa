"""Generates an scaled sample from a list of parameters

Arguments
---------
path_to_parameters : str
    File containing the parameters to generate a sample for

sample_file : str
    File path to save sample to

replicates : int
    The number of model runs to generate

Usage
-----
To run the script on the command line, type::

    python create_sample.py path/to/parameters.csv path/to/save.txt 10

The ``parameters.csv`` CSV file should be formatted as follows::

    name,group,indexes,min_value,max_value,dist,interpolation_index,action
    CapitalCost,pvcapex,"GLOBAL,GCPSOUT0N",500,1900,unif,YEAR,interpolate
    DiscountRate,discountrate,"GLOBAL,GCIELEX0N",0.05,0.20,unif,None,fixed
"""

from SALib.sample import morris
import pandas as pd
import numpy as np
import csv
import sys
import utils

from logging import getLogger

logger = getLogger(__name__)


def main(parameters: pd.DataFrame, replicates: int):

    problem = utils.create_salib_problem(parameters)

    sample = morris.sample(
        problem,
        N=100,
        optimal_trajectories=replicates,
        local_optimization=True,
        seed=42,
    )

    return pd.DataFrame(sample, columns=problem["names"])


if __name__ == "__main__":

    if "snakemake" in globals():
        param_file = snakemake.params.parameters
        replicates = int(snakemake.params.replicates)
        sample_file = snakemake.output.sample_file
    else:
        # param_file = sys.argv[1]
        # replicates = int(sys.argv[2])
        # sample_file = sys.argv[3]
        param_file = "../../config/parameters.csv"
        replicates = 10
        sample_file = "sample.csv"

    parameters = pd.read_csv(param_file)

    df = main(parameters, replicates)

    df.to_csv(sample_file, index=False)
