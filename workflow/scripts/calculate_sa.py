"""Analyzes objective value results from model

Arguments
---------
path_to_parameters : str
    File containing the parameters for generated sample
model_inputs : str
    File path to sample model inputs
model_outputs : str
    File path to model outputs
location_to_save : str
    File path to save results
result_type : str
    True for Objective result type
    False for user defined result type 

Usage
-----
To run the script on the command line, type::

    python analyze_results.py path/to/parameters.csv path/to/inputs.txt 
        path/to/model/results.csv path/to/save/SA/results.csv

The `parameters.csv` CSV file should be formatted as follows::

    name,group,indexes,min_value,max_value,dist,interpolation_index,action
    CapitalCost,pvcapex,"GLOBAL,GCPSOUT0N",500,1900,unif,YEAR,interpolate
    DiscountRate,discountrate,"GLOBAL,GCIELEX0N",0.05,0.20,unif,None,fixed

The `inputs.txt` should be the output from SALib.sample.morris.sample

"""

from math import ceil
from SALib.analyze import morris as analyze_morris
from SALib.plotting import morris as plot_morris
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

from logging import getLogger

logger = getLogger(__name__)


# def plot_histogram(problem: pd.DataFrame, X: np.array, fig: plt.figure):

#     # chnage histogram labels to legend for clarity
#     problem_hist = problem.copy()
#     problem_hist['names'] = [f'X{x}' for x, _ in enumerate(problem_hist['names'])]
#     legend_labels = [f"{problem_hist['names'][num]} = {problem['names'][num]}" for num, _ in enumerate(problem['names'])]
#     legend_handles = [mlines.Line2D([],[], color='w', marker='.', linewidth=0, markersize=0, label=label) for label in legend_labels]

#     # plot histogram
#     ax = fig.subplots(1)
#     plot_morris.sample_histograms(fig, X, problem_hist)
#     fig.patch.set_visible(False)
#     ax.axis('off')
#     ncols = 2 if len(legend_labels) < 3 else ceil(len(legend_labels)/2)
#     fig.legend(handles=legend_handles, ncol=ncols, frameon=False, fontsize='small')
#     fig.suptitle(' ', fontsize=(ncols * 20))


def sa_results(
    parameters: pd.DataFrame, X: np.array, Y: np.array, scaled: bool = False
) -> np.array:
    """Performs SA via SALib

    Parameters
    ----------
    parameters : Dict
        Parameters for generated sample
    X : np.array
        Input Sample
    Y : np.array
        Results
    scaled : bool = False
        If the input sample is scaled
    """

    problem = utils.create_salib_problem(parameters)
    si = analyze_morris.analyze(problem, X, Y, print_to_console=False, scaled=scaled)
    return si


def plot_si(si: np.array, name: str) -> tuple[plt.figure, plt.axes]:

    # save graphical resutls
    title = name
    fig, axs = plt.subplots(1, figsize=(10, 8))
    fig.suptitle(title, fontsize=20)
    unit = ""
    plot_morris.horizontal_bar_plot(axs, si, unit=unit)
    # plot_morris.covariance_plot(axs[1], si, unit=unit)

    return fig, axs


if __name__ == "__main__":
    if "snakemake" in globals():
        result_name = snakemake.wildcards.result
        parameters_f = snakemake.input.parameters
        sample_f = snakemake.input.sample
        results_f = snakemake.input.results
        scaled = snakemake.params.scaled
        csv = snakemake.output.csv
        png = snakemake.output.png
    else:
        result_name = "com_ashp_capacity"
        parameters_f = "config/parameters.csv"
        sample_f = "results/Western/sample.csv"
        results_f = "results/Western/results/com_ashp_capacity.csv"
        scaled = False
        csv = "csv.csv"
        png = "png.png"

    params = pd.read_csv(parameters_f)
    X = pd.read_csv(sample_f).to_numpy()
    Y = pd.read_csv(results_f)["value"].to_numpy()

    assert X.shape[0] == Y.shape[0]

    si = sa_results(params, X, Y, scaled)
    si.to_df().to_csv(csv)

    fig, axs = plot_si(si, result_name)
    fig.savefig(png, bbox_inches="tight")
