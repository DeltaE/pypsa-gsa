"""Sanitizes result file that describes uncertainity plots."""

import pandas as pd
from constants import VALID_UA_PLOTS

import logging
logger = logging.getLogger(__name__)

def sanitize_values(plots: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes values to fill nas and confirm types."""
    
    return plots.copy().fillna("")

def is_valid_plot_types(plots: pd.DataFrame) -> bool:
    """Confirm variables are valid."""

    df = plots.copy()
    df = df[~df.type.isin(VALID_UA_PLOTS)]

    if not df.empty:
        bad_types = set(df.type.to_list())
        print(f"Incorrect plot types of {bad_types}")
        return False
    else:
        return True

def is_unique_names(plots: pd.DataFrame) -> bool:
    """Checks that all result names are unique."""

    df = plots.copy()

    df = df[df.duplicated("name")]
    if not df.empty:
        duplicates = set(df.name.to_list())
        print(f"Duplicate definitions of {duplicates}")
        return False
    else:
        return True

def has_required_inputs(plots: pd.DataFrame) -> bool:
    """Required plotting information is procided."""
    
    df = plots.copy()
    
    if not all(df.name.notna()):
        print("Provide name for all UA plots")
        return False
    if not all(df.type.notna()):
        print("Provide types for UA plots")
        return False
    if not all(df.xaxis.notna()):
        print("Provide xaxis inputs for all UA plots")
        return False
    if not all(df.yaxis.notna()):
        print("Provide yaxis inputs for all UA plots")
        return False
    return True

def is_valid_axis(plots: pd.DataFrame, results: pd.DataFrame) -> bool:
    """Ensures plotting axis are valid results."""
    
    df = plots.copy()
    valid_results = results.name.to_list()
    
    df1 = df[~df.xaxis.isin(valid_results)]
    if not df1.empty:
        invalid = set(df1.xaxis.to_list())
        print(f"Invalide results of {invalid} in xaxis UA plots")
        return False
    
    df2 = df[~df.yaxis.isin(valid_results)]
    if not df2.empty:
        invalid = set(df2.xaxis.to_list())
        print(f"Invalide results of {invalid} in yaxis UA plots")
        return False
        
    return True
    
if __name__ == "__main__":
    if "snakemake" in globals():
        in_plots_f = snakemake.input.plots
        out_plots_f = snakemake.output.plots
        results_f = snakemake.input.results
    else:
        in_plots_f = "config/plots_ua.csv"
        out_plots_f = "results/updates/ua/plots.csv"
        results_f = "results/updates/ua/results.csv"
    
    results = pd.read_csv(results_f)
    plots = pd.read_csv(in_plots_f)

    assert has_required_inputs(plots)
    assert is_unique_names(plots)
    assert is_valid_plot_types(plots)
    assert is_valid_axis(plots, results)

    plots.to_csv(out_plots_f, index=False)
