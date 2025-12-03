"""Rules for processing results"""

from pathlib import Path

###
# INPUT FUNCTIONS
###

def get_sample_file(wildcards):
    if config["gsa"]["scale"]:
        return f"results/{wildcards.scenario}/gsa/sample_scaled.csv"
    else: 
        return f"results/{wildcards.scenario}/gsa/sample.csv"

def get_gsa_plotting_csvs(wildcards):
    csv = f"results/{wildcards.scenario}/gsa/results.csv"
    if Path(csv).exists():
        df = pd.read_csv(csv).dropna(subset=["gsa_plot"]).copy()
        df = df[df.gsa_plot.str.contains(wildcards.plot)]
        results = df.name.to_list()
        return [f"results/{wildcards.scenario}/gsa/SA/{x}.csv" for x in results]
    else: # Called on import
        return []

def get_ua_scatterplot_csvs(wildcards):
    # Read plots.csv (created by sanitize_ua_plot_params rule)
    # Note: This function is called during DAG construction, but plots.csv is an input
    # to plot_ua_scatter, so Snakemake will ensure sanitize_ua_plot_params runs first.
    # However, during initial DAG construction, the file may not exist yet.
    csv = f"results/{wildcards.scenario}/ua/plots.csv"
    try:
        df = pd.read_csv(csv, index_col=0)
        df = df[df["plot"] == wildcards.plot]
        if df.empty:
            return []
        csvs = df.xaxis.to_list() + df.yaxis.to_list()
        return [f"results/{wildcards.scenario}/ua/results/{x}.csv" for x in csvs]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # File doesn't exist yet - Snakemake will re-evaluate after sanitize_ua_plot_params runs
        # Return a placeholder that will be replaced when the file exists
        # The actual files will be determined when the rule executes
        return [f"results/{wildcards.scenario}/ua/results/placeholder_{wildcards.plot}.csv"]


def get_ua_barplot_csvs(wildcards):
    csv = f"results/{wildcards.scenario}/ua/results.csv"
    df = pd.read_csv(csv).dropna(subset=["ua_plot"]).copy()
    df = df[df.ua_plot == wildcards.plot]
    assert not df.empty
    csvs = df.name.to_list()
    return [f"results/{wildcards.scenario}/ua/results/{x}.csv" for x in csvs]

def get_combined_results_inputs(wildcards) -> list[str]:
    """Need input function as we need to get model run numbers."""
    if wildcards.mode == "gsa":
        modelruns = get_gsa_modelruns()
    elif wildcards.mode == "ua":
         modelruns = get_ua_modelruns()
    else:
        raise ValueError(f"Invalid result of {wildcards.mode} for model runs.")

    return [f"results/{wildcards.scenario}/{wildcards.mode}/modelruns/{run}/results.csv" for run in modelruns]

###
# SHARED RULES
###

rule extract_results:
    message: "Extracting result"
    wildcard_constraints:
        mode="gsa|ua"
    input:
        network = "results/{scenario}/{mode}/modelruns/{run}/network.nc",
        results = "results/{scenario}/{mode}/results.csv"
    output:
        csv = "results/{scenario}/{mode}/modelruns/{run}/results.csv"
    log: 
        "logs/extract_results/{scenario}_{mode}_{run}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/extract_gsa_results/{scenario}_{mode}_{run}.txt"
    group:
        "solve_{scenario}_{mode}_{run}"
    script:
        "../scripts/extract_results.py"

def get_sanitized_results_file(wildcards):
    """Get sanitized results file"""
    return f"results/{wildcards.scenario}/{wildcards.mode}/results.csv"

rule combine_results:
    message: "Collapsing all results into single summary file"
    wildcard_constraints:
        mode="gsa|ua"
    input:
        sanitized_results = get_sanitized_results_file,
        results = get_combined_results_inputs
    output:
        csv = temp("results/{scenario}/{mode}/results/all.csv")
    log: 
        "logs/combine_results/{scenario}_{mode}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/combine_results/{scenario}_{mode}.txt"
    group:
        "results"
    run:
        import pandas as pd
        data = [pd.read_csv(str(x)) for x in input.results]
        df = pd.concat(data)
        df.to_csv(output.csv, index=False)

###
# GSA RULES
###

def get_sanitized_gsa_results_file(wildcards):
    """Get sanitized GSA results file"""
    return f"results/{wildcards.scenario}/gsa/results.csv"

rule parse_gsa_results:
    message: "Parsing results by results file"
    params:
        base_dir = "results/{scenario}/gsa/results/"
    input:
        sanitized_results = get_sanitized_gsa_results_file,
        results = "results/{scenario}/gsa/results/all.csv"
    output:
        expand("results/{{scenario}}/gsa/results/{sa_result}.csv", sa_result=get_gsa_result_files())
    log: 
        "logs/parse_results/{scenario}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/parse_results/{scenario}.txt"
    group:
        "results"
    run:
        import pandas as pd
        from pathlib import Path 
        df = pd.read_csv(input.results)
        for name in df.name.unique():
            parsed = df[df.name == name].sort_values(by=["run"]).drop(columns=["name"])
            p = Path(params.base_dir, f"{name}.csv")
            parsed.to_csv(str(p), index=False)

rule calculate_SA:
    wildcard_constraints:
        # idk why but the regex on the next line doesnt work. 
        # sa_result="^(?!all$).*$", # sa_result can not be 'all'
        sa_result='|'.join([re.escape(x) for x in get_gsa_result_files()])
    message:
        "Calcualting sensitivity measures"
    params: 
        scaled = config["gsa"]["scale"]
    input: 
        sample = get_sample_file,
        parameters = "results/{scenario}/gsa/parameters.csv",
        results = "results/{scenario}/gsa/results/{sa_result}.csv"
    output: 
        csv = "results/{scenario}/gsa/SA/{sa_result}.csv",
        png = "results/{scenario}/gsa/SA/{sa_result}.png"
    log: 
        "logs/calculate_sa/{scenario}_{sa_result}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/calculate_sa/{scenario}_{sa_result}.txt"
    group:
        "results"
    script: 
        "../scripts/calculate_sa.py"

rule combine_sa_results:
    message: "Combining SA results"
    input:
        csvs = expand("results/{{scenario}}/gsa/SA/{sa_result}.csv", sa_result=get_gsa_result_files())
    output:
        csv = "results/{scenario}/gsa/SA/all.csv"
    log: 
        "logs/combine_sa_results/{scenario}_gsa.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/combine_sa_results/{scenario}_gsa.txt"
    group:
        "results"
    run:
        import pandas as pd
        from pathlib import Path 

        dfs = []
        for f in input.csvs:
            name = Path(f).stem
            df = pd.read_csv(f, index_col=0).mu_star
            df.name = name
            dfs.append(df)
        df = pd.concat(dfs, axis=1).round(5).reset_index(names="param")
        df.to_csv(output.csv, index=False)


rule calculate_rankings:
    message: "Calculating rankings"
    params:
        top_n = config["gsa"]["rankings"]["top_n"],
        subset = config["gsa"]["rankings"]["results"]
    input:
        results = "results/{scenario}/gsa/SA/all.csv"
    output:
        rankings_f = "results/{scenario}/gsa/rankings.csv",
        top_n_f = "results/{scenario}/gsa/rankings_top_n.csv"
    log: 
        "logs/calculate_rankings/{scenario}_gsa.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/calculate_rankings/{scenario}_gsa.txt"
    group:
        "results"
    script: 
        "../scripts/calculate_rankings.py"

rule heatmap:
    message:
        "Generating heat map"
    input:
        rules.combine_sa_results.output,
        params = "results/{scenario}/gsa/parameters.csv",
        results = "results/{scenario}/gsa/results.csv",
        csvs = get_gsa_plotting_csvs
    output:
        heatmap = "results/{scenario}/gsa/heatmaps/{plot}.png"
    log: 
        "logs/create_heatmap/{scenario}_{plot}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 500),
        runtime=1
    benchmark:
        "benchmarks/create_heatmap/{scenario}_{plot}.txt"
    group:
        "results"
    script:
        "../scripts/plot_gsa_heatmap.py"

rule barplot:
    message:
        "Generating barplot"
    input:
        rules.combine_sa_results.output,
        params = "results/{scenario}/gsa/parameters.csv",
        results = "results/{scenario}/gsa/results.csv",
        csvs = get_gsa_plotting_csvs
    output:
        barplot = "results/{scenario}/gsa/barplots/{plot}.png"
    log: 
        "logs/create_barplot/{scenario}_{plot}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 500),
        runtime=1
    benchmark:
        "benchmarks/create_barplot/{scenario}_{plot}.txt"
    group:
        "results"
    script:
        "../scripts/plot_gsa_barplot.py"

###
# UA RULES
###

def get_sanitized_ua_results_file(wildcards):
    """Get sanitized UA results file"""
    return f"results/{wildcards.scenario}/ua/results.csv"

rule parse_ua_results:
    message: "Parsing results by results file"
    params:
        base_dir = "results/{scenario}/ua/results/"
    input:
        sanitized_results = get_sanitized_ua_results_file,
        results = "results/{scenario}/ua/results/all.csv"
    output:
        expand("results/{{scenario}}/ua/results/{name}.csv", name=get_ua_result_files())
    log: 
        "logs/parse_results/{scenario}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/parse_results/{scenario}.txt"
    group:
        "results"
    run:
        import pandas as pd
        from pathlib import Path 
        df = pd.read_csv(input.results)
        for name in df.name.unique():
            parsed = df[df.name == name].sort_values(by=["run"]).drop(columns=["name"])
            p = Path(params.base_dir, f"{name}.csv")
            parsed.to_csv(str(p), index=False)

rule plot_ua_scatter:
    message:
        "Generating UA scatter plots plots"
    params:
        root_dir = "results/{scenario}/ua/results/"
    input:
        sanitized_results = get_sanitized_ua_results_file,
        csvs = get_ua_scatterplot_csvs,
        plots_csv = "results/{scenario}/ua/plots.csv"  # Explicit dependency on sanitize_ua_plot_params output
    output:
        plot = "results/{scenario}/ua/scatterplots/{plot}.png"
    log: 
        "logs/plot_ua/{scenario}_{plot}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 500),
        runtime=1
    benchmark:
        "benchmarks/plot_ua/{scenario}_{plot}.txt"
    group:
        "results"
    script:
        "../scripts/plot_ua_scatter.py"

rule plot_ua_barplots:
    message:
        "Generating UA barplots"
    params:
        root_dir = "results/{scenario}/ua/results/"
    input:
        sanitized_results = get_sanitized_ua_results_file,
        csvs = get_ua_barplot_csvs,
        results = "results/{scenario}/ua/results.csv"
    output:
        plot = "results/{scenario}/ua/barplots/{plot}.png"
    log: 
        "logs/plot_ua/{scenario}_{plot}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 500),
        runtime=1
    benchmark:
        "benchmarks/plot_ua/{scenario}_{plot}.txt"
    group:
        "results"
    script:
        "../scripts/plot_ua_barplot.py"