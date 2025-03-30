"""Rules for processing results"""

###
# INPUT FUNCTIONS
###

def get_sample_file(wildcards):
    if config["gsa"]["scale"]:
        return f"results/{wildcards.scenario}/gsa/sample_scaled.csv"
    else: 
        return f"results/{wildcards.scenario}/gsa/sample.csv"

def get_gsa_plotting_csvs(wildcards):
    csv = checkpoints.sanitize_results.get(scenario=wildcards.scenario, mode="gsa").output[0]
    df = pd.read_csv(csv)
    df = df[df.plots.str.contains(wildcards.plot)]
    results = df.name.to_list()

    return [f"results/{wildcards.scenario}/gsa/SA/{x}.csv" for x in results]

def get_ua_scatterplot_csvs(wildcards):
    csv = checkpoints.sanitize_ua_plot_params.get(scenario=wildcards.scenario).output[0]
    df = pd.read_csv(csv, index_col=0)
    df = df[df["plot"] == wildcards.plot]
    assert not df.empty
    csvs = df.xaxis.to_list() + df.yaxis.to_list()
    return [f"results/{wildcards.scenario}/ua/results/{x}.csv" for x in csvs]

def get_ua_barplot_csvs(wildcards):
    csv = checkpoints.sanitize_results.get(scenario=wildcards.scenario, mode="ua").output[0]
    df = pd.read_csv(csv)
    df = df[df.barplot == wildcards.plot]
    assert not df.empty
    csvs = df.name.to_list()
    return [f"results/{wildcards.scenario}/ua/results/{x}.csv" for x in csvs]

def get_combined_results_inputs(wildcards):
    """Need input function as we need to get model run numbers."""
    if wildcards.mode == "gsa":
        modelruns = GSA_MODELRUNS
    elif wildcards.mode == "ua":
         modelruns = UA_MODELRUNS
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

rule combine_results:
    message: "Collapsing all results into single summary file"
    wildcard_constraints:
        mode="gsa|ua"
    input:
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

rule parse_gsa_results:
    message: "Parsing results by results file"
    params:
        base_dir = "results/{scenario}/gsa/results/"
    input:
        results = "results/{scenario}/gsa/results/all.csv"
    output:
        expand("results/{{scenario}}/gsa/results/{name}.csv", name=GSA_RESULT_FILES)
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
    message:
        "Calcualting sensitivity measures"
    params: 
        scaled = config["gsa"]["scale"]
    input: 
        sample = get_sample_file,
        parameters = "results/{scenario}/gsa/parameters.csv",
        results = "results/{scenario}/gsa/results/{mode}.csv"
    output: 
        csv = "results/{scenario}/gsa/SA/{mode}.csv",
        png = "results/{scenario}/gsa/SA/{mode}.png"
    log: 
        "logs/calculate_sa/{scenario}_{mode}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/calculate_sa/{scenario}_{mode}.txt"
    group:
        "results"
    script: 
        "../scripts/calculate_sa.py"

rule heatmap:
    message:
        "Generating heat map"
    input:
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
        "../scripts/heatmap.py"

rule barplot:
    message:
        "Generating barplot"
    input:
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
        "../scripts/barplot.py"

###
# UA RULES
###

rule parse_ua_results:
    message: "Parsing results by results file"
    params:
        base_dir = "results/{scenario}/ua/results/"
    input:
        results = "results/{scenario}/ua/results/all.csv"
    output:
        expand("results/{{scenario}}/ua/results/{name}.csv", name=UA_RESULT_FILES)
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
        csvs = get_ua_scatterplot_csvs,
        results = "results/{scenario}/ua/plots.csv"
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