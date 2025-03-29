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
    csv = checkpoints.sanitize_results.get(scenario=wildcards.scenario, result="gsa").output[0]
    df = pd.read_csv(csv)
    df = df[df.plots.str.contains(wildcards.plot)]
    results = df.name.to_list()

    return ["results/{wildcards.scenario}/{result=wildcards.result}/SA/{x}.csv" for x in results]

def get_ua_plotting_csvs(wildcards):
    csv = checkpoints.sanitize_results.get(scenario=wildcards.scenario, result="ua").output[0]
    df = pd.read_csv(csv, index_col=0)
    results = [df.at[snakemake.plot, xaxis], df.at[snakemake.plot, yaxis]]
    return ["results/{wildcards.scenario}/ua/results/{x}.csv" for x in results]

def get_combine_results_inputs(wildards):
    """Need input function as we need to get model run numbers."""
    if wildards.result == "gsa":
        modelruns = GSA_MODELRUNS
    elif wildards.result == "ua":
         modelruns = UA_MODELRUNS
    else:
        raise ValueError(f"Invalid result of {wildards.result} for model runs.")

    return [f"results/{wildcards.scenario}/{wildards.result}/modelruns/{run}/results.csv" for run in modelruns]

###
# SHARED RULES
###

rule extract_results:
    message: "Extracting result"
    wildcard_constraints:
        result="gsa|ua"
    input:
        network = "results/{scenario}/{result}/modelruns/{run}/network.nc",
        results = "results/{scenario}/{result}/results.csv"
    output:
        csv = "results/{scenario}/{result}/modelruns/{run}/results.csv"
    log: 
        "logs/extract_results/{scenario}_{result}_{run}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/extract_gsa_results/{scenario}_{result}_{run}.txt"
    group:
        "solve_{scenario}_{result}_{run}"
    script:
        "../scripts/extract_results.py"

rule combine_results:
    message: "Collapsing all results into single summary file"
    wildcard_constraints:
        result="gsa|ua"
    input:
        results = get_combine_results_inputs
    output:
        csv = temp("results/{scenario}/{result}/results/all.csv")
    log: 
        "logs/combine_results/{scenario}_{result}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/combine_results/{scenario}_{result}.txt"
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
        results = "results/{scenario}/gsa/results/{result}.csv"
    output: 
        csv = "results/{scenario}/gsa/SA/{result}.csv",
        png = "results/{scenario}/gsa/SA/{result}.png"
    log: 
        "logs/calculate_sa/{scenario}_{result}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/calculate_sa/{scenario}_{result}.txt"
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

rule plot_ua:
    message:
        "Generating UA plots"
    input:
        results = "results/{scenario}/ua/plots.csv",
        csvs = get_ua_plotting_csvs
    output:
        plot = "results/{scenario}/ua/plots/{plot}.png"
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
        "../scripts/plot_ua.py"