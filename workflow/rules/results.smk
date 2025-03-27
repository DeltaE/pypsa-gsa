"""Rules for processing results"""

def get_sample_file(wildcards):
    if config["gsa"]["scale"]:
        return f"results/{wildcards.scenario}/gsa/sample_scaled.csv"
    else: 
        return f"results/{wildcards.scenario}/gsa/sample.csv"

def get_plotting_csvs(wildcards):
    csv = checkpoints.sanitize_results.get(scenario=wildcards.scenario).output[0]
    df = pd.read_csv(csv)
    df = df[df.plots.str.contains(wildcards.plot)]
    results = df.name.to_list()

    return [f"results/{wildcards.scenario}/SA/{x}.csv" for x in results]

rule extract_results:
    message: "Extracting result"
    input:
        network = "results/{scenario}/gsa/modelruns/{run}/network.nc",
        results = "results/{scenario}/gsa/results.csv"
    output:
        csv = "results/{scenario}/gsa/modelruns/{run}/results.csv"
    log: 
        "logs/extract_results/{scenario}_{run}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/extract_results/{scenario}_{run}.txt"
    group:
        "solve_{scenario}_{run}"
    script:
        "../scripts/extract_results.py"

rule combine_results:
    message: "Collapsing all results into single summary file"
    input:
        results = expand("results/{{scenario}}/gsa/modelruns/{run}/results.csv", run=MODELRUNS)
    output:
        csv = temp("results/{scenario}/gsa/results/all.csv")
    log: 
        "logs/combine_results/{scenario}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/combine_results/{scenario}.txt"
    group:
        "results"
    run:
        import pandas as pd
        data = [pd.read_csv(str(x)) for x in input.results]
        df = pd.concat(data)
        df.to_csv(output.csv, index=False)

rule parse_results:
    message: "Parsing results by results file"
    params:
        base_dir = "results/{scenario}/gsa/results/"
    input:
        results = "results/{scenario}/gsa/results/all.csv"
    output:
        expand("results/{{scenario}}/gsa/results/{result}.csv", result=RESULT_FILES)
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
        csvs = get_plotting_csvs
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
        csvs = get_plotting_csvs
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