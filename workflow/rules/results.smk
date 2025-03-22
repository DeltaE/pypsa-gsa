"""Rules for processing results"""

def get_sample_file(wildcards):
    if config["gsa"]["scale"]:
        return f"results/{wildcards.scenario}/sample_scaled.csv"
    else: 
        return f"results/{wildcards.scenario}/sample.csv"

def get_heatmap_csvs(wildcards):
    csv = checkpoints.sanitize_results.get(scenario=wildcards.scenario).output[0]
    df = pd.read_csv(csv)
    df = df[df.heatmap == wildcards.group]
    results = df.name.to_list()

    return [f"results/{wildcards.scenario}/SA/{x}.csv" for x in results]

rule extract_results:
    message: "Extracting result"
    input:
        network = "results/{scenario}/modelruns/{run}/network.nc",
        results = "results/{scenario}/results.csv"
    output:
        csv = "results/{scenario}/modelruns/{run}/results.csv"
    log: 
        "logs/extract_results/{scenario}_{run}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/extract_results/{scenario}_{run}.txt"
    script:
        "../scripts/extract_results.py"

rule combine_results:
    message: "Collapsing all results into single summary file"
    input:
        results = expand("results/{{scenario}}/modelruns/{run}/results.csv", run=MODELRUNS)
    output:
        csv = temp("results/{scenario}/results/all.csv")
    log: 
        "logs/combine_results/{scenario}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/combine_results/{scenario}.txt"
    run:
        import pandas as pd
        data = [pd.read_csv(str(x)) for x in input.results]
        df = pd.concat(data)
        df.to_csv(output.csv, index=False)

rule parse_results:
    message: "Parsing results by results file"
    params:
        base_dir = "results/{scenario}/results/"
    input:
        results = "results/{scenario}/results/all.csv"
    output:
        expand("results/{{scenario}}/results/{result}.csv", result=RESULT_FILES)
    log: 
        "logs/parse_results/{scenario}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/parse_results/{scenario}.txt"
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
        parameters = "results/{scenario}/parameters.csv",
        results = "results/{scenario}/results/{result}.csv"
    output: 
        csv = "results/{scenario}/SA/{result}.csv",
        png = "results/{scenario}/SA/{result}.png"
    log: 
        "logs/calculate_sa/{scenario}_{result}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 200),
        runtime=1
    benchmark:
        "benchmarks/calculate_sa/{scenario}_{result}.txt"
    script: 
        "../scripts/calculate_sa.py"

rule heatmap:
    message:
        "Generating heat map"
    input:
        results = "results/{scenario}/results.csv",
        csvs = get_heatmap_csvs
    output:
        heatmap = "results/{scenario}/heatmaps/{group}.png"
    log: 
        "logs/create_heatmap/{scenario}_{group}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 500),
        runtime=1
    benchmark:
        "benchmarks/create_heatmap/{scenario}_{group}.txt"
    script:
        "../scripts/heatmap.py"