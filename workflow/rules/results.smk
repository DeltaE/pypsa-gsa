"""Rules for processing results"""

def get_sample_name(wildcards):
    if config["gsa"]["scale"]:
        return f"results/{wildcards.scenario}/sample_scaled.csv"
    else: 
        return f"results/{wildcards.scenario}/sample.csv"

def get_heatmap_csvs(wildcards):
    df = pd.read_csv(f"results/{wildcards.scenario}/results.csv")
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
    script:
        "../scripts/extract_results.py"

rule combine_results:
    message: "Collapsing all results into single summary file"
    input:
        results = expand("results/{{scenario}}/modelruns/{run}/results.csv", run=MODELRUNS)
    output:
        csv = temp("results/{scenario}/results/all.csv")
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
        sample = get_sample_name,
        parameters = "results/{scenario}/parameters.csv",
        results = "results/{scenario}/results/{result}.csv"
    output: 
        csv = "results/{scenario}/SA/{result}.csv",
        png = "results/{scenario}/SA/{result}.png"
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
    script:
        "../scripts/heatmap.py"