"""Rules that generate data, which are not used in the main GSA/UA workflow"""

rule generate_tct_data:
    message: "Generating TCT data based on the AEO"
    params:
        include_tct = config["generated"]["include"].get("tct", {}),
        ccs_limit = config["generated"]["limits"]["ccgtccs_max"]
    input:
        network = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        tct_aeo = "results/{scenario}/generated/tct_aeo.csv",
        tct_gsa = "results/{scenario}/generated/tct_gsa.csv",
    resources:
        mem_mb_per_cpu=1000,
        runtime=5
    # group:
    #     "generate_data"
    script:
        "../scripts/process_tct.py"

rule retrieve_co2L_data:
    message: "Retrieving State level CO2 data."
    params:
        api = config["api"]["eia"],
    output:
        co2_2005 = "resources/emissions/co2_2005.csv",
        co2_2030 = "resources/emissions/co2_2005_50pct.csv",
    resources:
        mem_mb_per_cpu=1000,
        runtime=5
    # group:
    #     "generate_data"
    script:
        "../scripts/retrieve_co2L.py"

rule generate_co2L_data:
    message: "Generating CO2L data based on user inputs"
    params:
        include_co2L = config["generated"]["include"].get("co2L", {}),
        min_value = config["generated"]["limits"]["co2L_min"],
        max_value = config["generated"]["limits"]["co2L_max"],
    input:
        network =  f"config/pypsa-usa/{config['pypsa_usa']['network']}",
        co2_2005 = "resources/emissions/co2_2005.csv",
    output:
        co2_gsa = "results/{scenario}/generated/co2L_gsa.csv"
    resources:
        mem_mb_per_cpu=1000,
        runtime=5
    # group:
    #     "generate_data"
    script:
        "../scripts/process_co2L.py"

rule append_generated_parameters:
    message: "Appending generated GSA parameters."
    input:
        base=config["gsa"]["parameters"],
        tct="results/{scenario}/generated/tct_gsa.csv",
        co2L="results/{scenario}/generated/co2L_gsa.csv"
    output:
        csv=expand("results/{{scenario}}/generated/{param_f}", param_f = config["gsa"]["parameters"])
    resources:
        mem_mb_per_cpu=1000,
        runtime=5
    # group:
    #     "generate_data"
    run:
        import pandas as pd
        dfs = []
        base = pd.read_csv(input.base)
        dfs.append(base) 
        dfs.append(pd.read_csv(input.tct))
        dfs.append(pd.read_csv(input.co2L))
        df = pd.concat(dfs)
        df.to_csv(output.csv[0], index=False)

def get_input_parameters_file(wildards):
    return f"results/{wildards.scenario}/generated/{config['gsa']['parameters']}"

def get_raw_result_path(wildards):
    if wildards.mode == "gsa":
        return config["gsa"]["results"]
    elif wildards.mode == "ua":
         return config["uncertainity"]["results"]
    else:
        raise ValueError(f"Invalid input {wildards.mode} for raw result path.")

rule sanitize_parameters:
    message: "Sanitizing parameters"
    input:
        parameters=get_input_parameters_file,
        network = "results/{scenario}/base.nc",
        rps = "results/{scenario}/constraints/rps.csv",
        ces = "results/{scenario}/constraints/ces.csv",
    output:
        parameters="results/{scenario}/gsa/parameters.csv"
    log: 
        "logs/sanitize_parameters/{scenario}.log"
    benchmark:
        "benchmarks/sanitize_parameters/{scenario}.txt"
    resources:
        mem_mb_per_cpu=1000,
        runtime=5
    # group:
    #     "generate_data"
    script:
        "../scripts/sanitize_params.py"

rule sanitize_results:
    message: "Sanitizing results"
    wildcard_constraints:
        mode="gsa|ua"
    params:
        results=get_raw_result_path
    input:
        network = "results/{scenario}/base.nc"
    output:
        results="results/{scenario}/{mode}/results.csv"
    resources:
        mem_mb_per_cpu=1000,
        runtime=5
    benchmark:
        "benchmarks/sanitize_results/{scenario}_{mode}.txt"
    log: 
        "logs/sanitize_results/{scenario}_{mode}.log"
    # group:
    #     "generate_data"
    script:
        "../scripts/sanitize_results.py"
    
