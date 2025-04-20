"""Rules that generate data, which are not used in the main GSA workflow"""

rule generate_tct_data:
    message: "Generating TCT data based on the AEO"
    params:
        network = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        tct_aeo = "results/{scenario}/generated/tct_aeo.csv",
        tct_gsa = "results/{scenario}/generated/tct_gsa.csv",
    resources:
        mem_mb=100,
        runtime=3
    group:
        "generate_data"
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
        mem_mb=200,
        runtime=3
    group:
        "generate_data"
    script:
        "../scripts/retrieve_co2L.py"

rule generate_co2L_data:
    message: "Generating CO2L data based on user inputs"
    params:
        min_value = config["generated"]["co2L_min"],
        max_value = config["generated"]["co2L_max"],
    input:
        network =  f"config/pypsa-usa/{config['pypsa_usa']['network']}",
        co2_2005 = "resources/emissions/co2_2005.csv",
    output:
        co2_gsa = "results/{scenario}/generated/co2L_gsa.csv"
    resources:
        mem_mb=200,
        runtime=3
    group:
        "generate_data"
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
        mem_mb=200,
        runtime=3
    group:
        "generate_data"
    run:
        import pandas as pd
        base = pd.read_csv(input.base)
        tct = pd.read_csv(input.tct)
        co2L = pd.read_csv(input.co2L)
        df = pd.concat([base, tct, co2L])
        df.to_csv(output.csv[0], index=False)
    