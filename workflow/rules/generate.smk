"""Rules that generate data, which are not used in the main GSA workflow"""

rule generate_tct_data:
    message: "Generating TCT data based on the AEO"
    params:
        network = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        tct_aeo = "resources/generated/tct_aeo.csv",
        tct_gsa = "resources/generated/tct_gsa.csv",
    resources:
        mem_mb=100,
        runtime=3
    group:
        "generate_data"
    script:
        "../scripts/tct.py"

rule generate_co2L_data:
    message: "Generating CO2L data based on user inputs"
    params:
        api = config["api"]["eia"],
        min_value = config["generated"]["co2L_min"],
        max_value = config["generated"]["co2L_max"],
    input:
        network =  f"config/pypsa-usa/{config['pypsa_usa']['network']}",
    output:
        co2_2005 = "resources/policy/co2_2005.csv",
        co2_2030 = "resources/policy/co2_2030.csv",
        co2_gsa = "resources/generated/co2L_gsa.csv"
    resources:
        mem_mb=200,
        runtime=3
    group:
        "generate_data"
    script:
        "../scripts/co2L.py"
