rule copy_network:
    message: "Copying base network"
    input:
        n = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        n = "results/{scenario}/base.nc"
    shell:
        "cp {input.n} {output.n}"

rule copy_pop_layout:
    message: "Copying population layout"
    input:
        csv = f"config/pypsa-usa/{config['pypsa_usa']['pop_layout']}"
    output:
        csv = "results/{scenario}/constraints/pop_layout.csv"
    shell:
        "cp {input.csv} {output.csv}"

rule process_reeds_policy:
    message: "Copying ReEDS {wildcards.policy} file"
    wildcard_constraints:
        policy="ces|rps"
    input:
        policy = "resources/reeds/{policy}_fraction.csv"
    output:
        policy = "results/{scenario}/constraints/{policy}.csv",
    script:
        "../scripts/rps.py"

rule retrieve_natural_gas_data:
    message: "Retrieving import/export natural gas data"
    params:
        api = config["api"]["eia"],
        year = config["pypsa_usa"]["era5_year"]
    output:
        domestic = "resources/natural_gas/domestic.csv",
        international = "resources/natural_gas/international.csv"
    log: "logs/retrieve_natural_gas_data.log"
    script:
        "../scripts/retrieve_ng_data.py"

rule process_tct_data:
    message: "Generating TCT data based on the AEO"
    params:
        aeo_tct = config["pypsa_usa"]["aeo_tct"]
    input:
        network = "results/{scenario}/base.nc"
    output:
        tct = temp("results/{scenario}/constraints/tct_aeo.csv")
        tct_params = temp("results/{scenario}/tct_params.csv")
    script:
        "../scripts/tct.py"

rule process_tct_constraint:
    message: "Grouping custom and AEO TCT data"
    input:
        custom = "resources/policy/technology_limits.csv"
        aeo = "results/{scenario}/constraints/tct_aeo.csv"
    output:
        tct = "results/{scenario}/constraints/tct.csv"
    run:
        import pandas as pd
        tct_base = pd.read_csv(input.custom)
        tct_aeo = pd.read_csv(input.aeo)
        df = pd.concat([tct_base, tct_aeo], axis=0)
        assert len(df.name.unique()) == 1
        df.to_csv(output.tct, index=False)

rule process_tct_gsa_params:
    message: "Processing TCT params generated for the GSA"
    params:
        parameters=config["gsa"]["parameters"]
    input:
        tct = "results/{scenario}/tct_params.csv"
    output:
        params = temp("results/{scenario}/params_merged.csv")
    run:
        import pandas as pd
        base = pd.read_csv(params.parameters)
        tct = pd.read_csv(input.tct)
        df = pd.concat([tct_base, tct_aeo], axis=0)
        df.to_csv(output.params, index=False)

rule sanitize_parameters:
    message: "Sanitizing parameters"
    params:
        parameters=config["gsa"]["parameters"]
    input:
        parameters="results/{scenario}/params_merged.csv"
    output:
        parameters="results/{scenario}/parameters.csv"
    log: "logs/sanitize_{scenario}_parameters.log"
    script:
        "../scripts/sanitize_params.py"

# checkpoint needed for heatmap input function
checkpoint sanitize_results:
    message: "Sanitizing results"
    params:
        results=config["gsa"]["results"]
    input:
        network = "results/{scenario}/base.nc"
    output:
        results="results/{scenario}/results.csv"
    log: "logs/sanitize_{scenario}_results.log"
    script:
        "../scripts/sanitize_results.py"

rule filter_constraint_files:
    message: "Filtering constraint files"
    input:
        network = "results/{scenario}/base.nc",
        ng_domestic = "resources/natural_gas/domestic.csv",
        ng_international = "resources/natural_gas/international.csv"
    output:
        ng_domestic = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international = "results/{scenario}/constraints/ng_international.csv",
    script:
        "../scripts/filter_constraints.py"

rule testing:
    input:
        "results/Testing/constraints/ng_domestic.csv"
        # "results/Testing/sample.csv"