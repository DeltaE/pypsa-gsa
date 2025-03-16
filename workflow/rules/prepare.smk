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

rule copy_tct_data:
    message: "Copying TCT data"
    input:
        csv="resources/policy/technology_limits.csv"
    output:
        csv="results/{scenario}/constraints/tct.csv"
    shell:
        "cp {input.csv} {output.csv}"

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

rule generate_tct_data:
    message: "Generating TCT data based on the AEO"
    params:
        network = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        tct_aeo = "resources/generated/tct_aeo.csv",
        tct_gsa = "resources/generated/tct_gsa.csv",
    script:
        "../scripts/tct.py"

rule sanitize_parameters:
    message: "Sanitizing parameters"
    params:
        parameters=config["gsa"]["parameters"]
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