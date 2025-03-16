rule copy_network:
    message: "Copying base network"
    input:
        n = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        n = "results/{scenario}/base.nc"
    resources:
        mem_mb=1000,
        runtime=2
    shell:
        "cp {input.n} {output.n}"


rule copy_pop_layout:
    message: "Copying population layout"
    input:
        csv = f"config/pypsa-usa/{config['pypsa_usa']['pop_layout']}"
    output:
        csv = "results/{scenario}/constraints/pop_layout.csv"
    resources:
        mem_mb=1000,
        runtime=2
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
    resources:
        mem_mb=1000,
        runtime=5
    script:
        "../scripts/rps.py"
        

rule copy_tct_data:
    message: "Copying TCT data"
    input:
        csv="resources/policy/technology_limits.csv"
    output:
        csv="results/{scenario}/constraints/tct.csv"
    resources:
        mem_mb=1000,
        runtime=2
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
    retries: 3
    resources:
        mem_mb=2000,
        runtime=5
    script:
        "../scripts/retrieve_ng_data.py"

# this rule is not part of the actual workflow! 
rule generate_tct_data:
    message: "Generating TCT data based on the AEO"
    params:
        network = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        tct_aeo = "resources/generated/tct_aeo.csv",
        tct_gsa = "resources/generated/tct_gsa.csv",
    resources:
        mem_mb=1000,
        runtime=2
    script:
        "../scripts/tct.py"

rule sanitize_parameters:
    message: "Sanitizing parameters"
    params:
        parameters=config["gsa"]["parameters"]
    output:
        parameters="results/{scenario}/parameters.csv"
    log: "logs/sanitize_{scenario}_parameters.log"
    resources:
        mem_mb=1000,
        runtime=2
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
    resources:
        mem_mb=1000,
        runtime=2
    log: "logs/sanitize_{scenario}_results.log"
    script:
        "../scripts/sanitize_results.py"

rule process_natural_gas:
    message: "Filtering constraint files"
    input:
        network = "results/{scenario}/base.nc",
        ng_domestic = "resources/natural_gas/domestic.csv",
        ng_international = "resources/natural_gas/international.csv"
    output:
        ng_domestic = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international = "results/{scenario}/constraints/ng_international.csv",
    resources:
        mem_mb=lambda wc, input: max(1.5 * input.size_mb, 1000),
        runtime=2
    script:
        "../scripts/process_ng.py"