rule copy_network:
    message: "Copying base network"
    input:
        n = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        n = "results/{scenario}/base.nc"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 100),
        runtime=1
    benchmark:
        "benchmarks/copy_network/{scenario}.txt"
    shell:
        "cp {input.n} {output.n}"


rule copy_pop_layout:
    message: "Copying population layout"
    input:
        csv = f"config/pypsa-usa/{config['pypsa_usa']['pop_layout']}"
    output:
        csv = "results/{scenario}/constraints/pop_layout.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 100),
        runtime=1
    benchmark:
        "benchmarks/copy_pop_layout/{scenario}.txt"
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
    benchmark:
        "benchmarks/process_reeds/{scenario}_{policy}.txt"
    resources:
        mem_mb=100,
        runtime=1
    script:
        "../scripts/rps.py"
        

rule copy_tct_data:
    message: "Copying TCT data"
    input:
        csv="resources/policy/technology_limits.csv"
    output:
        csv="results/{scenario}/constraints/tct.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 100),
        runtime=1
    shell:
        "cp {input.csv} {output.csv}"

rule copy_ev_policy_data:
    message: "Copying EV Policy data"
    input:
        csv="resources/policy/ev_policy.csv"
    output:
        csv="results/{scenario}/constraints/ev_policy.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 100),
        runtime=1
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
    log: 
        "logs/retrieve_ng/benchmark.log"
    benchmark:
        "benchmarks/retrieve_ng/benchmark.txt"
    retries: 3
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 100),
        runtime=2
    script:
        "../scripts/retrieve_ng_data.py"

rule sanitize_parameters:
    message: "Sanitizing parameters"
    input:
        parameters=config["gsa"]["parameters"]
    output:
        parameters="results/{scenario}/parameters.csv"
    log: 
        "logs/sanitize_parameters/{scenario}.log"
    benchmark:
        "benchmarks/sanitize_parameters/{scenario}.txt"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
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
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 300),
        runtime=1
    benchmark:
        "benchmarks/sanitize_results/{scenario}.txt"
    log: 
        "logs/sanitize_results/{scenario}.log"
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
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 300),
        runtime=1
    benchmark:
        "benchmarks/process_ng/{scenario}.txt"
    log: 
        "logs/process_ng/{scenario}.log"
    script:
        "../scripts/process_ng.py"