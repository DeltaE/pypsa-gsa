rule copy_network:
    message: "Copying base network"
    input:
        n = f"config/pypsa-usa/{config['pypsa_usa']['network']}"
    output:
        n = temp("results/{scenario}/copy.nc")
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 100),
        runtime=1
    group:
        "prepare_data"
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
    group:
        "prepare_data"
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
    resources:
        mem_mb=100,
        runtime=1
    group:
        "prepare_data"
    benchmark:
        "benchmarks/process_reeds/{scenario}_{policy}.txt"
    script:
        "../scripts/process_rps.py"
        
def get_extra_tct_data(wildards):
    if config["scenario"]["include_generated"]:
        return "results/{scenario}/generated/tct_aeo.csv"
    else:
        return []

rule copy_tct_data:
    message: "Copying TCT data"
    input:
        base="resources/policy/technology_limits.csv",
        extras=get_extra_tct_data
    output:
        csv="results/{scenario}/constraints/tct.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 100),
        runtime=1
    group:
        "prepare_data"
    run:
        import shutil
        import pandas as pd

        if input.extras:
            base = pd.read_csv(input.base)
            extras = pd.read_csv(input.extras)
            df = pd.concat([base, extras])
            df.to_csv(output.csv)
        else:
            shutil.copy(input.base, output.csv)

rule copy_ev_policy_data:
    message: "Copying EV Policy data"
    input:
        csv="resources/policy/ev_policy.csv"
    output:
        csv="results/{scenario}/constraints/ev_policy.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 100),
        runtime=1
    group:
        "prepare_data"
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
    group:
        "prepare_data"
    script:
        "../scripts/retrieve_ng_data.py"

def get_input_parameters_file(wildards):
    if config["scenario"]["include_generated"]:
        return f"results/{wildards.scenario}/generated/{config['gsa']['parameters']}"
    else:
        return config["gsa"]["parameters"]

rule sanitize_parameters:
    message: "Sanitizing parameters"
    input:
        parameters=get_input_parameters_file
    output:
        parameters="results/{scenario}/gsa/parameters.csv"
    log: 
        "logs/sanitize_parameters/{scenario}.log"
    benchmark:
        "benchmarks/sanitize_parameters/{scenario}.txt"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    group:
        "prepare_data"
    script:
        "../scripts/sanitize_params.py"


def get_raw_result_path(wildards):
    if wildards.mode == "gsa":
        return config["gsa"]["results"]
    elif wildards.mode == "ua":
         return config["uncertainity"]["results"]
    else:
        raise ValueError(f"Invalid input {wildards.mode} for raw result path.")


checkpoint sanitize_results:
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
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 300),
        runtime=1
    benchmark:
        "benchmarks/sanitize_results/{scenario}_{mode}.txt"
    log: 
        "logs/sanitize_results/{scenario}_{mode}.log"
    group:
        "prepare_data"
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
    group:
        "prepare_data"
    script:
        "../scripts/process_ng.py"

rule process_interchange_data:
    message: "Processing import/export data"
    input:
        network = "results/{scenario}/copy.nc", # use copy to avoid cyclic dependency
        regions = "resources/interchanges/regions.csv",
        membership = "resources/interchanges/membership.csv",
        flowgates = "resources/interchanges/transmission_capacity_init_AC_ba_NARIS2024.csv",
    params:
        api = config["api"]["eia"],
        year = config["pypsa_usa"]["era5_year"],
        balancing_period = "month", # only one supported right now
        pudl_path = "s3://pudl.catalyst.coop/v2025.2.0"
    output:
        net_flows = "results/{scenario}/constraints/import_export_flows.csv",
        capacities = "results/{scenario}/constraints/import_export_capacity.csv",
        costs = "results/{scenario}/constraints/import_export_costs.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 5000),
        runtime=3
    benchmark:
        "benchmarks/process_interchanges/{scenario}.txt"
    log: 
        "logs/process_interchanges/{scenario}.log"
    group:
        "prepare_data"
    script:
        "../scripts/process_imports_exports.py"

rule add_import_export_to_network:
    message: "Adding import/export to network"
    input:
        network = "results/{scenario}/copy.nc", # base network will include imports/exports
        capacities_f = "results/{scenario}/constraints/import_export_capacity.csv",
        elec_costs_f = "results/{scenario}/constraints/import_export_costs.csv"
    output:
        network = "results/{scenario}/base.nc",
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 500),
        runtime=1
    group:
        "prepare_data"
    script:
        "../scripts/apply_import_export.py"

# for the uncertainity propogation
rule prepare_static_values:
    message: "Setting static paramters for the uncertainity."
    params:
        to_remove=config["uncertainity"]["parameters"]
    input:
        # use the gsa file as its been sanitized
        parameters="results/{scenario}/gsa/parameters.csv"
    output:
        parameters="results/{scenario}/ua/set_values.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 300),
        runtime=1
    benchmark:
        "benchmarks/prepare_set_values/{scenario}.txt"
    log: 
        "logs/prepare_set_values/{scenario}.log"
    group:
        "prepare_data"
    script:
        "../scripts/prepare_static_values.py"


rule prepare_ua_params:
    message: "Getting parameters for the uncertainity sample."
    params:
        to_sample=config["uncertainity"]["parameters"]
    input:
        # use the gsa file as its been sanitized
        parameters="results/{scenario}/gsa/parameters.csv"
    output:
        parameters="results/{scenario}/ua/parameters.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 300),
        runtime=1
    benchmark:
        "benchmarks/prepare_ua_params/{scenario}.txt"
    log: 
        "logs/prepare_ua_params/{scenario}.log"
    group:
        "prepare_data"
    run:
        import pandas as pd
        df = pd.read_csv(input.parameters)
        df = df[df.name.isin(params.to_sample)]
        assert len(df) == len(params.to_sample)
        df.to_csv(output.parameters, index=False)


checkpoint sanitize_ua_plot_params:
    message: "Sanitizing uncertainity analysis plotting parameters."
    input:
        plots=config["uncertainity"]["plots"],
        results="results/{scenario}/ua/results.csv"
    output:
        plots="results/{scenario}/ua/plots.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 300),
        runtime=1
    benchmark:
        "benchmarks/prepare_ua_params/{scenario}.txt"
    log: 
        "logs/prepare_ua_params/{scenario}.log"
    group:
        "prepare_data"
    script:
        "../scripts/sanitize_ua_plot_params.py"