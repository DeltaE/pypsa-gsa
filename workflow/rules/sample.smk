def get_sampling_method(wildards):
    if wildards.mode == "gsa":
        return "morris"
    elif wildards.mode == "ua":
        return config["uncertainity"]["sample"]
    else:
        raise ValueError(f"{wildards.mode} is not valid for selecing a sampling method.")

def get_replicates(wildards):
    if wildards.mode == "gsa":
        return config["gsa"]["replicates"]
    elif wildards.mode == "ua":
        return config["uncertainity"]["replicates"]
    else:
        raise ValueError(f"{wildards.mode} is not valid for selecing replicates.")

def get_static_values(wildards):
    """For the uncertainity analysis, we lock a bunch of values to the average of the unceratinity range."""
    if wildards.mode == "gsa":
        return []
    elif wildards.mode == "ua":
        return "results/{scenario}/ua/set_values.csv"
    else:
        raise ValueError(f"{wildards.mode} is not valid for static values.")

rule create_sample:
    message: "Creating sample"
    wildcard_constraints:
        mode="gsa|ua"
    params:
        replicates=get_replicates,
        method=get_sampling_method
    input:
        parameters="results/{scenario}/{mode}/parameters.csv"
    output: 
        sample_file = "results/{scenario}/{mode}/sample.csv",
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    log: 
        "logs/create_sample/{scenario}_{mode}.log"
    benchmark:
        "benchmarks/create_sample/{scenario}_{mode}.txt"
    group:
        "create_and_apply_sample"
    script:
        "../scripts/create_sample.py"

# Apply sample creates all samples, rather than one sample at a time to prevent the 
# need of reding in the base network many times. This is a little hacky as the 'run' variable 
# the the output should really be a wildcard to follow snakemake principals. 


checkpoint apply_gsa_sample_to_network:
    message: "Applying sample"
    wildcard_constraints:
        mode="gsa"
    params:
        root_dir = "results/{scenario}/{mode}/modelruns/",
        meta_yaml = config["metadata"]["yaml"],
        meta_csv = config["metadata"]["csv"],
        testing = False,
    input: 
        parameters = "results/{scenario}/{mode}/parameters.csv",
        set_values_file = get_static_values,
        sample_file = "results/{scenario}/{mode}/sample.csv",
        network = "results/{scenario}/base.nc",
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
        ev_policy_f = "results/{scenario}/constraints/ev_policy.csv",
        elec_trade_f = "results/{scenario}/constraints/import_export_flows.csv"
    output:
        n = temp(expand("results/{{scenario}}/{{mode}}/modelruns/{run}/n.nc", run=get_gsa_modelruns())),
        scaled_sample = "results/{scenario}/{mode}/sample_scaled.csv",
        meta_constriant = expand("results/{{scenario}}/{{mode}}/modelruns/{run}/constraints.csv", run=get_gsa_modelruns())
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 600),
        runtime=1
    benchmark:
        "benchmarks/apply_sample/{scenario}_{mode}.txt"
    group:
        "create_and_apply_sample"
    log: 
        "logs/apply_sample/{scenario}_{mode}.log"
    script:
        "../scripts/apply_sample.py"

checkpoint apply_ua_sample_to_network:
    message: "Applying sample"
    wildcard_constraints:
        mode="ua"
    params:
        root_dir = "results/{scenario}/{mode}/modelruns/",
        meta_yaml = config["metadata"]["yaml"],
        meta_csv = config["metadata"]["csv"],
        testing = False,
    input: 
        parameters = "results/{scenario}/{mode}/parameters.csv",
        set_values_file = get_static_values,
        sample_file = "results/{scenario}/{mode}/sample.csv",
        network = "results/{scenario}/base.nc",
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
        ev_policy_f = "results/{scenario}/constraints/ev_policy.csv",
        elec_trade_f = "results/{scenario}/constraints/import_export_flows.csv"
    output:
        n = temp(expand("results/{{scenario}}/{{mode}}/modelruns/{run}/n.nc", run=get_ua_modelruns())),
        scaled_sample = "results/{scenario}/{mode}/sample_scaled.csv",
        meta_constriant = expand("results/{{scenario}}/{{mode}}/modelruns/{run}/constraints.csv", run=get_ua_modelruns())
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 600),
        runtime=1
    benchmark:
        "benchmarks/apply_sample/{scenario}_{mode}.txt"
    group:
        "create_and_apply_sample"
    log: 
        "logs/apply_sample/{scenario}_{mode}.log"
    script:
        "../scripts/apply_sample.py"

rule test_apply_gsa_sample_to_network:
    message: "Applying test sample"
    wildcard_constraints:
        mode="gsa"
    params:
        root_dir = "results/{scenario}/{mode}/modelruns/",
        meta_yaml = config["metadata"]["yaml"],
        meta_csv = config["metadata"]["csv"],
        testing = True
    input: 
        parameters = "results/{scenario}/{mode}/parameters.csv",
        set_values_file = get_static_values,
        sample_file = "results/{scenario}/{mode}/sample.csv",
        network = "results/{scenario}/base.nc",
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
        ev_policy_f = "results/{scenario}/constraints/ev_policy.csv",
        elec_trade_f = "results/{scenario}/constraints/import_export_flows.csv"
    output:
        n = "results/{scenario}/{mode}/modelruns/testing/0/n.nc",
        scaled_sample = "results/{scenario}/{mode}/testing/sample_scaled.csv",
        meta_constriant = "results/{scenario}/{mode}/modelruns/testing/0/constraints.csv",
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 600),
        runtime=1
    benchmark:
        "benchmarks/apply_sample/{scenario}_{mode}.txt"
    group:
        "{scenario}_{mode}_testing"
    log: 
        "logs/apply_sample/{scenario}_{mode}.log"
    script:
        "../scripts/apply_sample.py"