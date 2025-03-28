def get_sampling_method(wildards):
    if wildards.result == "gsa":
        return "morris"
    elif wildards.result == "ua":
        return config["uncertainity"]["sample"]
    else:
        raise ValueError(f"{wildards.result} is not valid for selecing a sampling method.")

def get_replicates(wildards):
    if wildards.result == "gsa":
        return config["gsa"]["replicates"]
    elif wildards.result == "ua":
        return config["uncertainity"]["replicates"]
    else:
        raise ValueError(f"{wildards.result} is not valid for selecing replicates.")

def get_set_values(wildards):
    """For the uncertainity analysis, we lock a bunch of values to the average of the unceratinity range."""
    if wildards.result == "gsa":
        return []
    elif wildards.result == "ua":
        return "results/{scenario}/ua/set_values.csv"
    else:
        raise ValueError(f"{wildards.result} is not valid for set values.")

rule create_sample:
    message: "Creating sample"
    wildcard_constraints:
        result="gsa|ua"
    params:
        replicates=get_replicates,
        method=get_sampling_method
    input:
        # result is 'gsa' or 'uncertainity'
        parameters="results/{scenario}/{result}/parameters.csv"
    output: 
        sample_file = "results/{scenario}/{result}/sample.csv",
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    log: 
        "logs/create_sample/{scenario}_{result}.log"
    benchmark:
        "benchmarks/create_sample/{scenario}_{result}.txt"
    group:
        "create_sample"
    script:
        "../scripts/create_sample.py"

# Apply sample creates all samples, rather than one sample at a time to prevent the 
# need of reding in the base network many times. This is a little hacky as the 'run' variable 
# the the output should really be a wildcard to follow snakemake principals. 

# due to this, we have to break this into two rules, as there is no way to access the add 
# logic to determine the modelruns variable in the output :( 

rule apply_gsa_sample_to_network:
    message: "Applying sample"
    wildcard_constraints:
        result="gsa"
    params:
        root_dir = "results/{scenario}/gsa/modelruns/",
        meta_yaml = config["metadata"]["yaml"],
        meta_csv = config["metadata"]["csv"],
    input: 
        parameters = "results/{scenario}/{result}/parameters.csv",
        set_values_file = get_set_values,
        sample_file = "results/{scenario}/{result}/sample.csv",
        network = "results/{scenario}/base.nc",
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
        ev_policy_f = "results/{scenario}/constraints/ev_policy.csv"
    output:
        n = temp(expand("results/{{scenario}}/{{result}}/modelruns/{run}/n.nc", run=GSA_MODELRUNS)),
        scaled_sample = "results/{scenario}/{result}/sample_scaled.csv",
        meta_constriant = expand("results/{{scenario}}/{{result}}/modelruns/{run}/constraints.csv", run=GSA_MODELRUNS)
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 600),
        runtime=1
    benchmark:
        "benchmarks/apply_sample/{scenario}_{result}.txt"
    group:
        "apply_sample"
    log: 
        "logs/apply_sample/{scenario}_{result}.log"
    script:
        "../scripts/apply_sample.py"

rule apply_ua_sample_to_network:
    message: "Applying sample"
    wildcard_constraints:
        result="ua"
    params:
        root_dir = "results/{scenario}/gsa/modelruns/",
        meta_yaml = config["metadata"]["yaml"],
        meta_csv = config["metadata"]["csv"],
    input: 
        parameters = "results/{scenario}/{result}/parameters.csv",
        set_values_file = get_set_values,
        sample_file = "results/{scenario}/{result}/sample.csv",
        network = "results/{scenario}/base.nc",
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
        ev_policy_f = "results/{scenario}/constraints/ev_policy.csv"
    output:
        n = temp(expand("results/{{scenario}}/{{result}}/modelruns/{run}/n.nc", run=UA_MODELRUNS)),
        scaled_sample = "results/{scenario}/{result}/sample_scaled.csv",
        meta_constriant = expand("results/{{scenario}}/{{result}}/modelruns/{run}/constraints.csv", run=UA_MODELRUNS)
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 600),
        runtime=1
    benchmark:
        "benchmarks/apply_sample/{scenario}_{result}.txt"
    group:
        "apply_sample"
    log: 
        "logs/apply_sample/{scenario}_{result}.log"
    script:
        "../scripts/apply_sample.py"