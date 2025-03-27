def get_sampling_method(wildards):
    if wildards.result == "gsa":
        return "morris"
    elif wildards.result == "uncertainity":
        return config["uncertainity"]["sample"]
    else:
        raise ValueError(f"{wildards.result} is not valid for selecing a sampling method.")

def get_replicates(wildards):
    if wildards.result == "gsa":
        return config["gsa"]["replicates"]
    elif wildards.result == "uncertainity":
        return config["uncertainity"]["replicates"]
    else:
        raise ValueError(f"{wildards.result} is not valid for selecing replicates.")

rule create_sample:
    message: "Creating sample"
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
# need of reding in the base network many times

rule apply_sample_to_network:
    message: "Applying sample"
    params:
        root_dir = "results/{scenario}/gsa/modelruns/",
        meta_yaml = config["metadata"]["yaml"],
        meta_csv = config["metadata"]["csv"]
    input: 
        parameters = "results/{scenario}/gsa/parameters.csv",
        sample_file = "results/{scenario}/gsa/sample.csv",
        network = "results/{scenario}/base.nc",
        pop_layout_f = "results/{scenario}/gsa/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/gsa/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/gsa/constraints/ng_international.csv",
        rps_f = "results/{scenario}/gsa/constraints/rps.csv",
        ces_f = "results/{scenario}/gsa/constraints/ces.csv",
        ev_policy_f = "results/{scenario}/gsa/constraints/ev_policy.csv"
    output:
        n = temp(expand("results/{{scenario}}/gsa/modelruns/{run}/n.nc", run=MODELRUNS)),
        scaled_sample = "results/{scenario}/gsa/sample_scaled.csv",
        meta_constriant = expand("results/{{scenario}}/gsa/modelruns/{run}/constraints.csv", run=MODELRUNS)
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 600),
        runtime=1
    benchmark:
        "benchmarks/apply_sample/{scenario}.txt"
    group:
        "apply_sample"
    log: 
        "logs/apply_sample/{scenario}.log"
    script:
        "../scripts/apply_sample.py"

