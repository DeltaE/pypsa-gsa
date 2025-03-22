rule create_sample:
    message: "Creating sample"
    params:
        replicates=config["gsa"]["replicates"],
    input:
        parameters="results/{scenario}/parameters.csv"
    output: 
        sample_file = "results/{scenario}/sample.csv",
        # scaled_sample_file = "results/{scenario}/sample_scaled.csv"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    log: 
        "logs/create_sample/{scenario}.log"
    benchmark:
        "benchmarks/create_sample/{scenario}.txt"
    script:
        "../scripts/create_sample.py"

# Apply sample creates all samples, rather than one sample at a time to prevent the 
# need of reding in the base network many times

rule apply_sample_to_network:
    message: "Applying sample"
    params:
        root_dir = "results/{scenario}/modelruns/",
        meta_yaml = config["metadata"]["yaml"],
        meta_csv = config["metadata"]["csv"]
    input: 
        parameters = "results/{scenario}/parameters.csv",
        sample_file = "results/{scenario}/sample.csv",
        network = "results/{scenario}/base.nc",
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
    output:
        n = temp(expand("results/{{scenario}}/modelruns/{run}/n.nc", run=MODELRUNS)),
        scaled_sample = "results/{scenario}/sample_scaled.csv",
        # meta = expand("results/{{scenario}}/modelruns/{run}/meta.{format}", run=MODELRUNS, format=META_FORMAT),
        meta_constriant = expand("results/{{scenario}}/modelruns/{run}/constraints.csv", run=MODELRUNS)
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 600),
        runtime=1
    benchmark:
        "benchmarks/apply_sample/{scenario}.txt"
    log: 
        "logs/apply_sample/{scenario}.log"
    script:
        "../scripts/apply_sample.py"

