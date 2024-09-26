rule create_sample:
    message: "Creating sample for '{params.replicates}' trajectories and '{params.parameters}' parameters"
    params:
        replicates=config["gsa"]["replicates"],
        parameters=config["gsa"]["parameters"]
    output: 
        sample_file = "results/{scenario}/sample.csv"
    conda: "../envs/sample.yaml"
    log: "logs/create_{scenario}_sample.log"
    script:
        "../scripts/create_sample.py"

# Apply sample creates all samples, rather than one sample at a time to prevent the 
# need of reding in the base network many times

rule apply_sample_to_network:
    message: "Applying sample"
    params:
        parameters = config["gsa"]["parameters"],
        root_dir = "results/{scenario}/modelruns/"
    input: 
        sample_file = "results/{scenario}/sample.csv",
        network = config["scenario"]["network"]
    output:
        n = temp(expand("results/{{scenario}}/modelruns/{run}/n.nc", run=MODELRUNS)),
        meta = expand("results/{{scenario}}/modelruns/{run}/meta.yaml", run=MODELRUNS)
    log: "logs/apply_{scenario}_sample.log"
    script:
        "../scripts/apply_sample.py"
