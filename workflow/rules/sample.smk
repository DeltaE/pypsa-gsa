rule create_sample:
    message: "Creating sample with '{params.replicates}' trajectories"
    params:
        replicates=config["gsa"]["replicates"],
    input:
        parameters="results/{scenario}/parameters.csv"
    output: 
        sample_file = "results/{scenario}/sample.csv"
    log: "logs/create_{scenario}_sample.log"
    script:
        "../scripts/create_sample.py"

rule testing:
    input:
        "results/Testing/sample.csv",
        "resources/natural_gas/domestic.csv"

# Apply sample creates all samples, rather than one sample at a time to prevent the 
# need of reding in the base network many times

rule apply_sample_to_network:
    message: "Applying sample"
    params:
        parameters = config["gsa"]["parameters"],
        root_dir = "results/{scenario}/modelruns/"
    input: 
        sample_file = "results/{scenario}/sample.csv",
        network = "results/{scenario}/network.nc"
    output:
        n = temp(expand("results/{{scenario}}/modelruns/{run}/n.nc", run=MODELRUNS)),
        meta = expand("results/{{scenario}}/modelruns/{run}/meta.yaml", run=MODELRUNS)
    log: "logs/apply_{scenario}_sample.log"
    script:
        "../scripts/apply_sample.py"
