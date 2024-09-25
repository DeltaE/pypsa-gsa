rule create_sample:
    message: "Creating sample for '{params.replicates}' trajectories and '{params.parameters}' parameters"
    params:
        replicates=config["replicates"],
        parameters=config["parameters"]
    output: 
        sample_file = "modelruns/{scenario}/sample.csv"
    conda: "../envs/sample.yaml"
    log: "results/log/create_{scenario}_sample.log"
    script:
        "../scripts/create_sample.py"

# Apply sample creates all samples, rather than one sample at a time to prevent the 
# need of reding in the base network many times

rule apply_sample:
    message: "Applying sample"
    params:
        parameters = config["parameters"]
        root_dir = "modelruns/{scenario}"
    input: 
        sample_file = "modelruns/{scenario}/sample.csv"
        network = config["scenario"]["network"]
    output:
        temp(expand("modelruns/{{scenario}}/{run}/n.nc", run=MODELRUNS))
    log: "results/log/apply_{scenario}_sample.log"
    script:
        "../scripts/apply_sample.py"

rule solve_network:
    message: "Solving network"
    input:
        network = "modelruns/{scenario}/{run}/n.nc"
    output:
        network = "modelruns/{scenario}/{run}/network.nc"
    log: "results/log/solve_{scenario}.log"
    script:
        "../scripts/solve_network.py"

rule cal