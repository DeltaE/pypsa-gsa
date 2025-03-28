
rule extract_uncertainity_results:
    message: "Extracting summary results from uncertainity analysis"
    input:
        network = "results/{scenario}/ua/modelruns/{run}/network.nc",
        results = "results/{scenario}/ua/results.csv"
    output:
        csv = "results/{scenario}/ua/modelruns/{run}/results.csv"
    log: 
        "logs/extract_results/{scenario}_{run}.log"
    resources:
        mem_mb=lambda wc, input: max(1.25 * input.size_mb, 250),
        runtime=1
    benchmark:
        "benchmarks/extract_uncertainity_results/{scenario}_{run}.txt"
    group:
        "solve_{scenario}_{run}"
    script:
        "../scripts/extract_results.py"
