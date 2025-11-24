def get_solver_options(wildards) -> dict[str,str|float]:
    profile = config["solver"]["options"]
    try:
        return config["solving"]["solver_options"][profile]
    except KeyError:
        # print(f"Can not get {profile} solving options")
        return {}

def get_network_from_checkpoint(wildcards):
    """Get network file from checkpoint output for specific run.
    Directly construct path since checkpoint creates all files at once."""
    return f"results/{wildcards.scenario}/{wildcards.mode}/modelruns/{wildcards.run}/n.nc"

def get_constraints_from_checkpoint(wildcards):
    """Get constraints file from checkpoint output for specific run.
    Directly construct path since checkpoint creates all files at once."""
    return f"results/{wildcards.scenario}/{wildcards.mode}/modelruns/{wildcards.run}/constraints.csv"

def get_sanitized_parameters(wildcards):
    """Get sanitized parameters file"""
    return f"results/{wildcards.scenario}/gsa/parameters.csv"

def get_sample_checkpoint_done(wildcards):
    """Get a single file from checkpoint to ensure it's evaluated once.
    This creates a single dependency rather than querying checkpoint for each run."""
    if wildcards.mode == "gsa":
        # Return the scaled_sample file which is created once by the checkpoint
        return checkpoints.apply_gsa_sample_to_network.get(
            scenario=wildcards.scenario, mode="gsa"
        ).output.scaled_sample
    elif wildcards.mode == "ua":
        return checkpoints.apply_ua_sample_to_network.get(
            scenario=wildcards.scenario, mode="ua"
        ).output.scaled_sample
    else:
        raise ValueError(f"Invalid mode: {wildcards.mode}")

rule solve_network:
    message: "Solving network"
    wildcard_constraints:
        mode="gsa|ua",
        run=r"\d+"
    params:
        solver = config["solver"]["name"],
        solver_opts = get_solver_options,
        solving_opts = config["solving"]["options"],
        model_opts = config["model_options"],
    input:
        # Single checkpoint dependency evaluated once per mode, not per run
        _checkpoint_done = get_sample_checkpoint_done,
        sanitized_parameters = get_sanitized_parameters,
        network = get_network_from_checkpoint,
        constraints = get_constraints_from_checkpoint,
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
        tct_f = "results/{scenario}/constraints/tct.csv",
        ev_policy_f = "results/{scenario}/constraints/ev_policy.csv",
        import_export_flows_f = "results/{scenario}/constraints/import_export_flows.csv",
    output:
        network = temp("results/{scenario}/{mode}/modelruns/{run}/network.nc") if not config['metadata']['networks'] else "results/{scenario}/{mode}/modelruns/{run}/network.nc",
    threads: 12
    resources:
        mem_mb=2000,
        runtime=2
    benchmark:
        "benchmarks/solve/{scenario}_{mode}_{run}.txt"
    log: 
        python = "logs/solve/{scenario}_{mode}_{run}_python.log",
        solver = "logs/solve/{scenario}_{mode}_{run}_solver.log",
    group:
        "solve_{scenario}_{mode}_{run}"
    script:
        "../scripts/solve.py"

rule test_solve_network:
    message: "Solving network"
    wildcard_constraints:
        mode="gsa"
    params:
        solver = config["solver"]["name"],
        solver_opts = get_solver_options,
        solving_opts = config["solving"]["options"],
        model_opts = config["model_options"],
    input:
        network = "results/{scenario}/{mode}/modelruns/testing/0/n.nc",
        constraints = "results/{scenario}/{mode}/modelruns/testing/0/constraints.csv",
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
        tct_f = "results/{scenario}/constraints/tct.csv",
        ev_policy_f = "results/{scenario}/constraints/ev_policy.csv",
        import_export_flows_f = "results/{scenario}/constraints/import_export_flows.csv",
    output:
        network = temp("results/{scenario}/{mode}/modelruns/testing/0/network.nc") if not config['metadata']['networks'] else "results/{scenario}/{mode}/modelruns/testing/0/network.nc",
    threads: 12
    resources:
        mem_mb=2000,
        runtime=2
    benchmark:
        "benchmarks/solve/{scenario}_{mode}_testing.txt"
    log: 
        python = "logs/solve/{scenario}_{mode}_testing_python.log",
        solver = "logs/solve/{scenario}_{mode}_testing_solver.log",
    group:
        "solve_{scenario}_{mode}_testing"
    script:
        "../scripts/solve.py"