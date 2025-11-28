def get_solver_options(wildards) -> dict[str,str|float]:
    profile = config["solver"]["options"]
    try:
        return config["solving"]["solver_options"][profile]
    except KeyError:
        # print(f"Can not get {profile} solving options")
        return {}

rule solve_network:
    message: "Solving network"
    params:
        solver = config["solver"]["name"],
        solver_opts = get_solver_options,
        solving_opts = config["solving"]["options"],
        pypsa_usa_opts = config["pypsa_usa"]
    input:
        network = "results/{scenario}/modelruns/{run}/n.nc",
        constraints = "results/{scenario}/modelruns/{run}/constraints.csv",
        pop_layout_f = "results/{scenario}/constraints/pop_layout.csv",
        ng_domestic_f = "results/{scenario}/constraints/ng_domestic.csv",
        ng_international_f = "results/{scenario}/constraints/ng_international.csv",
        rps_f = "results/{scenario}/constraints/rps.csv",
        ces_f = "results/{scenario}/constraints/ces.csv",
        tct_f = "results/{scenario}/constraints/tct.csv",
        ev_policy_f = "results/{scenario}/constraints/ev_policy.csv"
    output:
        network = "results/{scenario}/modelruns/{run}/network.nc",
    threads: 12
    resources:
        mem_mb=8000,  # Increased for solver with 12 threads
        runtime=2
    benchmark:
        "benchmarks/solve/{scenario}_{run}.txt"
    log: 
        python = "logs/solve/{scenario}_{run}_python.log",
        solver = "logs/solve/{scenario}_{run}_solver.log",
    group:
        "solve_{scenario}_{run}"
    script:
        "../scripts/solve.py"