def get_solver_options(wildards) -> dict[str,str|float]:
    profile = config["solver"]["options"]
    try:
        return config["solving"]["solver_options"][profile]
    except KeyError:
        print(f"Can not get {profile} solving options")
        return {}

def get_constraint_files(wildcards):
    data = {}
    for policy in config["policy"]:
        if config["policy"][policy]:
            data[policy] = f"results/{wildcards.scenario}/constraints/{policy}.csv"
        else:
            data[policy] = []
    return data

rule solve_network:
    message: "Solving network"
    params:
        solver = config["solver"]["name"],
        solver_opts = get_solver_options,
        solving_opts = config["solving"]["options"]
    input:
        unpack(get_constraint_files),
        network = "results/{scenario}/modelruns/{run}/n.nc",
    output:
        network = "results/{scenario}/modelruns/{run}/network.nc",
    log: 
        python = "logs/solve_{scenario}_{run}_python.log",
        solver = "logs/solve_{scenario}_{run}_solver.log",
    script:
        "../scripts/solve_network.py"