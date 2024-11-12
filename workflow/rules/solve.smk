def get_solver_options(wildards) -> dict[str,str|float]:
    profile = config["solver"]["options"]
    try:
        return config["solving"]["solver_options"][profile]
    except KeyError:
        # print(f"Can not get {profile} solving options")
        return {}

def get_constraint_files(wildcards):
    data = {}
    for policy in config["policy"]:
        if config["policy"][policy]:
            data[policy] = f"results/{wildcards.scenario}/constraints/{policy}.csv"
        else:
            data[policy] = []
    return data

def get_pypsa_usa_files(wildards):
    data = {}
    for policy in config["pypsa_usa"]:
        if not policy:
            continue
        if policy == "ng_limits":
            data["ng_domestic_imports_f"] = "config/pypsa-usa/domestic_imports.csv"
            data["ng_domestic_exports_f"] = "config/pypsa-usa/domestic_exports.csv"
            data["ng_international_imports_f"] = "config/pypsa-usa/international_imports.csv"
            data["ng_international_exports_f"] = "config/pypsa-usa/international_exports.csv"
        elif policy == "hp_capacity":
            data["pop_layout_f"] = "config/pypsa-usa/pop_layout_elec_s33_c4m.csv"
    return data

rule solve_network:
    message: "Solving network"
    params:
        solver = config["solver"]["name"],
        solver_opts = get_solver_options,
        solving_opts = config["solving"]["options"],
        pypsa_usa_opts = config["pypsa_usa"]
    input:
        unpack(get_constraint_files),
        unpack(get_pypsa_usa_files),
        network = "results/{scenario}/modelruns/{run}/n.nc",
    output:
        network = "results/{scenario}/modelruns/{run}/network.nc",
    threads: 12
    resources:
        mem_mb=5000
    log: 
        python = "logs/solve_{scenario}_{run}_python.log",
        solver = "logs/solve_{scenario}_{run}_solver.log",
    script:
        "../scripts/solve.py"