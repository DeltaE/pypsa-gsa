"""Prepares data for the dashboard."""

isos = ["caiso", "ercot", "isone", "miso", "nyiso", "pjm", "spp", "northwest", "southeast", "southwest"]

rule collapse_sa:
    input:
        expand("results/{scenario}/gsa/SA/{result}.csv", scenario=SCENARIO, result=GSA_RESULT_FILES),
    output:
        "dashboard/data/{scenario}/sa.csv"
    run:
        import pandas as pd
        from pathlib import Path
        dfs = []
        for f in input[0]:
            name = Path(f).stem
            df = pd.read_csv(f).mu_star
            dfs.concat(df.rename(columns={"mu_star":name}))
        df = pd.concat(dfs, axis=1)
        df.to_csv(output[0], index=False)

rule collapse_sa_runs:
    input:
        expand("results/{scenario}/gsa/results/{result}.csv", scenario=SCENARIO, result=GSA_RESULT_FILES),
    output:
        "dashboard/data/{scenario}/models_sa.csv"
    run:
        import pandas as pd
        from pathlib import Path
        dfs = []
        for f in input[0]:
            name = Path(f).stem
            df = pd.read_csv(f, index_col=1).value
            dfs.concat(df.rename(columns={"value":name}))
        df = pd.concat(dfs, axis=1)
        df.to_csv(output[0], index=True)