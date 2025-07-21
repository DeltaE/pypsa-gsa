[![python](https://img.shields.io/badge/python-3.11|_3.12|_3.13-blue.svg)](https://github.com/DeltaE/pypsa-gsa)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/pypi/l/pypsa.svg)](LICENSE)

# PyPSA-USA: Near Term Emission Targets
Uncertainty Analysis Workflow for PyPSA-USA

## Intro

This repo contains the code used for the paper "xxx". Broadly, the workflow allows users to run a uncertainty analysis over [PyPSA-USA](https://github.com/PyPSA/pypsa-usa) networks. Included is 1. Uncertainty Parameterization. 2. Global Sensitivity Analysis (GSA). 3. Uncertainty Propogation (UP). This readme walks through how to configure and run the workflow. 

## Install 

Installation requires uses to clone the GitHub repository and install required dependencies. 

### Clone the Repository 

Users can clone the repository using HTTPS, SSH, or GitHub CLI. See [GitHub docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for information on the different cloning methods. Run one of the following command to clone the repository. Once cloned, open the `pypsa-gsa` project directory. 

#### via HTTPS
```bash
git clone https://github.com/DeltaE/pypsa-gsa.git
```

#### via SSH

If it your first time cloning a repository through ssh, you will need to set up your git with an ssh-key by following [these directions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

```bash
git clone git@github.com:DeltaE/pypsa-gsa.git
```

#### via GitHub CLI
```bash
gh repo clone DeltaE/pypsa-gsa
```

### Dependencies 

Users can install dependencies via [Anaconda](https://anaconda.org/) or [UV](https://docs.astral.sh/uv/). 

#### via Conda

Install mamba (a drop in replacement for conda) following the instructions [here](https://mamba.readthedocs.io/en/latest/index.html). Once installed, activate the environment (`pypsa-gsa`) with the following command. 

```bash
conda env create --file pypsa-gsa.yaml
conda activate pypsa-gsa
```

#### via UV

Install uv following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods). Once installed, activate the environment with the following command. 

```bash 
uv venv
source .venv/bin/activate
```

## How to Configure

This section will walk through how configure the workflow. The results in this paper can be replicated by supplying the network files found in the Zenodo deposit. All configuration file options can be found in the `config/` directory. 

### Scenario

To run a new scenario (ie. the GSA and UP workflow), give a new scenario name in the `config/config.yaml` file. Additionally, for the scenario, you can select if CH4 is counted against the emission budget. As CH4 can lead to eaisly lead to infeasabilities if running with a CO2 limit, users have the option to track CH4, but not count it against the emission limit. 

```yaml
scenario: 
  name: "caiso"
  ch4: false # if true, ch4 counts against emission budget. Else, ch4 still tracked, but not counted against emission budget. 
```

### Network File 

This workflow uses [PyPSA-USA](https://github.com/PyPSA/pypsa-usa) to create a base network of different regions in the USA. The network file and the population layout data (generated during the PyPSA-USA workflow) must be placed in the `config/pypsa-usa/` directory. In the `config/config.yaml` file, update the filepath names for `pypsa-usa`. The `era5_year` parameter extracts the minimum and maximum operating conditions for both natural gas and electrical trading. **If running a sector study, this must be 2018.**  

```yaml
pypsa_usa:
  network: *.nc # network file name
  pop_layout: *.csv # population file name
  era5_year: 2018 # year for energy trade limits
```

### Uncertainty Parameterization

All uncertain parameters are defined in the `config/parameters.csv` file. For each uncertain parameter, the user must provide the information in the following table. The template file `config/parameters.csv` gives an example of the data schema. 

| Column Header | Description |
|---|---|
| name | Unique name to track the uncertain parameter |
| group | Name of group for the parameter to be included in for the sensitivity analysis |
| nice_name | Plotting name for the group |
| component | PyPSA component name. If applying to the timeseries value, ensure you include the '_t' (ie. 'link' and 'link_t' are treated seperatly) |
| carrier | PyPSA-USA carrier to filter the component by |
| attribute* | PyPSA attribute to apply the uncertainty range to |
| range* | Either 'percent' or 'absolute'. Percent will apply a relative percent change to the reference value. Absolute will disregard the reference value and apply the range specified. |
| unit* | Unit of input value. If using a 'percent' range, put 'percent'. All units are converted to PyPSA-USA base units |
| min_value | Minimum value to sample  |
| max_value | Maximum value to sample |
| source | (Optional) Source of the data  |
| notes | (Optional) Any additional info on the data |

*Constraint uncertainty (for example, renewable portfolio standards or electric vehicle policies) are treated a little different (see `config/parameters.csv` for examples). The following constraint uncertainities are supported.
- Transport electrification limits
- Renewable portfolio standards 
- Clean energy standards
- Technology capacity targets 
- Carbon limits 
- Natural gas import/exports to/from outside model scope
- Electrical import/exports to/from outside model scope

When running the workflow, numerous checks are in place to ensure data is inputted corerctly. Moreover, the user can write out metadata associated with the sample to ensure final ranges are reasonable. This is specified in the configuration file as given below. `metadata.csv` and `metadata.yaml` will print out the same data, just in different formats. 

```yaml
metadata:
  csv: True
  yaml: True
  networks: True # keep solved networks 
```

### Gloabl Sensitivity Analysis

The following GSA configuration options are available: 

```yaml
gsa:
  parameters: config/parameters.csv 
  results: config/results_gsa.csv # results to run SA over
  replicates: 10 
  scale: True # Scale Elementary Effect
  rankings: # get union of most impactful parameters
    top_n: 3
    results:
    - marginal_cost_energy
    - marginal_cost_elec
    - carbon
```

The `gsa.parameters` should point to the uncertainty parameterization file described in the previous section. The `gsa.results` points to a file describing what results to run the GSA over (see table below). `gsa.replicates` and `gsa.scale` are options for the Method of Morris described [here](https://salib.readthedocs.io/en/latest/api.html#method-of-morris). If running a GSA over results of different units, it is important to scale the results! `gsa.rankings` is a post processing step to easily extract the what uncertainities are the most impactful on the specified results. 

The following table shows how to configure the GSA results. See the file `config/results_gsa.csv` for an example: 

| Column Header | Description |
|---|---|
| name | Unique name to track the gsa result |
| nice_name | Nice name for the result |
| component | PyPSA component name. If applying to the timeseries value, the average is taken (ie. the average marginal cost) |
| carrier | PyPSA-USA carrier to filter the component by |
| variable | PyPSA attribute to apply the sensitivity to |
| plots | What plots for the variable to be applied to (ie. if you want to plot multiple results against each other) |
| unit | Unit for the plot | 

### Uncertainty Propogation

The following UP configuration options are available:

```yaml
uncertainity:
  sample: lhs # (lhs|sobol)
  replicates: 600
  parameters: # **index names** from parameters csv to include in sample
  - capex_com_elec_water_heater
  - lpg_cost
  - ng_leakage
  results: config/results_ua.csv # results to extract from ua
  plots: config/plots_ua.csv
```

The `uncertainty.sample` is the sampling method to use. Only Latin Hypercube Sampling (LHS) and Sobol Sampling are supported, and are generated using [SALib](https://salib.readthedocs.io/en/latest/). `uncertainty.replicates` modifies the number of samples, with information on how it is used [here](https://salib.readthedocs.io/en/latest/api/SALib.sample.html#module-SALib.sample.latin) for LHS and [here](https://salib.readthedocs.io/en/latest/api/SALib.sample.html#module-SALib.sample.sobol) for Sobol. `uncertainty.parameters` specifies what parameters to include in the sample; these values will typically be the most impactful parameters identified from the GSA (and written out to the file `results/{scenario}/gsa/rankings_top_n.csv`). All other parameters in the `config/parameters.csv` file will take on their average value. `uncertainty.results` and `uncertainty.plots` describes the results and plots to extract from the UP. Details on how to format these files is given below. 

The following table shows how to configure the UP results. See the file `config/results_ua.csv` for an example: 

| Column Header | Description |
|---|---|
| name | Unique name to track the uncertainty result |
| nice_name | Nice name for the result |
| component | PyPSA component name. If applying to the timeseries value, the average is taken (ie. the average marginal cost) |
| carrier | PyPSA-USA carrier to filter the component by |
| variable | PyPSA attribute to apply the sensitivity to |
| barplot* | What barplots to group the result on |
| unit* | Unit for the barplot | 

The following table shows how to configure plots generated from the UP. See the file `config/plots_ua.csv` for an example:

| Column Header | Description |
|---|---|
| name | Unique name to track the uncertainty result |
| nice_name | Nice name for the result |
| type | Type of plot (`scatter`|`bar`) |
| xaxis | Result to plot on xaxis |
| xlabel | Label of xaxis |
| yaxis | Result to plot on yaxis (for scatter plot)|
| ylabel | Label of yaxis |
| plot | Name of plotted file |
| group | Subplot group | 
| xhue | Legend Title (for scatter) or y label (for bar) | 

*TODO: Move this to the dedicated UP plotting config. 

### Solver

Four solving options are provided; [`cbc`](https://github.com/coin-or/Cbc), [`HiGHS`](https://github.com/ERGO-Code/HiGHS), [`CPLEX`](https://www.ibm.com/products/ilog-cplex-optimization-studio), and [`Gurobi`](https://www.gurobi.com/). The user selects what solver and solver profile they want with the following `config/config.yaml` options.

```yaml
# Choose a solver
solver:
  name: gurobi # (cbc|gurobi|cplex|highs)
  options: gurobi-default # see solving config
```

Solver profiles are configured in the `config/solving.yaml` file. An example of a solving profile is given below. This profile is taken from the main PyPSA-USA project [here](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/repo_data/config/config.default.yaml), which is in turn taken from the PyPSA-Eur project [here](https://github.com/PyPSA/pypsa-eur/blob/master/config/config.default.yaml). 

```yaml
solving:
  solver_options:
    gurobi-default:
      threads: 8
      method: 2 # barrier
      crossover: 0
      BarConvTol: 1.e-4
      OptimalityTol: 1.e-4
      FeasibilityTol: 1.e-3
      Seed: 123
      AggFill: 0
      PreDual: 0
      GURO_PAR_BARDENSETHRESH: 200
```

## How to Run

[`snakemake`](https://snakemake.readthedocs.io/en/stable/) is used for workflow orchastration. This section will walk through how to use `snakemake` to execute the workflow. 

### Sankemake Profile

If running locally, default `snakemake` configuration options will likley be fine. If you do want to tune `snakemake` resources, update the `workflow/profiles/default/config.yaml` file. Information on how to write profiles can be found [here](https://snakemake.readthedocs.io/en/stable/executing/cli.html#executing-profiles). 

### Tuning Resources for HPC

If you are deploying the workflow to a HPC, some manual resource tuning may need to happen. First, ensure you change the `snakemake.profile.executor` to the HPC [scheduler](https://snakemake.readthedocs.io/en/stable/tutorial/additional_features.html). All rules are placed in groups for easier scheduling, and will likely **not** need to change. Depending on your PyPSA network, the resouces allocated to each rule may need to change (particullarly with the solve). Dyanmically allocated resources based on file size are set by default as a starting point. You can use the `snakemake.profile.set-resource` argument to override the default. For example, to change the solve to allocate `20GB` of memory, add the following to the `snakemake.profile` (more info [here](https://snakemake.readthedocs.io/en/stable/executing/cli.html#executing-profiles)). 

```yaml
set-resources:
    solve_network:
        mem: 20000MB
```

Additionally, benchmarks files are written out for each rule in the `benchmarks/` directory to inspect resouce usage after the workflow has run. Finally, instructions on how to run a single solve for both the GSA and UP workflow to guess resource requirements are given below. 

### Generate Network Specific Data

Uncertain parameters to the specific network first need to be generated. These are parameters like CO2 targets and technology targets that use growth rates. To generate this data, first update the following configuration options in `config/config.yaml`. If you do **not** want to apply CO2 targets, set the values to `False`. 

```yaml
# config options for data that is generated
generated:
  co2L_min: False # 40 # as percentage of 2005 level emissions
  co2L_max: False # 50 # as percentage of 2005 level emissions
  ccgtccs_max: 10 # as a percentage of max natgas capacity 
```

Then run the following command. A file called `results/{scenario}/generated/config/parameters.csv` will be generated that includes your original parameters and new paremeters appened to the bottom. **Do not manually modify this file!** (Note, this can not be added to the main `Snakemake` workflow due to the `apply_*_sample_to_network` rule exporting all networks in one rule rather than running the rule once for each sample. Its MUCH quicker to just export all at once to redule i/o operations.)

```bash 
snakemake -s workflow/Snakefile.generate
```

### Global Sensitivity Analysis

Next, do a **dry run** of the global sensitivity analysis with the following command: 

```bash
snakemake gsa -n
```

You should see many hundreds or thouands of steps be prompted. If everything looks correct, run the workflow (for real!) with the command: 

```bash
snakemake gsa
```

If you are running on an HPC and want to test the resources required for each solve, you can run the workflow through to one solve with the following command, then check the resouces required with the `seff` command and the job number.  

```bash 
snakemake test_gsa
```

```bash
$ seff 54867276
Job ID: 54867276
Cluster: 
User/Group: 
State: TIMEOUT (exit code 0)
Cores: 1
CPU Utilized: 03:51:58
CPU Efficiency: 96.61% of 04:00:06 core-walltime
Job Wall-clock time: 04:00:06
Memory Utilized: 84.31 GB
Memory Efficiency: 5.43% of 1.52 TB (1.52 TB/node)
```

All results will be available in the `results/{scenario}/gsa/` directory. 

### Uncertainty Propogation

Once the GSA has completed, you can check what parameters are the most influential in the `results/{scenario}/gsa/rankings_top_n.csv` file. These are the parameters to copy over the to `uncertainty.parameters` config option. All other uncertain will be locked to their average value, while these uncertain parameters will be included in the UP. 

Once the uncertain parameters have been inputted, do a **dry run** of the uncertainity propogation with the following command: 

```bash
snakemake up -n
```

You should see many hundreds or thouands of steps be prompted. If everything looks correct, run the workflow (for real!) with the command: 

```bash
snakemake up
```

If you are running on an HPC and want to test the resources required for each solve, you can run the workflow through to one solve with the following command, then check the resouces required with the `seff` command and the job number.  

```bash 
snakemake test_up
```

All results will be available in the `results/{scenario}/up/` directory.

### Result Dashboard

Parsing through static images to understand the GSA and UP results can be very difficult; as there is so much data! A dashboard is available to help users decipher and understand their results. To locally run the dashboard for your results, run the following commands. Note, this dashboard is designed to analyze data at an ISO level, and is not compatiable for smaller zones. 

```bash
# move to the dashboard directory
cd dashboard

# extract data from the results for the dashboard
python collect_data.py

# run the dashboard
python app.py
```

## References

This work uses the following tools: 

> T. Brown, J. Hörsch, D. Schlachtberger, PyPSA: Python for Power System Analysis, 2018, Journal of Open Research Software, 6(1), arXiv:1707.09913, DOI:10.5334/jors.188

> Tehranchi, K., Barnes, T., Frysztacki, M., Hofmann, F., & Azevedo, I. L. PyPSA-USA: An Open-Source Energy System Optimization Model for the United States (Version 0.0.1) [Computer software]. https://doi.org/10.5281/zenodo.10815964

> Iwanaga, T., Usher, W., & Herman, J. (2022). Toward SALib 2.0: Advancing the accessibility and interpretability of global sensitivity analyses. Socio-Environmental Systems Modelling, 4, 18155. https://doi.org/10.18174/sesmo.18155

This work is heavily inspired by the following literature: 

> US Energy-Related Greenhouse Gas Emissions in the Absence of Federal Climate Policy. Hadi Eshraghi, Anderson Rodrigo de Queiroz, and Joseph F. DeCarolis Environmental Science & Technology 2018 52 (17), 9595-9604. https://doi.org/10.1021/acs.est.8b01586 

> Usher W, Barnes T, Moksnes N and Niet T. Global sensitivity analysis to enhance the transparency and rigour of energy system optimisation modelling [version 1; peer review: 1 approved, 2 approved with reservations]. Open Res Europe 2023, 3:30. https://doi.org/10.12688/openreseurope.15461.1

> Moret, S., Gironès, V. C., Bierlaire, M., & Maréchal, F. (2017). Characterization of input uncertainties in strategic energy planning models. Applied Energy, 202, 597–617. https://doi.org/10.1016/j.apenergy.2017.05.106
