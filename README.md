# PyPSA-USA: Near Term Emission Targets
Uncertainity Analysis Workflow for PyPSA-USA

## Intro

This repo contains the code used for the paper "xxx". Broadly, the workflow allows users to run a uncertainity analysis over [PyPSA-USA](https://github.com/PyPSA/pypsa-usa) networks. Included is 1. Uncertainity Parameterization. 2. Global Sensitivity Analysis. 3. Uncertainity Propogation. This readme walks through how the workflow is setup and how to replicate results. 

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

## How to use 

This section will walk through how to run the workflow. The results in this paper can be replicated by supplying the network files found in the Zenodo deposit. All configuration file options can be found in the `config/` directory. 

### Network File 

This workflow uses [PyPSA-USA](https://github.com/PyPSA/pypsa-usa) to create a base network of different regions in the USA. The network file and the population layout data (generated during the PyPSA-USA workflow) must be placed in the `config/pypsa-usa/` directory. In the `config/config.yaml` file, update the filepath names for `pypsa-usa`. The `era5_year` parameter extracts the minimum and maximum operating conditions for both natural gas and electrical trading. **If running a sector study, this must be 2018.**  

```yaml
    pypsa_usa:
    network: *.nc # network file name
    pop_layout: *.csv # population file name
    era5_year: 2018 # year for energy trade limits
```

### Uncertainity Parameterization

All uncertain parameters are defined in the `config/parameters.csv` file. For each uncertain parameter, the user must provide the following: 

| Column Header | Description |
|---|---|
| name | Unique name to track the uncertain parameter |
| group | Name of group for the parameter to be included in for the sensitivity analysis |
| nice_name | Plotting name for the group |
| component | PyPSA component name. If applying to the timeseries value, ensure you include the '_t' (ie. 'link' and 'link_t' are treated seperatly) |
| carrier | PyPSA-USA carrier to filter the component by |
| attribute* | PyPSA attribute to apply the uncertainity range to |
| range* | Either 'percent' or 'absolute'. Percent will apply a relative percent change to the reference value. Absolute will disregard the reference value and apply the range specified. |
| unit* | Unit of input value. If using a 'percent' range, put 'percent'. All units are converted to PyPSA-USA base units |
| min_value | Minimum value to sample  |
| max_value | Maximum value to sample |
| source | (Optional) Source of the data  |
| notes | (Optional) Any additional info on the data |

*Constraint uncertainity (for example, renewable portfolio standards or electric vehicle policies) are treated a little different. The following constraint uncertainiteis are supported. 

to be added. 

When running the workflow, numerous checks are in place to ensure data is inputted corerctly. Moreover, the user can write out metadata associated with the sample to ensure final ranges are reasonable. 

### Gloabl Sensitivity Analysis


### Uncertainity Propogation


### Result Dashboard


## References

[1] Usher W, Barnes T, Moksnes N and Niet T. Global sensitivity analysis to enhance the transparency and rigour of energy system optimisation modelling [version 1; peer review: 1 approved, 2 approved with reservations]. Open Res Europe 2023, 3:30 (https://doi.org/10.12688/openreseurope.15461.1)

