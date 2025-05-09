# PyPSA-USA Global Sensitivity Analysis
Global Sensitivity Analysis Workflow for PyPSA-USA

## Intro

This repo contains the code used for the paper "High-Impact Options to Achieve Near-Term Emission Targets in the USA". The repository is split into three sections; 1. Uncertainity Characterization. 2. Global Sensitivity Analysis. 3. Uncertainity Propogation. This readme walks through how the workflow is setup and how to replicate results. 

## Install 

Install dependencies using conda or mamba into a new environment called `pypsa-gsa`. Activate the new environment. 

```bash
conda env create --file pypsa-gsa.yaml
conda activate pypsa-gsa
```

## How to use 

This section will walk through how to run the workflow. The results in this paper can be replicated by supplying the network files found in the Zenodo deposit.

### Network File 

This workflow uses [PyPSA-USA](https://github.com/PyPSA/pypsa-usa) to create a base network of different regions in the USA. The network file and the population layout data (generated during the PyPSA-USA workflow) must be placed in the `config/pypsa-usa/` directory. In the `config/config.yaml` file, update the filepath names for `pypsa-usa`. The `era5_year` parameter extracts the minimum and maximum operating conditions. If running a sector study, this must be 2018. 

```yaml
    pypsa_usa:
    network: *.nc # network file name
    pop_layout: *.csv # population file name
    era5_year: 2018 # year for natural gas trade limits
```

### Uncertainity Characterization

All uncertain parameters are documented in the `config/parameters.csv` file. 



## References

[1] Usher W, Barnes T, Moksnes N and Niet T. Global sensitivity analysis to enhance the transparency and rigour of energy system optimisation modelling [version 1; peer review: 1 approved, 2 approved with reservations]. Open Res Europe 2023, 3:30 (https://doi.org/10.12688/openreseurope.15461.1)

