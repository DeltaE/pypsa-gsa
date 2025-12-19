#!/bin/bash

# ------------------------------
# Load Alliance HPC modules
# ------------------------------
module load StdEnv/2023
module load openmpi/4.1.5
# module load netcdf-mpi/4.9.2
module load netcdf
# module load mpi4py/3.1.6
module load scipy-stack
module load arrow
module load ipopt
module load gurobi/11.0.3
module load mycplex/22.1.2
# must load python after mpi
module load python/3.11


# ------------------------------
# Address MPI version issue
# ------------------------------
export OMPI_MCA_ess=^pmi
export OMPI_MCA_plm=isolated
export OMPI_MCA_pml=ob1
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl=^openib

# Prevent ORTE "session_dir" crashes
export OMPI_MCA_orte_base_help_aggregate=0

# ------------------------------
# PROJ fix for PyPSA / pyproj
# ------------------------------

module load proj
export PROJ_DATA="$EBROOTPROJ/share/proj"

# ------------------------------
# Activate Python virtual environment
# ------------------------------
source ~/envs/gsa/bin/activate

# ------------------------------
# Set cplex location
# ------------------------------
# docplex config --upgrade /project/6060200/trevor23/cplex

# ------------------------------
# Enable Rust/Cargo (needed for polars / pypsa)
# ------------------------------
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# ------------------------------
# Optional: environment variables for Gurobi
# (needed only if Gurobi complains)
# ------------------------------
# export GUROBI_HOME=$EBROOTGUROBI
# export GRB_LICENSE_FILE=$HOME/gurobi.lic
# export PATH=$GUROBI_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
