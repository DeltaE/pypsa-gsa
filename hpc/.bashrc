# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
. "$HOME/.cargo/env"

# Path for custom modules
export MODULEPATH=$HOME/modulefiles:$MODULEPATH

# Path for CPLEX executable
export CPLEX_STUDIO_DIR=/home/trevor23/projects/def-tniet-ab/trevor23/cplex_2212
export PATH=$CPLEX_STUDIO_DIR/cplex/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$CPLEX_STUDIO_DIR/cplex/bin/x86-64_linux:$LD_LIBRARY_PATH

# for python location
export CPLEX_STUDIO_BINARIES=$CPLEX_STUDIO_DIR/cplex/bin/x86-64_linux
