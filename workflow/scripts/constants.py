"""Constants"""

DEFAULT_LINK_ATTRS = [
    "name",
    "bus0",
    "bus1",
    "type",
    "carrier",
    "efficiency",
    "build_year",
    "lifetime",
    "p_nom",
    "p_nom_mod",
    "p_nom_extendable",
    "p_nom_min",
    "p_nom_max",
    "p_set",
    "p_min_pu",
    "p_max_pu",
    "capital_cost",
    "marginal_cost",
    "marginal_cost_quadratic",
    "stand_by_cost",
    "length",
    "terrain_factor",
    "committable",
    "start_up_cost",
    "shut_down_cost",
    "min_up_time",
    "min_down_time",
    "up_time_before",
    "down_time_before",
    "ramp_limit_up",
    "ramp_limit_down",
    "ramp_limit_start_up",
    "ramp_limit_shut_down",
    "p0",
    "p1",
    "p_nom_opt",
    "status",
    "mu_lower",
    "mu_upper",
    "mu_p_set",
    "mu_ramp_limit_up",
    "mu_ramp_limit_down",
    "bus2",
    "efficiency2",
    "p2",
]
DEFAULT_GENERATOR_ATTRS = [
    "name",
    "bus",
    "control",
    "type",
    "p_nom",
    "p_nom_mod",
    "p_nom_extendable",
    "p_nom_min",
    "p_nom_max",
    "p_min_pu",
    "p_max_pu",
    "p_set",
    "q_set",
    "sign",
    "carrier",
    "marginal_cost",
    "marginal_cost_quadratic",
    "build_year",
    "lifetime",
    "capital_cost",
    "efficiency",
    "committable",
    "start_up_cost",
    "shut_down_cost",
    "stand_by_cost",
    "min_up_time",
    "min_down_time",
    "up_time_before",
    "down_time_before",
    "ramp_limit_up",
    "ramp_limit_down",
    "ramp_limit_start_up",
    "ramp_limit_shut_down",
    "weight",
    "p",
    "q",
    "p_nom_opt",
    "status",
    "mu_upper",
    "mu_lower",
    "mu_p_set",
    "mu_ramp_limit_up",
    "mu_ramp_limit_down",
]
DEFAULT_STORE_ATTRS = [
    "name",
    "bus",
    "type",
    "carrier",
    "e_nom",
    "e_nom_mod",
    "e_nom_extendable",
    "e_nom_min",
    "e_nom_max",
    "e_min_pu",
    "e_max_pu",
    "e_initial",
    "e_initial_per_period",
    "e_cyclic",
    "e_cyclic_per_period",
    "p_set",
    "q_set",
    "sign",
    "marginal_cost",
    "marginal_cost_quadratic",
    "marginal_cost_storage",
    "capital_cost",
    "standing_loss",
    "build_year",
    "lifetime",
    "p",
    "q",
    "e",
    "e_nom_opt",
    "mu_upper",
    "mu_lower",
    "mu_energy_balance",
]