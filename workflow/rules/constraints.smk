"""Rules for preprocessing policy files"""

rule rps_policy:
    message: "Preparing ReEDS RPS and CES constraints for {wildcards.scenario} Scenario"
    input:
        ces = "resources/reeds/ces_fraction.csv",
        rps = "resources/reeds/rps_fraction.csv",
        network = config["scenario"]["network"],
    output:
        csv = "results/{scenario}/constraints/rps.csv"
    script:
        "../scripts/rps.py"

rule interface_transmission_limits:
    message: "Preparing ReEDS ITL constraints for {wildcards.scenario} Scenario"
    input:
        network = config["scenario"]["network"],
        itl = "resources/reeds/transmission_capacity_init_AC_ba_NARIS2024.csv"
    output:
        csv = "results/{scenario}/constraints/itl.csv"
    script:
        "../scripts/itl.py"

rule safer_policy: 
    message: "Preparing ReEDS SAFER constraints for {wildcards.scenario} Scenario"
    input:
        network = config["scenario"]["network"],
        safer = "resources/reeds/prm_annual.csv",
    output:
        csv = "results/{scenario}/constraints/safer.csv"
    script:
        "../scripts/itl.py"

rule co2_policy:
    message: "Preparing CO2 Limit constraints for {wildcards.scenario} Scenario"
    input:
        network = config["scenario"]["network"],
        co2L = "resources/policy/sector_co2_limits.csv",
    output:
        csv = "results/{scenario}/constraints/co2L.csv"
    script:
        "../scripts/carbon_limits.py"

