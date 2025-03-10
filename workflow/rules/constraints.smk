"""Rules for preprocessing policy files"""

rule rps_policy:
    message: "Preparing ReEDS RPS and CES constraints for {wildcards.scenario} Scenario"
    input:
        ces = "resources/reeds/ces_fraction.csv",
        rps = "resources/reeds/rps_fraction.csv",
        network = config["pypsa_usa"]["network"],
    output:
        csv = "results/{scenario}/constraints/rps.csv"
    script:
        "../scripts/rps.py"

rule co2_policy:
    message: "Preparing CO2 Limit constraints for {wildcards.scenario} Scenario"
    input:
        network = config["pypsa_usa"]["network"],
        co2L = "resources/policy/sector_co2_limits.csv",
    output:
        csv = "results/{scenario}/constraints/co2L.csv"
    script:
        "../scripts/carbon_limits.py"

