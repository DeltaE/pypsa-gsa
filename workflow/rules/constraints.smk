"""Rules for preprocessing policy files"""

rule co2_policy:
    message: "Preparing CO2 Limit constraints for {wildcards.scenario} Scenario"
    input:
        network = config["pypsa_usa"]["network"],
        co2L = "resources/policy/sector_co2_limits.csv",
    output:
        csv = "results/{scenario}/constraints/co2L.csv"
    script:
        "../scripts/carbon_limits.py"

