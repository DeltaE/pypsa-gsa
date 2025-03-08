"""Gets natural gas trade data."""

from eia import Trade

if __name__ == "__main__":

    if "snakemake" in globals():
        year = snakemake.params.year
        api = snakemake.params.api
        domestic_csv = snakemake.output.domestic
        international_csv = snakemake.output.international
    else:
        year = 2018
        api = ""
        domestic_csv = "resources/domestic_ng.csv"
        international_csv = "resources/interational_ng.csv"
        
    year += 1 # need to index by one for correct eia year
        
    domestic = Trade("gas", False, "exports", year, api).get_data()
    international = Trade("gas", True, "exports", year, api).get_data()
    
    domestic.to_csv(domestic_csv)
    international.to_csv(international_csv)