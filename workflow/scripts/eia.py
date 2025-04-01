"""Interfaces with EIA API.

Code adapted from PyPSA-USA repository at:
https://github.com/PyPSA/pypsa-usa/blob/master/workflow/scripts/eia.py
"""

from abc import ABC, abstractmethod
from typing import ClassVar
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from constants import POINTS_OF_ENTRY

import logging
logger = logging.getLogger(__name__)

API_BASE = "https://api.eia.gov/v2/"

AEO_SCENARIOS = {
    "reference": "ref2023",  # reference
    "aeo2022": "aeo2022ref",  # AEO2022 Reference case
    "no_ira": "noIRA",  # No inflation reduction act
    "low_ira": "lowupIRA",  # Low Uptake of Inflation Reduction Act
    "high_ira": "highupIRA",  # High Uptake of Inflation Reduction Act
    "high_growth": "highmacro",  # High Economic Growth
    "low_growth": "lowmacro",  # Low Economic Growth
    "high_oil_price": "highprice",  # High Oil Price
    "low_oil_price": "lowprice",  # Low Oil Price
    "high_oil_gas_supply": "highogs",  # High Oil and Gas Supply
    "low_oil_gas_supply": "lowogs",  # Low Oil and Gas Supply
    "high_ztc": "highZTC",  # High Zero-Carbon Technology Cost
    "low_ztc": "lowZTC",  # Low Zero-Carbon Technology Cost
    "high_growth_high_ztc": "highmachighZTC",  # High Economic Growth-High Zero-Carbon Technology Cost
    "high_growth_low_ztc": "highmaclowZTC",  # High Economic Growth-Low Zero-Carbon Technology Cost
    "low_growth_high_ztc": "lowmachighZTC",  # Low Economic Growth-High Zero-Carbon Technology Cost
    "low_growth_low_ztc": "lowmaclowZTC",  # Low Economic Growth-Low Zero-Carbon Technology Cost
    "fast_build_high_lng": "lng_hp_fast",  # Fast Builds Plus High LNG Price
    "high_lng": "lng_hp",  # High LNG Price
    "low_lng": "lng_lp",  # Low LNG Price
}


class InputPropertyError(Exception):
    """Class for exceptions."""

    def __init__(self, propery, valid_options, recived_option) -> None:
        self.message = (
            f" {propery} must be in {valid_options}; recieved {recived_option}"
        )

    def __str__(self):
        return self.message


class EiaData(ABC):
    """Creator class to extract EIA data."""

    @abstractmethod
    def data_creator(self):  # type DataExtractor
        """Gets the data."""
        pass

    def get_data(self, pivot: bool = False) -> pd.DataFrame:
        """Get formated data."""
        product = self.data_creator()
        df = product.retrieve_data()
        df = product.format_data(df)
        if pivot:
            df = product._pivot_data(df)
        return df

    def get_api_call(self) -> pd.DataFrame:
        """Get API URL."""
        product = self.data_creator()
        return product.build_url()

    def get_raw_data(self) -> pd.DataFrame:
        """Get unformatted data from API."""
        product = self.data_creator()
        return product.retrieve_data()


class DataExtractor(ABC):
    """Extracts and formats data."""

    def __init__(self, year: int, api_key: str | None = None):
        self.api_key = api_key
        # self.year = self._set_year(year)
        self.year = year

    @abstractmethod
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats retrieved data from EIA.

                    series-description    value  units   state
        period
        2020-01-15  description of data   2.74  $/MCF    U.S.
        ...
        """
        pass

    @abstractmethod
    def build_url(self) -> str:
        """Builds API url."""
        pass

    def retrieve_data(self) -> pd.DataFrame:
        """Retrieves and converts API data into dataframe."""
        url = self.build_url()
        data = self._request_eia_data(url)
        return pd.DataFrame.from_dict(data["response"]["data"])

    @staticmethod
    def _set_year(year: int) -> int:
        if year < 2009:
            logger.info(f"year must be > 2008. Recieved {year}. Setting to 2009")
            return 2009
        elif year > 2022:
            logger.info(f"year must be < 2023. Recieved {year}. Setting to 2022")
            return 2022
        else:
            return year

    @staticmethod
    def _request_eia_data(url: str) -> dict[str, dict | str]:
        """
        Retrieves data from EIA API.

        url in the form of "https://api.eia.gov/v2/" followed by api key and facets
        """
        # sometimes running into HTTPSConnectionPool error. adding in retries helped
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        response = session.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()  # Assumes the response is in JSON format
        else:
            logger.error(f"EIA Request failed with status code: {response.status_code}")
            raise requests.ConnectionError(f"Status code {response.status_code}")

    @staticmethod
    def _format_period(dates: pd.Series) -> pd.Series:
        """Parses dates into a standard monthly format."""
        try:  # try to convert to YYYY-MM-DD format
            return pd.to_datetime(dates, format="%Y-%m-%d")
        except ValueError:
            try:  # try to convert to YYYY-MM format
                return pd.to_datetime(dates + "-01", format="%Y-%m-%d")
            except ValueError:
                return pd.NaT

    @staticmethod
    def _pivot_data(df: pd.DataFrame) -> pd.DataFrame:
        """Pivots data on period and state."""
        df = df.reset_index()
        try:
            return df.pivot(
                index="period",
                columns="state",
                values="value",
            )
        # Im not actually sure why sometimes we are hitting this :(
        # ValueError: Index contains duplicate entries, cannot reshape
        except ValueError:
            logger.info("Reshaping using pivot_table and aggfunc='mean'")
            return df.pivot_table(
                index="period",
                columns="state",
                values="value",
                aggfunc="mean",
            )

    @staticmethod
    def _assign_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(
            {"series-description": str, "value": float, "units": str, "state": str},
        )

class Emissions(EiaData):
    """State level emissions data."""

    def __init__(
        self,
        sector: str,
        year: int,
        api: str,
        fuel: str | None = None,
    ) -> None:
        self.sector = sector  # (power|residential|commercial|industry|transport|total)
        self.year = year  # 1970 - 2021
        self.api = api
        self.fuel = "all" if not fuel else fuel  # (coal|oil|gas|all)

    def data_creator(self):
        """Initializes data extractor."""
        return _StateEmissions(self.sector, self.fuel, self.year, self.api)

class EnergyDemand(EiaData):
    """
    Energy demand at a annual national level.

    If historical year is provided, monthly energy consumption for that
    year is provided. If a future year is provided, annual projections
    from 2023 up to that year are provided based on the scenario given
    """

    def __init__(
        self,
        sector: str,
        year: int,
        api: str,
        scenario: str | None = None,
    ) -> None:
        self.sector = sector  # (residential, commercial, transport, industry)
        self.year = year
        self.api = api
        self.scenario = scenario  # only for AEO scenario

    def data_creator(self) -> pd.DataFrame:
        """Initializes data extractor."""
        if self.year < 2024:
            if self.scenario:
                logger.debug("Can not apply AEO scenario to historical demand")
            return _HistoricalSectorEnergyDemand(self.sector, self.year, self.api)
        elif self.year >= 2024:
            aeo = "reference" if not self.scenario else self.scenario
            return _ProjectedSectorEnergyDemand(self.sector, self.year, aeo, self.api)
        else:
            raise InputPropertyError(
                propery="EnergyDemand",
                valid_options="year",
                recived_option=self.year,
            )


class TransportationDemand(EiaData):
    """
    Transportation demand in VMT (or similar).

    If historical year is provided, monthly energy consumption for that
    year is provided. If a future year is provided, annual projections
    from 2023 up to that year are provided based on the scenario given
    """

    def __init__(
        self,
        vehicle: str,
        year: int,
        api: str,
        units: str = "travel",  # travel | btu
        scenario: str | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.year = year
        self.api = api
        self.units = units
        self.scenario = scenario

    def data_creator(self) -> pd.DataFrame:
        """Initializes data extractor."""
        if self.units == "travel":
            if self.year < 2024:
                return _HistoricalTransportTravelDemand(
                    self.vehicle,
                    self.year,
                    self.api,
                )
            elif self.year >= 2024:
                aeo = "reference" if not self.scenario else self.scenario
                return _ProjectedTransportTravelDemand(
                    self.vehicle,
                    self.year,
                    aeo,
                    self.api,
                )
            else:
                raise InputPropertyError(
                    propery="TransportationTravelDemand",
                    valid_options=range(2017, 2051),
                    recived_option=self.year,
                )
        elif self.units == "btu":
            if self.year < 2024:
                return _HistoricalTransportBtuDemand(self.vehicle, self.year, self.api)
            elif self.year >= 2024:
                aeo = "reference" if not self.scenario else self.scenario
                return _ProjectedTransportBtuDemand(
                    self.vehicle,
                    self.year,
                    aeo,
                    self.api,
                )
            else:
                raise InputPropertyError(
                    propery="TransportationBtuDemand",
                    valid_options=range(2017, 2051),
                    recived_option=self.year,
                )
        else:
            raise InputPropertyError(
                propery="TransportationDemand",
                valid_options=("travel", "btu"),
                recived_option=self.units,
            )


class Trade(EiaData):
    """Natural gas trade data."""

    def __init__(
        self,
        fuel: str,
        international: bool,
        direction: str,
        year: int,
        api: str,
    ) -> None:
        self.fuel = fuel
        self.international = international
        self.direction = direction  # (imports|exports)
        self.year = year
        self.api = api

    def data_creator(self) -> pd.DataFrame:
        """Initializes data extractor."""
        if self.fuel == "gas":
            if self.international:
                # gives monthly values
                return _InternationalGasTrade(self.direction, self.year, self.api)
            else:
                # gives annual values
                return _DomesticGasTrade(self.direction, self.year, self.api)
        else:
            raise InputPropertyError(
                propery="Energy Trade",
                valid_options=["gas"],
                recived_option=self.fuel,
            )

class InstalledCapacity(EiaData):
    """Technology Capacity Installations."""

    def __init__(
        self,
        sector: str,
        fuel: bool,
        scenario: str,
        year: int,
        api: str,
    ) -> None:
        self.sector = sector 
        self.fuel = fuel
        self.scenario = scenario
        self.year = year
        self.api = api

    def data_creator(self) -> pd.DataFrame:
        """Initializes data extractor."""
        if self.sector == "power":
            return _PowerCapacity(self.fuel, self.year, self.scenario, self.api)
        else:
            raise InputPropertyError(
                propery="InstalledCapacity",
                valid_options=["power"],
                recived_option=self.sector,
            )


class _HistoricalSectorEnergyDemand(DataExtractor):
    """
    Extracts historical energy demand at a yearly national level.

    Note, this is end use energy consumed (does not include losses)
    - https://www.eia.gov/totalenergy/data/flow-graphs/electricity.php
    - https://www.eia.gov/outlooks/aeo/pdf/AEO2023_Release_Presentation.pdf (pg 17)
    """

    sector_codes: ClassVar[dict[str, str]] = {
        "residential": "TNR",
        "commercial": "TNC",
        "industry": "TNI",
        "transport": "TNA",
        "all": "TNT",  # total energy consumed by all end-use sectors
    }

    def __init__(self, sector: str, year: int, api: str) -> None:
        self.sector = sector
        if sector not in self.sector_codes.keys():
            raise InputPropertyError(
                propery="Historical Energy Demand",
                valid_options=list(self.sector_codes),
                recived_option=sector,
            )
        super().__init__(year, api)

    def build_url(self) -> str:
        base_url = "total-energy/data/"
        facets = f"frequency=annual&data[0]=value&facets[msn][]={self.sector_codes[self.sector]}CBUS&start={self.year}&end=2023&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesDescription": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df = df[["series-description", "value", "units", "state"]].sort_index()
        assert len(df.units.unique()) == 1
        assert df.units.unique()[0] == "Trillion Btu"
        df["value"] = df.value.astype(float)
        df["value"] = df.value.div(1000).round(6)
        df["units"] = "quads"
        return self._assign_dtypes(df)


class _ProjectedSectorEnergyDemand(DataExtractor):
    """Extracts projected energy demand at a national level from AEO 2023."""

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # note, these are all "total energy use by end use - total gross end use consumption"
    # https://www.eia.gov/totalenergy/data/flow-graphs/electricity.php
    sector_codes: ClassVar[dict[str, str]] = {
        "residential": "cnsm_enu_resd_NA_dele_NA_NA_qbtu",
        "commercial": "cnsm_enu_comm_NA_dele_NA_NA_qbtu",
        "industry": "cnsm_enu_idal_NA_dele_NA_NA_qbtu",
        "transport": "cnsm_enu_trn_NA_dele_NA_NA_qbtu",
    }

    def __init__(self, sector: str, year: int, scenario: str, api: str):
        super().__init__(year, api)
        self.scenario = scenario
        self.sector = sector
        if scenario not in self.scenario_codes.keys():
            raise InputPropertyError(
                propery="Projected Energy Demand Scenario",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        if sector not in self.sector_codes.keys():
            raise InputPropertyError(
                propery="Projected Energy Demand Sector",
                valid_options=list(self.sector_codes),
                recived_option=sector,
            )

    def build_url(self) -> str:
        base_url = "aeo/2023/data/"
        facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={self.sector_codes[self.sector]}&start=2024&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(columns={"seriesName": "series-description", "unit": "units"})
        df["state"] = "U.S."
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class _HistoricalTransportTravelDemand(DataExtractor):
    """Gets Transport demand in units of travel."""

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # units will be different umong these!
    vehicle_codes: ClassVar[dict[str, str]] = {
        "light_duty": "kei_trv_trn_NA_ldv_NA_NA_blnvehmls",
        "med_duty": "kei_trv_trn_NA_cml_NA_NA_blnvehmls",
        "heavy_duty": "kei_trv_trn_NA_fght_NA_NA_blnvehmls",
        "bus": "_trv_trn_NA_bst_NA_NA_bpm",
        "rail_passenger": "_trv_trn_NA_rlp_NA_NA_bpm",
        "boat_shipping": "kei_trv_trn_NA_dmt_NA_NA_blntnmls",
        "rail_shipping": "kei_trv_trn_NA_rail_NA_NA_blntnmls",
        "air": "kei_trv_trn_NA_air_NA_NA_blnseatmls",
    }

    def __init__(self, vehicle: str, year: int, api: str) -> None:
        self.vehicle = vehicle
        if vehicle not in self.vehicle_codes.keys():
            raise InputPropertyError(
                propery="Historical Transport Travel Demand",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        year = self.check_available_data_year(year)
        super().__init__(year, api)

    def check_available_data_year(self, year: int) -> int:
        if self.vehicle in ("bus", "rail_passenger"):
            if year < 2018:
                logger.error(
                    f"{self.vehicle} data not available for {year}. Returning data for year 2018.",
                )
                return 2018
        return year

    def build_url(self) -> str:
        if self.year >= 2022:
            aeo = 2023
        elif self.year >= 2015:
            aeo = self.year + 1
        else:
            raise NotImplementedError

        base_url = f"aeo/{aeo}/data/"
        scenario = f"ref{aeo}"

        facets = f"frequency=annual&data[0]=value&facets[scenario][]={scenario}&facets[seriesId][]={self.vehicle_codes[self.vehicle]}&start={self.year}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split("Transportation : Travel Indicators : ")[1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class _ProjectedTransportTravelDemand(DataExtractor):
    """Gets Transport demand in units of travel."""

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # units will be different umong these!
    vehicle_codes: ClassVar[dict[str, str]] = {
        "light_duty": "kei_trv_trn_NA_ldv_NA_NA_blnvehmls",
        "med_duty": "kei_trv_trn_NA_cml_NA_NA_blnvehmls",
        "heavy_duty": "kei_trv_trn_NA_fght_NA_NA_blnvehmls",
        "bus": "_trv_trn_NA_bst_NA_NA_bpm",
        "rail_passenger": "_trv_trn_NA_rlp_NA_NA_bpm",
        "boat_shipping": "kei_trv_trn_NA_dmt_NA_NA_blntnmls",
        "rail_shipping": "kei_trv_trn_NA_rail_NA_NA_blntnmls",
        "air": "kei_trv_trn_NA_air_NA_NA_blnseatmls",
    }

    def __init__(self, vehicle: str, year: int, scenario: str, api: str) -> None:
        self.vehicle = vehicle
        self.scenario = scenario
        if scenario not in self.scenario_codes.keys():
            raise InputPropertyError(
                propery="Projected Transport Travel Demand Scenario",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        if vehicle not in self.vehicle_codes.keys():
            raise InputPropertyError(
                propery="Projected Transport Travel Demand",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        super().__init__(year, api)

    def build_url(self) -> str:
        base_url = "aeo/2023/data/"
        facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={self.vehicle_codes[self.vehicle]}&start=2024&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split("Transportation : Travel Indicators : ")[1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class _HistoricalTransportBtuDemand(DataExtractor):
    """Gets Transport demand in units of btu."""

    # units will be different umong these!
    vehicle_codes: ClassVar[dict[str, str]] = {
        "light_duty": "cnsm_NA_trn_ldv_use_NA_NA_qbtu",
        "med_duty": "cnsm_NA_trn_cml_use_NA_NA_qbtu",
        "heavy_duty": "cnsm_NA_trn_fght_use_NA_NA_qbtu",
        "bus": "cnsm_NA_trn_bst_use_NA_NA_qbtu",
        "rail_passenger": "cnsm_NA_trn_rlp_use_NA_NA_qbtu",
        "boat_shipping": "cnsm_NA_trn_shdt_use_NA_NA_qbtu",
        "rail_shipping": "cnsm_NA_trn_rlf_use_NA_NA_qbtu",
        "air": "cnsm_NA_trn_air_use_NA_NA_qbtu",
        "boat_international": "cnsm_NA_trn_shint_use_NA_NA_qbtu",
        "boat_recreational": "cnsm_NA_trn_rbt_use_NA_NA_qbtu",
        "military": "cnsm_NA_trn_milu_use_NA_NA_qbtu",
        "lubricants": "cnsm_NA_trn_lbc_use_NA_NA_qbtu",
        "pipeline": "cnsm_NA_trn_pft_use_NA_NA_qbtu",
    }

    def __init__(self, vehicle: str, year: int, api: str) -> None:
        self.vehicle = vehicle
        if vehicle not in self.vehicle_codes.keys():
            raise InputPropertyError(
                propery="Historical BTU Transport Demand",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        year = self.check_available_data_year(year)
        super().__init__(year, api)

    def check_available_data_year(self, year: int) -> int:
        if self.vehicle in ("bus", "rail_passenger"):
            if year < 2018:
                logger.error(
                    f"{self.vehicle} data not available for {year}. Returning data for year 2018.",
                )
                return 2018
        return year

    def build_url(self) -> str:
        if self.year >= 2022:
            aeo = 2023
        elif self.year >= 2015:
            aeo = self.year + 1
        else:
            raise NotImplementedError

        base_url = f"aeo/{aeo}/data/"
        scenario = f"ref{aeo}"

        facets = f"frequency=annual&data[0]=value&facets[scenario][]={scenario}&facets[seriesId][]={self.vehicle_codes[self.vehicle]}&start={self.year}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split("Transportation : Energy Use by Mode : ")[1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class _ProjectedTransportBtuDemand(DataExtractor):
    """Gets Transport demand in units of quads."""

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # units will be different umong these!
    vehicle_codes: ClassVar[dict[str, str]] = {
        "light_duty": "cnsm_NA_trn_ldv_use_NA_NA_qbtu",
        "med_duty": "cnsm_NA_trn_cml_use_NA_NA_qbtu",
        "heavy_duty": "cnsm_NA_trn_fght_use_NA_NA_qbtu",
        "bus": "cnsm_NA_trn_bst_use_NA_NA_qbtu",
        "rail_passenger": "cnsm_NA_trn_rlp_use_NA_NA_qbtu",
        "boat_shipping": "cnsm_NA_trn_shdt_use_NA_NA_qbtu",
        "rail_shipping": "cnsm_NA_trn_rlf_use_NA_NA_qbtu",
        "air": "cnsm_NA_trn_air_use_NA_NA_qbtu",
        "boat_international": "cnsm_NA_trn_shint_use_NA_NA_qbtu",
        "boat_recreational": "cnsm_NA_trn_rbt_use_NA_NA_qbtu",
        "military": "cnsm_NA_trn_milu_use_NA_NA_qbtu",
        "lubricants": "cnsm_NA_trn_lbc_use_NA_NA_qbtu",
        "pipeline": "cnsm_NA_trn_pft_use_NA_NA_qbtu",
    }

    def __init__(self, vehicle: str, year: int, scenario: str, api: str) -> None:
        self.vehicle = vehicle
        self.scenario = scenario
        if scenario not in self.scenario_codes.keys():
            raise InputPropertyError(
                propery="Projected Transport BTU Demand Scenario",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        if vehicle not in self.vehicle_codes.keys():
            raise InputPropertyError(
                propery="Projected Transport BTU Demand",
                valid_options=list(self.vehicle_codes),
                recived_option=vehicle,
            )
        super().__init__(year, api)

    def build_url(self) -> str:
        base_url = "aeo/2023/data/"
        facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={self.vehicle_codes[self.vehicle]}&start=2024&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split("Transportation : Energy Use by Mode : ")[1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)


class FuelCosts(EiaData):
    """Primary fuel cost data."""

    valid_fuels: list[str] = [
        "electricity",
        "gas",
        "coal",
        "lpg",
        "nuclear",
        "heating_oil",
        "propane",
    ]

    def __init__(
        self, fuel: str, year: int, api: str, scenario: str | None = None
    ) -> None:
        self.fuel = fuel
        self.year = year
        self.api = api
        self.scenario = scenario

    def data_creator(self) -> pd.DataFrame:
        """Initializes data extractor."""
        if self.fuel in self.valid_fuels:
            return _FutureCosts(self.fuel, self.year, self.scenario, self.api)
        else:
            raise InputPropertyError(
                propery="Fuel Costs",
                valid_options=self.valid_fuels,
                recived_option=self.fuel,
            )


class _FutureCosts(DataExtractor):
    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    fuel_code_prefix: str = "prce_real_"
    fuel_code_suffix: str = "_NA_NA_y13dlrpmmbtu"

    fuel_codes: ClassVar[dict[str, str]] = {
        "electricity": f"{fuel_code_prefix}ten_NA_elc{fuel_code_suffix}",
        "gas": f"{fuel_code_prefix}elep_NA_ng{fuel_code_suffix}",
        "coal": f"{fuel_code_prefix}elep_NA_stc{fuel_code_suffix}",
        "lpg": f"{fuel_code_prefix}ten_NA_mgs{fuel_code_suffix}",
        "nuclear": f"{fuel_code_prefix}elep_NA_u{fuel_code_suffix}",
        "heating_oil": f"{fuel_code_prefix}elep_NA_dfo{fuel_code_suffix}",
        "propane": f"{fuel_code_prefix}ten_NA_prop{fuel_code_suffix}",
    }

    def __init__(self, fuel: str, year: int, scenario: str, api_key: str) -> None:
        self.fuel = fuel
        self.scenario = scenario
        if scenario not in self.scenario_codes:
            raise InputPropertyError(
                propery="AEO Scenario",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "aeo/2023/data/"
        facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]={self.fuel_codes[self.fuel]}&start=2024&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(
            columns={"seriesName": "series-description", "unit": "units"},
        )
        df["state"] = "U.S."
        df["series-description"] = df["series-description"].map(
            lambda x: x.split(" : ")[-1],
        )
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)

class _InternationalGasTrade(DataExtractor):
    """
    Gets imports/exports by point of entry.

    This filters for ONLY canada and mexico imports/exports
    """

    direction_codes: ClassVar[dict[str, str]] = {
        "imports": "IMI",
        "exports": "EEI",
    }

    points_of_entry = POINTS_OF_ENTRY

    def __init__(self, direction: str, year: int, api_key: str) -> None:
        self.direction = direction
        if self.direction not in list(self.direction_codes):
            raise InputPropertyError(
                propery="Natural Gas International Imports and Exports",
                valid_options=list(self.direction_codes),
                recived_option=direction,
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "natural-gas/move/ist/data/"
        facets = f"frequency=annual&data[0]=value&facets[process][]={self.direction_codes[self.direction]}&start={self.year - 1}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["period"] = pd.to_datetime(df.period).map(lambda x: x.year)

        # extract only canada and mexico trade
        # note, this will still include states extracting
        df = df[
            (df.duoarea.str.endswith("-NCA")) | (df.duoarea.str.endswith("-NMX"))
        ].copy()

        df["from"] = df.duoarea.map(lambda x: x.split("-")[0][1:])  # two letter state
        df["to"] = df["from"].map(self.points_of_entry)

        # drop lng to international locations
        df = df.dropna().copy()

        df["state"] = df["from"] + "-" + df["to"]

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)

    @staticmethod
    def extract_state(description: str) -> str:
        """
        Extracts state from series descripion.

        Input will be in one of the following forms
        - "Massena, NY Natural Gas Pipeline Imports From Canada"
        - "U.S. Natural Gas Pipeline Imports From Mexico"
        """
        try:  # state level
            return description.split(",")[1].split(" ")[1]
        except IndexError:  # country level
            return description.split(" Natural Gas Pipeline")[0]


class _DomesticGasTrade(DataExtractor):
    """
    Gets imports/exports by state.

    Return format of data is a two state code giving from-to values (for
    example, "CA-OR" will represent from California to Oregon
    """

    direction_codes: ClassVar[dict[str, str]] = {
        "imports": "MIR",
        "exports": "MID",
    }

    def __init__(self, direction: str, year: int, api_key: str) -> None:
        self.direction = direction
        if self.direction not in list(self.direction_codes):
            raise InputPropertyError(
                propery="Natural Gas Domestic Imports and Exports",
                valid_options=list(self.direction_codes),
                recived_option=direction,
            )
        super().__init__(year, api_key)

    def build_url(self) -> str:
        base_url = "natural-gas/move/ist/data/"
        facets = f"frequency=annual&data[0]=value&facets[process][]={self.direction_codes[self.direction]}&start={self.year - 1}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["period"] = pd.to_datetime(df.period).map(lambda x: x.year)

        # drop Federal Offshore--Gulf of Mexico Natural Gas Interstate Receipts
        df = df[
            ~(df.duoarea.str.startswith("R3FM-") | df.duoarea.str.endswith("-R3FM"))
        ].copy()

        # drop net movement values
        df = df[~df.duoarea.str.endswith("-Z0S")].copy()

        df["from"] = df["duoarea"].map(lambda x: x.split("-")[0][1:])
        df["to"] = df["duoarea"].map(lambda x: x.split("-")[1][1:])

        df["state"] = df["from"] + "-" + df["to"]

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)

    @staticmethod
    def extract_state(description: str) -> str:
        """
        Extracts state from series descripion.

        Input will be in one of the following forms
        - "Massena, NY Natural Gas Pipeline Imports From Canada"
        - "U.S. Natural Gas Pipeline Imports From Mexico"
        """
        try:  # state level
            return description.split(",")[1].split(" ")[1]
        except IndexError:  # country level
            return description.split(" Natural Gas Pipeline")[0]

class _PowerCapacity(DataExtractor):
    """Extracts projected power sector capacity AEO 2023."""

    # https://www.eia.gov/outlooks/aeo/assumptions/case_descriptions.php
    scenario_codes = AEO_SCENARIOS

    # See here for definitions on what is in each fuel 
    # https://www.eia.gov/outlooks/aeo/data/browser/#/?id=9-AEO2023&cases=ref2023&sourcekey=0
    fuel_codes: ClassVar[dict[str, str]] = {
        "coal": "cl",
        "combined_cycle": "cmc",
        "combustion_turbine": "ctd",
        "distributed": "distgen",
        "diurnal_storage": "diurn",
        "fuel_cells": "fcl",
        "nuclear": "nup",
        "oil_gas_steam": "ong",
        "pumped_storage": "pps",
        "renewables": "rnwbsrc",
        "total": "tot",
    }

    def __init__(self, fuel: str, year: int, scenario: str, api: str):
        super().__init__(year, api)
        self.scenario = scenario
        self.fuel = fuel
        if scenario not in self.scenario_codes.keys():
            raise InputPropertyError(
                propery="Projected Energy Demand Scenario",
                valid_options=list(self.scenario_codes),
                recived_option=scenario,
            )
        if fuel not in self.fuel_codes.keys():
            raise InputPropertyError(
                propery="Projected Energy Demand Sector",
                valid_options=list(self.fuel_codes),
                recived_option=fuel,
            )

    def build_url(self) -> str:
        base_url = "aeo/2023/data/"
        facets = f"frequency=annual&data[0]=value&facets[scenario][]={self.scenario_codes[self.scenario]}&facets[seriesId][]=cap_NA_elep_pow_{self.fuel_codes[self.fuel]}_NA_usa_gw&start=2021&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.period)
        df.index = df.index.year
        df = df.rename(columns={"seriesName": "series-description", "unit": "units"})
        df["state"] = "U.S."
        df = df[["series-description", "value", "units", "state"]].sort_index()
        return self._assign_dtypes(df)

class _StateEmissions(DataExtractor):
    """State Level CO2 Emissions."""

    sector_codes: ClassVar[dict[str, str]] = {
        "commercial": "CC",
        "power": "EC",
        "industrial": "IC",
        "residential": "RC",
        "transport": "TC",
        "total": "TT",
    }

    fuel_codes: ClassVar[dict[str, str]] = {
        "coal": "CO",
        "gas": "NG",
        "oil": "PE",
        "all": "TO",  # coal + gas + oil = all emissions
    }

    def __init__(self, sector: str, fuel: str, year: int, api_key: str) -> None:
        self.sector = sector
        self.fuel = fuel
        if self.sector not in list(self.sector_codes):
            raise InputPropertyError(
                propery="State Level Emissions",
                valid_options=list(self.sector_codes),
                recived_option=sector,
            )
        if self.fuel not in list(self.fuel_codes):
            raise InputPropertyError(
                propery="State Level Emissions",
                valid_options=list(self.fuel_codes),
                recived_option=fuel,
            )
        super().__init__(year, api_key)
        if self.year > 2021:
            logger.debug(f"Emissions data only available until {2021}")
            self.year = 2021

    def build_url(self) -> str:
        base_url = "co2-emissions/co2-emissions-aggregates/data/"
        facets = f"frequency=annual&data[0]=value&facets[sectorId][]={self.sector_codes[self.sector]}&facets[fuelId][]={self.fuel_codes[self.fuel]}&start={self.year}&end={self.year}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
        return f"{API_BASE}{base_url}?api_key={self.api_key}&{facets}"

    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[~(df["state-name"] == "NA")].copy()
        df = df.rename(
            columns={
                "value-units": "units",
                "state-name": "state",
                "sector-name": "series-description",
            },
        )
        df["series-description"] = df["series-description"].str.cat(
            df["fuel-name"],
            sep=" - ",
        )

        df = (
            df[["series-description", "value", "units", "state", "period"]]
            .sort_values(["state", "period"])
            .set_index("period")
        )

        return self._assign_dtypes(df)

if __name__ == "__main__":
    api = ""
    df = FuelCosts("lpg", 2040, api, "reference").get_data(pivot=True)
    print(df)
