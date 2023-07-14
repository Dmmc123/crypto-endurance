import yfinance as yf
import pandas as pd
import argparse
import os
from finta import TA
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred


class DataCollector(BaseModel):
    ticker_symbol: str
    fred_api_key: str

    def _get_prices(self) -> pd.DataFrame:
        """
        Retrieve historical stock price data from Yahoo Finance and format it into a Pandas DataFrame.

        Returns:
            pd.DataFrame: The stock price data, including only the date and OHLCV data.
        """
        stock = yf.Ticker(self.ticker_symbol)
        prices = stock.history(period="max", interval="1d").reset_index()
        # leave out date and OHLCV
        prices = prices[["Date", "Open", "High", "Low", "Close", "Volume"]]
        # leave out only info about the day
        prices["Date"] = pd.to_datetime(prices["Date"].dt.date)
        return prices

    @staticmethod
    def _enrich_indicators(features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the stock price data.

        Args:
            features (pd.DataFrame): The stock price data.

        Returns:
            pd.DataFrame: The stock price data with technical indicators appended as additional columns.
        """
        features["MA"] = TA.SMA(features, period=10)
        features["EMA"] = TA.EMA(features, period=10)
        features["WMA"] = TA.WMA(features, period=10)
        features["MFI"] = TA.MFI(features, period=10)
        features["MOM"] = TA.MOM(features, period=10)
        return features.dropna()

    def _enrich_worldwide_data(self, features: pd.DataFrame, world_series_idx: list[str]) -> pd.DataFrame:
        """
        Retrieve worldwide economic data from the Federal Reserve Economic Data (FRED)
        and append it to the stock price data.

        Args:
            features (pd.DataFrame): The stock price data.
            world_series_idx (list[str]): A list of FRED series IDs to retrieve.

        Returns:
            pd.DataFrame: The stock price data with worldwide economic data appended as additional columns.
        """
        # connect to database
        fred = Fred(api_key=self.fred_api_key)
        # join and append each series individually
        for series_id in world_series_idx:
            min_date, max_date = features["Date"].min(), features["Date"].max(),
            feature = fred.get_series(
                series_id=series_id,
                observation_start=min_date,
                observation_end=max_date
            )
            # transform array into dataframe
            feature = feature.reset_index()
            feature.columns = ["Date", series_id]
            # concat features
            features = pd.merge(left=features, right=feature, on="Date", how="inner")
        return features

    def collect(self, world_series_idx: list[str], output_dir: str) -> None:
        """
        Collect and save stock price data, technical indicators, and worldwide economic data for a given ticker symbol.

        Args:
            world_series_idx (list[str]): A list of FRED series IDs to retrieve.
            output_dir (str): The directory in which to save the output CSV file.
        """
        features = self._get_prices()
        features = DataCollector._enrich_indicators(features)
        features = self._enrich_worldwide_data(features, world_series_idx)
        features = DataCollector._postprocess(features)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        features.to_csv(f"{output_dir}/{self.ticker_symbol}.csv", index=True)

    @staticmethod
    def _postprocess(features: pd.DataFrame) -> pd.DataFrame:
        # fill in the missing days
        features.set_index("Date", inplace=True)
        features = features.resample("D").mean()
        rolling_avg = features.rolling(30, min_periods=1, center=True).mean()
        return features.fillna(rolling_avg)


def main():
    # define arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ticker", "-t", required=True, help="4-symbol ticker of asset")
    argparser.add_argument("--output-dir", "-out", required=True, help="Folder for storing downloaded datasets")
    argparser.add_argument("--env", required=False, default=".env", help="Path to environmental file .env")
    args = argparser.parse_args()
    # collect dataset
    load_dotenv(args.env)
    collector = DataCollector(ticker_symbol=args.ticker, fred_api_key=os.environ["fred-api-key"])
    collector.collect(
        world_series_idx=["T10YIE", "SP500", "SOFR180DAYAVG", "IHLIDXUS", "OBMMIC15YF"],
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

