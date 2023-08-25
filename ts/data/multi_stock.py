import yfinance as yf
import pandas as pd
import requests
import argparse
import tqdm
import os

from concurrent import futures
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path


class MultiAssetDataCollector(BaseModel):
    candidate_symbols_url: str = "https://rest.coinapi.io/v1/symbols"

    def get_top_assets_by_volume(self, output_dir: str, n_assets: int = 500) -> None:
        symbols_df = self._get_candidate_tickers_df()
        symbols_df = self._filter_unavailable_data(df=symbols_df)
        prices_df = self._get_prices_df_by_tickers(tickers=symbols_df.index.tolist()[:n_assets])
        prices_df.to_csv(Path(output_dir) / "prices.csv")

    def _get_candidate_tickers_df(self) -> pd.DataFrame:
        raw_symbol_data = self._get_raw_symbol_data()
        raw_symbol_df = self._candidate_df_from_raw(json_data=raw_symbol_data)
        return raw_symbol_df

    def _get_raw_symbol_data(self) -> dict:
        headers = {"X-CoinAPI-Key": os.getenv("coin-api-key")}
        response = requests.get(self.candidate_symbols_url, headers=headers)
        symbols_info = response.json()
        return symbols_info

    def _candidate_df_from_raw(self, json_data: dict) -> pd.DataFrame:
        symbols = []
        seen = set()
        for symbol_info in json_data:
            if "volume_1mth_usd" in symbol_info \
                    and symbol_info["asset_id_quote"] == "USD" \
                    and symbol_info["asset_id_base"] not in seen \
                    and "USD" not in symbol_info["asset_id_base"] \
                    and "EUR" not in symbol_info["asset_id_base"] \
                    and symbol_info["asset_id_base"] not in {"JW", "LADYS"}:
                symbols.append({
                    "ticker": symbol_info["asset_id_base"],
                    "volume_1mth_usd": symbol_info["volume_1mth_usd"]
                })
                seen.add(symbol_info["asset_id_base"])
        # index df by ticker and sort by trading volume
        symbols_df = pd.DataFrame.from_records(data=symbols)
        symbols_df.set_index("ticker", inplace=True)
        symbols_df.sort_values(by="volume_1mth_usd", inplace=True, ascending=False)
        return symbols_df

    def _filter_unavailable_data(self, df: pd.DataFrame) -> pd.DataFrame:

        def exists_on_yf(ticker: str, out: dict[str, bool]) -> None:
            yf_ticker_name = f"{ticker}-USD"
            stock = yf.Ticker(ticker=yf_ticker_name)
            history: pd.DataFrame = stock.history(period="1d")
            out[ticker] = len(history) > 0

        tickers = df.index.tolist()
        status = {}
        # request 1 day of data for each ticker
        with futures.ThreadPoolExecutor(max_workers=10) as tp:
            threads = [tp.submit(exists_on_yf, ticker=ticker, out=status) for ticker in tickers]
            pbar = tqdm.tqdm(
                iterable=futures.as_completed(threads),
                total=len(threads),
                desc="Checking availability on YF"
            )
            for future in pbar:
                future.result()
        # merge and filter out unavailable tickers
        df["available"] = df.index.map(status)
        df = df[df["available"]]
        return df

    def _get_prices_df_by_tickers(self, tickers: list[str]) -> pd.DataFrame:
        tickers = [f"{ticker}-USD" for ticker in tickers]
        stocks = yf.Tickers(tickers)
        prices_df = stocks.history(period="max", interval="1d")
        return prices_df


def main() -> None:
    # define arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output-dir", "-out", required=True, help="Folder for storing downloaded datasets")
    argparser.add_argument("--env", required=False, default=".env", help="Path to environmental file .env")
    args = argparser.parse_args()
    # collect dataset
    load_dotenv(args.env)
    collector = MultiAssetDataCollector()
    collector.get_top_assets_by_volume(output_dir=args.output_dir)


if __name__ == "__main__":
    main()







