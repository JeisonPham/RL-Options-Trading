import numpy as np
import polars as pl
import pickle
import os
from polygon import RESTClient
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import datetime as dt


class PolygonDownloader:
    def __init__(self, start_date, end_date, API_KEY):
        self.start_date = start_date
        self.end_date = end_date

        self.API = RESTClient(api_key=API_KEY)

    def download_available_options_contracts(self, underlying_symbol, file_path="./Data/"):
        data = [agg.__dict__ for agg in self.API.list_options_contracts(
            underlying_ticker=underlying_symbol,
            contract_type='call',
            expiration_date_gte=self.start_date,
            expiration_date_lte=self.end_date,
            expired=True,
        )]
        df = pl.DataFrame(data)
        df_puts = df.with_columns(
            contract_type=pl.lit("put"),
            ticker=pl.col('ticker').str.replace("C", "P")
        )

        df = pl.concat([df_puts, df], how='vertical')

        os.makedirs(file_path, exist_ok=True)

        file_name = os.path.join(file_path, f"{underlying_symbol}_options_tickers.pkl")
        with open(file_name, "wb") as file:
            pickle.dump(df, file)

    def download_options_history(self, underlying_symbol, file_path="./Data"):
        file_name = os.path.join(file_path, f"{underlying_symbol}_options_tickers.pkl")
        if os.path.exists(file_name):
            with open(file_name, "rb") as file:
                tickers = pickle.load(file)
        else:
            self.download_available_options_contracts(underlying_symbol, file_path)

        history = dict()

        def helper_func(ticker):
            data = [agg.__dict__ for agg in self.API.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='minute',
                from_=self.start_date,
                to=self.end_date
            )]

            if len(data) == 0:
                return 0

            df = pl.DataFrame(data).with_columns(
                date=pl.from_epoch('timestamp', time_unit='ms')
            )
            return df

        tickers_list = tickers.select("ticker").to_numpy().flatten()
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(helper_func, tickers_list), total=len(tickers_list)))

        for ticker, result in zip(tickers_list, results):
            if isinstance(result, pl.DataFrame):
                history[ticker] = result

        file_name = os.path.join(file_path, f"{underlying_symbol}_options_history.pkl")
        with open(file_name, "wb") as file:
            pickle.dump(history, file)

    def download_stock_history(self, underlying_symbol, file_path="./Data"):
        file_name = os.path.join(file_path, f"{underlying_symbol}_prices.pkl")
        if os.path.exists(file_name):
            with open(file_name, "rb") as file:
                tickers = pickle.load(file)

            return

        start = dt.datetime.strptime(self.start_date, "%Y-%m-%d")
        end = dt.datetime.strptime(self.end_date, "%Y-%m-%d")

        dates = pl.datetime_range(start=start, end=end, interval="1d", eager=True).to_list()

        temp_df = pl.DataFrame()

        for date in tqdm(dates):
            data = [agg.__dict__ for agg in self.API.get_aggs(
                ticker=underlying_symbol,
                multiplier=1,
                timespan='minute',
                from_=date.strftime("%Y-%m-%d"),
                to=date.strftime("%Y-%m-%d"),
                limit=None
            )]

            if len(data) == 0:
                continue

            df = pl.DataFrame(data).with_columns(
                date=pl.from_epoch('timestamp', time_unit='ms')
            )
            temp_df = pl.concat([temp_df, df])

        with open(file_name, "wb") as file:
            pickle.dump(temp_df, file)


if __name__ == "__main__":
    downloader = PolygonDownloader("2023-01-01", "2024-06-14", "SP7gdSLbEGk_UDZgaeY0V_dGBfQpVULd")
    downloader.download_available_options_contracts("SPY")
    # downloader.download_options_history("SPY")
    # downloader.download_stock_history("SPY")