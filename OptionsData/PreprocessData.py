from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import polars as pl
import pandas as pd
import datetime as dt
import pickle
import os
import pandas_market_calendars as mcal
from tqdm import tqdm

from tqdm import tqdm


class Preprocessor():
    def __init__(self,
                 ticker_list: List[str],
                 start_date: str,
                 end_date: str,
                 time_interval: str,
                 strike_delta: int,
                 expiration_delta: int,
                 data_folder: str = "./Data", *args, **kwargs):

        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.strike_delta = strike_delta
        self.expiration_delta = expiration_delta
        self.data_folder = data_folder

        if time_interval == "1Day":
            self.multiplier, self.timespan = 1, "day"
            every = "1d"
        elif time_interval == "1Hour":
            self.multiplier, self.timespan = 1, "hour"
            every = "1h"
        elif time_interval == "30minute":
            self.multiplier, self.timespan = 30, 'minute'
            every = "30m"
        elif time_interval == "15minute":
            self.multiplier, self.timespan = 15, 'minute'
            every = "15m"
        elif time_interval == "1minute":
            self.multiplier, self.timespan = 1, "minute"
            every = "1m"
        else:
            raise ValueError("time_interval can only be `1Day` or `1Hour` or `30minute` or `15minute` or `1minute`")

        self.every = every

        self.options_history = dict()
        self.options_tickers = dict()
        self.stock_prices = dict()

        for ticker in self.ticker_list:
            option_tickers_file_path = os.path.join(data_folder, f"{ticker}_options_tickers.pkl")
            with open(option_tickers_file_path, "rb") as file:
                df = pickle.load(file)

            df = (
                df
                .filter(pl.col('ticker').str.contains("C"))
                .group_by(['expiration_date', 'strike_price'])
                .agg(pl.col("ticker").first())
                .sort("expiration_date")
                .pivot(index='strike_price', columns='expiration_date', values='ticker')
                .sort('strike_price')
            )
            self.options_tickers[ticker] = df

            options_history_file_path = os.path.join(data_folder, f"{ticker}_options_history.pkl")
            with open(options_history_file_path, "rb") as file:
                df = pickle.load(file)
            self.options_history[ticker] = df

            stock_prices_file_path = os.path.join(data_folder, f"{ticker}_prices.pkl")
            with open(stock_prices_file_path, "rb") as file:
                df = pickle.load(file)
            self.stock_prices[ticker] = (
                df
                .sort("date")
                .group_by_dynamic("date", every=self.every)
                .agg(
                    pl.col("open").first(),
                    pl.col("close").last(),
                    pl.col("high").max(),
                    pl.col("low").min(),
                    pl.col("volume").sum()
                )
            )

        self.day = 0
        nyse = mcal.get_calendar("NYSE")
        early = nyse.schedule(start_date=start_date, end_date=end_date)

        if time_interval == "1Day":
            self._valid_dates = [date.replace(hour=0, tzinfo=None) for date in mcal.date_range(early, frequency="1D").to_pydatetime()]
        else:
            self._valid_dates = mcal.date_range(early, frequency=f"{self.multiplier}{self.timespan[0]}", closed='left') - dt.timedelta(hours=5)
            self._valid_dates = [date.replace(tzinfo=None) for date in self._valid_dates.to_pydatetime()]

    def get_stock_price_strike_prices_expiration_dates(self, ticker, date):
        date_str = date.strftime("%Y-%m-%d")
        price = (
            self.stock_prices[ticker]
            .filter(pl.col('date') >= pl.lit(date))
            .row(0, named=True)
        )['close']

        strike_prices = (
            self.options_tickers[ticker]
            .select("strike_price")
            .to_numpy()
            .flatten()
        )

        idx = np.argmin(np.abs(price - strike_prices))
        half_delta = self.strike_delta // 2
        strike_prices = strike_prices[idx - half_delta: idx + half_delta]

        columns = np.array(self.options_tickers[ticker].columns)
        column_index = np.argmax(columns == date_str)
        columns = columns[column_index: column_index + self.expiration_delta]

        return price, strike_prices, columns

    def get_options_price(self, date, underlying_ticker,
                          strike_prices, expiration_date, type_,
                          next=False):
        date_str = date.strftime("%Y-%m-%d")
        avail_tickers = (
            self.options_tickers[underlying_ticker]
            .filter(pl.col("strike_price").is_in(strike_prices))
            .select(expiration_date)
        )

        def helper(ticker, underlying):
            if not isinstance(ticker, str):
                return 0

            ticker = ticker.replace("C", type_)
            if ticker not in self.options_history[underlying]:
                return 0

            df = self.options_history[underlying][ticker]
            df = (
                df
                .group_by_dynamic("date", every=self.every)
                .agg(
                    pl.col("close").last(),
                    pl.col("open").first(),
                    pl.col("high").max(),
                    pl.col("low").min()
                )
                .filter(pl.col("date") >= date)
                .sort("date")
            )
            if len(df) == 0:
                return 0
            df = df.row(0, named=True)
            return df['close']

        option_tickers = avail_tickers.to_numpy().flatten()
        values = {
            f"{'next_' if next else ''}date": date
        }
        for i, ticker in enumerate(option_tickers):
            values[f"{'next_' if next else ''}price_{i + 1}"] = helper(ticker, underlying_ticker)

        for i, strike in enumerate(strike_prices):
            values[f"{'next_' if next else ''}strike_{i}"] = strike

        for i, date in enumerate(expiration_date):
            values[f"{'next_' if next else ''}date_{i}"] = date
        return values

    def preprocess_underlying_ticker(self, underlying_ticker, type_="C"):
        results = []
        dates = list(zip(self._valid_dates[:-1], self._valid_dates[1:]))

        def helper(arg):
            current_date, next_date = arg
            price, strike_prices, expiration_dates = self.get_stock_price_strike_prices_expiration_dates(
                underlying_ticker, current_date)

            result = self.get_options_price(current_date, underlying_ticker, strike_prices, expiration_dates, type_)
            result['stock_price'] = price
            result['contract_type'] = type_
            next_result = self.get_options_price(next_date, underlying_ticker, strike_prices, expiration_dates,
                                                 type_, next=True)

            price, _, _ = self.get_stock_price_strike_prices_expiration_dates(underlying_ticker, next_date)
            next_result['next_stock_price'] = price
            result = pd.DataFrame(result, index=[0])
            next_result = pd.DataFrame(next_result, index=[0])
            return pd.concat([result, next_result], axis=1)

        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(tqdm(executor.map(helper, dates), total=len(dates)))

        df = pd.concat(results, axis=0).reset_index(drop=True)
        file_path = os.path.join(self.data_folder, f"{underlying_ticker}_{type_}_processed.csv")
        df.to_csv(file_path, index=False)


if __name__ == "__main__":
    env_kwargs = {
        'ticker_list': ['SPY'],
        'time_interval': "1Day",
        'hmax': 200,
        "start_date": "2023-01-01",
        "end_date": "2024-06-10",
        'initial_amount': 1e3,
        'reward_scaling': 1e-4,
        'strike_delta': 20,
        'expiration_delta': 5,
        'fee': 0.03,
        'cooldown': 1,
        'API_KEY': "SP7gdSLbEGk_UDZgaeY0V_dGBfQpVULd"

    }

    processor = Preprocessor(**env_kwargs)
    processor.preprocess_underlying_ticker("SPY")
    processor.preprocess_underlying_ticker("SPY", type_="P")
