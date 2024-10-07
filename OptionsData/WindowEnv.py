import gymnasium as gym
import polars as pl
import pandas as pd
import numpy as np
import pickle


class WindowEnv(gym.Env):
    def __init__(self):
        with open("/Users/jeison/Desktop/OptionsTradingEnv/OptionsData/Data/SPY_options_tickers.pkl", "rb") as file:
            self.tickers = pickle.load(file)

        self.dates = (
            self.tickers
            .with_columns(
                pl.col("expiration_date").str.to_datetime("%Y-%m-%d")
            )
            .select(pl.col("expiration_date"))
            .unique()
            .sort("expiration_date")
            .get_column("expiration_date")
            .to_list()
            # .to_numpy()
            # .flatten()
        )

        self.dates = np.sort(self.dates)

        with open("/Users/jeison/Desktop/OptionsTradingEnv/OptionsData/Data/SPY_prices.pkl", "rb") as file:
            self.spy_prices = pickle.load(file)

        self.spy_prices = (
            self.spy_prices
            .group_by_dynamic('date', every='1d')
            .agg(
                pl.col("open").first(),
                pl.col("close").last(),
                pl.col("high").max(),
                pl.col("low").min()
            )
            .filter(pl.col("date") >= self.dates[0])
            .filter(pl.col('date') <= self.dates[-1])
        )

        print(self.spy_prices)


if __name__ == "__main__":
    WindowEnv()
