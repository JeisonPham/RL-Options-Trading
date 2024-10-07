import numpy as np
import polars as pl
import pandas as pd
import gymnasium as gym
import pickle
import datetime as dt
import random

from gymnasium.core import ObsType
from gymnasium.utils import seeding

from OptionsData.utils import extract_time_span

from typing import Union, List, Tuple, Dict, Any


class OTRLFramework(gym.Env):
    """An Option Trading Environment for OpenAI Gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: Union[pd.DataFrame, pl.DataFrame, str],
                 start_date: Union[dt.datetime, str],
                 end_date: Union[dt.datetime, str],
                 initial_amount: float = 1e6,
                 threshold: float = 0.01,
                 random_start: bool = False,
                 freq: str = '1Day',
                 commission_rate: float = 0.001,
                 window_size: int = 10
                 ):

        if isinstance(df, str):
            if ".csv" in df:
                df = pd.read_csv(df)
            elif "pkl" in df:
                with open(df, 'rb') as file:
                    df = pickle.load(file)
            else:
                raise ValueError("File path does not have specified extension")

        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        if isinstance(start_date, str):
            start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')

        self.commission_rate = commission_rate
        self.initial_amount = initial_amount
        self.threshold = threshold
        self.random_start = random_start

        # extract multiplier, timespan, and every from freq
        _, _, every = extract_time_span(freq)

        self.data = (
            df
            .group_by_dynamic("date", every=every)
            .agg(
                pl.col("close").last(),
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("volume").sum()
            )
            .filter((pl.col("date") >= pl.lit(start_date)) & (pl.col("date") <= end_date))
        )

        for col in ['close', 'open', 'high', 'low', 'volume']:
            self.data = (
                self.data
                .with_columns(
                    rolling_std=pl.col(col).rolling_std(window_size=window_size),
                    rolling_mean=pl.col(col).rolling_mean(window_size=window_size)
                )
                .with_columns(
                    ((pl.col(col) - pl.col('rolling_mean')) / pl.col('rolling_std'))
                    .alias("norm_" + col)
                )
                .select(pl.exclude("rolling_mean", "rolling_std"))
            )

        self.data = self.data.drop_nulls()

        self.valid_dates = self.data.select("date").to_numpy().flatten()

        if random_start:
            self.start = random.randint(0, len(self.valid_dates))
        else:
            self.start = 0

        self.day = self.start
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,))

        self.balance = initial_amount
        self.owned = 0

        self.terminal = False
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episodes = 0

        self._seed()

    @property
    def current_datetime(self):
        return self.valid_dates[self.day]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.day = self.start
        self.balance = self.initial_amount
        self.owned = 0
        self.cost = 0
        self.reward = 0
        self.trades = 0

        self.episodes += 1

        return self.state, {}

    @property
    def value(self):
        return self.balance + self.get_close() * self.owned

    def get_close(self):
        return self.data.row(self.day, named=True)['close']

    def _get_data(self):
        columns = [name for name in self.data.columns if "norm" in name]
        return np.array(self.data.select(pl.exclude('date')).select(columns).row(self.day))

    @property
    def state(self):
        state = [self.balance] + self._get_data().tolist() + [self.owned]
        return state

    def step(self, action):
        self.terminal = self.day >= (len(self.valid_dates) - 2)
        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}