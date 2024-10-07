from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import polars as pl
import pandas as pd
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import copy
import pickle
import os
import time
import re

import ray
import torch
import pandas_market_calendars as mcal

from gymnasium.core import ObsType
from gymnasium.utils import seeding
from polygon import RESTClient
from gymnasium import spaces
from concurrent.futures import ThreadPoolExecutor
from OptionsData.StateClass import StateClass
from OptionsData.utils import extract_time_span
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class SingleAssetTradingEnv(gym.Env):
    """An Option Trading Environment for OpenAI Gym"""
    metadata = {'render.modes': ['human']}
    BEST_VALUE = 0

    def __init__(
            self,
            data_file_path: str,
            time_interval: str,
            hmax: int,
            start_date: str,
            end_date: str,
            initial_amount: float,
            reward_scaling: float,
            cooldown: int = 1,
            fee: float = 0.03,
            buy_pct: float = 0,
            sell_pct: float = 0,
            iteration: int = 0,
            make_plots: bool = False,
            print_verbosity: int = 10,
            model_name: str = "",
            mode: str = "",
            data_folder: str = "./Data",
            *args, **kwargs
    ):
        self.data_file_path = data_file_path
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.cooldown = cooldown
        self.fee = fee
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.model_name = model_name
        self.mode = mode
        self.data_folder = data_folder
        self.multiplier, self.timespan, self.every = extract_time_span(time_interval)
        self.buy_pct = buy_pct
        self.sell_pct = sell_pct
        self.iteration = iteration

        with open(self.data_file_path, "rb") as file:
            self.data = pickle.load(file)

        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")
        self.day = 0

        self.data = (
            self.data
            .group_by_dynamic("date", every=self.every)
            .agg(
                pl.col("close").last(),
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("volume").sum()
            )
            .filter((pl.col("date") >= pl.lit(start_date)) & (pl.col('date') <= pl.lit(end_date)))
        )

        window_size = 10
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

        self._valid_dates = self.data.select(pl.col("date")).to_numpy().flatten()

        self.action_space = spaces.Discrete(3)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_dim = 1 + 5 + 1  # balance, close, open, high, low, volume, owned

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,),
                                            dtype=np.float32)

        self.balance, self.owned = self._initialize()

        self.terminal = False
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episodes = 0

        self.rewards_memory = []
        self.action_memory = []
        self.unmod_action_memory = []
        self.state_memory = []
        self.date_memory = [self.day]
        self.asset_memory = [
            self.value
        ]
        self._seed()

    @property
    def current_datetime(self):
        return self._valid_dates[self.day]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episodes}.png")
        plt.close()

    def _get_data(self):
        columns = [name for name in self.data.columns if "norm" in name]
        return np.array(self.data.select(pl.exclude('date')).select(columns).row(self.day))

    def get_close(self, day=None):
        if day is None:
            day = self.day
        return self.data.select("close").to_numpy().flatten()[day]

    @property
    def value(self):
        return self.balance + self.get_close() * self.owned  # balance + close * owned

    @property
    def state(self):
        _state = [self.balance] + self._get_data().tolist() + [self.owned]
        return np.array(_state)

    def _initialize(self):
        return self.initial_amount, 0

    def save_asset_memory(self):
        return pd.DataFrame({
            "date": self.date_memory, "account_value": self.asset_memory
        })

    def save_action_memory(self):
        return pd.DataFrame({
            "date": self.date_memory[:-1], "actions": self.action_memory, "unmod_actions": self.unmod_action_memory
        })

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.day = 0
        self.balance, self.owned = self._initialize()
        self.asset_memory = [self.value]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.action_memory = []
        self.unmod_action_memory = []
        self.date_memory = [self.current_datetime]

        self.episodes += 1

        return self.state, {}

    def step(self, action):
        self.terminal = self.day > len(self._valid_dates) - 3
        if self.terminal:
            if self.make_plots:
                self._make_plot()

            end_total_asset = self.value
            tot_reward = end_total_asset - self.asset_memory[0]

            SingleAssetTradingEnv.BEST_VALUE = tot_reward

            df_total_value = pd.DataFrame({
                "date": self.date_memory, "account_value": self.asset_memory
            })
            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)

            if df_total_value['daily_return'].std() != 0:
                sharpe = df_total_value['daily_return'].mean() / df_total_value['daily_return'].std() * (252 ** 0.5)

            df_rewards = pd.DataFrame({
                "date": self.date_memory[:-1], "account_rewards": self.rewards_memory
            })

            if self.episodes % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episodes}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

                if (self.model_name != "") and (self.mode != ""):
                    df_actions = self.save_action_memory()
                    df_actions.to_csv(
                        "results/actions_{}_{}_{}.csv".format(
                            self.mode, self.model_name, self.iteration
                        )
                    )
                    df_total_value.to_csv(
                        "results/account_value_{}_{}_{}.csv".format(
                            self.mode, self.model_name, self.iteration
                        ),
                        index=False,
                    )
                    df_rewards.to_csv(
                        "results/account_rewards_{}_{}_{}.csv".format(
                            self.mode, self.model_name, self.iteration
                        ),
                        index=False,
                    )
                    plt.plot(self.asset_memory, "r")
                    plt.plot(self.buy_and_hold(), 'b')
                    plt.savefig(
                        "results/account_value_{}_{}_{}.png".format(
                            self.mode, self.model_name, self.iteration
                        )
                    )
                    plt.close()

            return self.state, self.reward, self.terminal, False, {}
        else:
            # action = (action * self.hmax).astype(int)[0]
            action = (action - 1) * self.hmax
            self.unmod_action_memory.append(action)
            begin_total_asset = self.value

            if action > 0:
                new_action = self.buy(action)
                action = new_action

            elif action < 0:
                action = self.sell(action)
            else:
                action = 0

            self.day += 1
            end_total_asset = self.value

            self.action_memory.append(action)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.current_datetime)
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}

    def buy(self, action):
        price = self.get_close() * (1 + self.buy_pct) + self.fee
        available_amount = self.balance // price

        buy_num_shares = min(available_amount, action)
        buy_amount = buy_num_shares * price

        self.balance -= buy_amount
        self.owned += buy_num_shares

        self.cost += (self.get_close() * self.buy_pct + self.fee) * buy_num_shares
        self.trades += 1

        return buy_num_shares

    def sell(self, action):
        sell_num_shares = min(abs(action), self.owned)
        price = self.get_close() * (1 - self.sell_pct) - self.fee
        sell_amount = sell_num_shares * price

        self.balance += sell_amount
        self.owned -= sell_num_shares
        self.cost += (self.get_close() * self.sell_pct + self.fee) * sell_num_shares
        self.trades += 1
        return -sell_num_shares

    def buy_and_hold(self):
        close_prices = self.data.select(pl.col("close")).to_numpy().flatten().tolist()[:self.day]

        owned = self.initial_amount // close_prices[0]
        leftover = self.initial_amount - owned * close_prices[0]
        prices = [self.initial_amount]
        for price in close_prices[1:]:
            prices.append(leftover + price * owned)
        return prices

    def visualize_buy_sell_signals(self):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.date_memory, self.asset_memory, 'r', label='Agent')
        ax.plot(self.date_memory[0:-1], self.buy_and_hold(), 'b', label="B&H")

        buy_signal = []
        sell_signal = []
        for date, action, asset in zip(self.date_memory, self.action_memory, self.asset_memory):
            if action > 0:
                buy_signal.append((date, asset))
            elif action < 0:
                sell_signal.append((date, asset))

        buy_signal = np.array(buy_signal).reshape(-1, 2)
        sell_signal = np.array(sell_signal).reshape(-1, 2)

        ax.scatter(buy_signal[:, 0], buy_signal[:, 1], color='k', marker='^', s=10, label='buy')
        ax.scatter(sell_signal[:, 0], sell_signal[:, 1], color='k', marker='v', s=10, label='sell')

        plt.legend()

        plt.savefig("results/buy_and_hold_signals.png")


def env_generator(config):
    def helper(*args, **kwargs):
        return SingleAssetTradingEnv(**config)

    return helper


if __name__ == "__main__":
    from finrl.agents.stablebaselines3.models import DRLAgent
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.env_checker import check_env

    from finrl.main import check_and_make_directories
    from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR

    from ray.tune.registry import register_env
    from ray.rllib.algorithms import ppo
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.models import ModelCatalog
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.search import optuna, concurrency_limiter
    from ray.tune.registry import get_trainable_cls
    from ray import tune
    import copy

    import ray

    import itertools

    # ray.init()

    # from CustomModels.Moghar import Moghar, MogharTF

    # ModelCatalog.register_custom_model("Moghar", MogharTF)

    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])


    env_kwargs = {
        'data_file_path': "/Users/jeison/Desktop/OptionsTradingEnv/OptionsData/Data/SPY_prices.pkl",
        'ticker_list': ['SPY'],
        'time_interval': "1Day",
        'hmax': 200,
        'start_date': "2005-01-01",
        "end_date": "2017-12-31",
        "initial_amount": 1e4,
        "reward_scaling": 1,
        "strike_delta": 20,
        "expiration_delta": 5,
        "fee": 0.03,
        "cooldown": 1,
        "print_verbosity": 1,
        'API_KEY': "SP7gdSLbEGk_UDZgaeY0V_dGBfQpVULd",
        "mode": "DCA",
        "model_name": "test_run",
        "previously_owned": None

    }

    register_env("train_env", env_generator(env_kwargs))

    wait_times = [1, 2, 5, 10, 20]
    failure_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    buy_delay = [0, 1, 2, 3]
    sell_delay = [0, 1, 2, 3]

    env = env_generator(env_kwargs)()

    prices = env.get_close(slice(0, len(env._valid_dates) + 1, None))

    for wait, failure, bd, sd in itertools.product(wait_times, failure_rate, buy_delay, sell_delay):
        env.reset()
        done = False
        while not done:
            current_price_window = env.get_close(slice(env.day, env.day + wait, None))
            next_price_window = env.get_close(slice(env.day))

    env = env_generator(env_kwargs)()
    env.reset()
    done = False
    while not done:
        next_price = env.get_close(env.day + 1)
        current_price = env.get_close(env.day)

        if current_price < next_price:
            action = 2
        elif current_price > next_price:
            action = 0
        else:
            action = 1

        state, reward, done, _, info = env.step(action)
    env.visualize_buy_sell_signals()
