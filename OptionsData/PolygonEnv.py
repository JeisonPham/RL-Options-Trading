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

from gymnasium.core import ObsType
from gymnasium.utils import seeding
from polygon import RESTClient
from gymnasium import spaces
from concurrent.futures import ThreadPoolExecutor
from OptionsData.StateClass import StateClass
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class PolygonEnv(gym.Env):
    """An Option Trading Environment for OpenAI Gym"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            ticker_list: List[str],
            time_interval: str,
            hmax: int,
            start_date: str,
            end_date: str,
            initial_amount: float,
            reward_scaling: float,
            strike_delta: int,
            expiration_delta: int,
            cooldown: int = 1,
            fee: float = 0.03,
            make_plots: bool = False,
            print_verbosity: int = 10,
            day: int = 0,
            model_name: str = "",
            iteration: int = 0,
            API_KEY: Optional[str] = None,
            API: Optional[RESTClient] = None,
            data_folder: str = "./Data",
            *args, **kwargs
    ):
        self.ticker_list = ticker_list
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

        self.hmax = hmax
        self.start_date = start_date
        self.end_date = end_date
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.strike_delta = strike_delta
        self.expiration_delta = expiration_delta
        self.cooldown = cooldown
        self.fee = fee
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.day = day
        self.model_name = model_name
        self.iteration = iteration

        if API is None:
            self.API = RESTClient(api_key=API_KEY)
        else:
            self.API = API

        self.action_dim = (2, len(self.ticker_list), strike_delta, expiration_delta)

        # self._state = StateClass(
        #     balance=self.initial_amount,
        #     num_tickers=len(self.ticker_list),
        #     strike_delta=self.strike_delta,
        #     expiration_delta=self.expiration_delta,
        #     num_indicators=3
        # )

        self.day = 0
        self.terminal = False
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episodes = 0

        self.rewards_memory = []
        self.action_memory = []
        self.state_memory = []
        self.date_memory = [self.day]
        self._seed()

        self.data = dict()
        for ticker in self.ticker_list:
            file_path = os.path.join(self.data_folder, f"{ticker}_C_processed.csv")
            calls = pl.read_csv(file_path)
            puts = pl.read_csv(file_path.replace("_C_", "_P_"))
            self.data[ticker] = (
                pl.concat([calls, puts], how='vertical')
                .filter(pl.col("date") >= pl.lit(self.start_date))
            )

        self.max_dates = len(self.data[ticker])

        self._state = self._initialize()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self._state().shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=self.action_dim, dtype=np.float32)

        self.asset_memory = [
            self._state.value()
        ]

    @property
    def state(self):
        return self._state()

    def _update(self):
        # sell assets that are no longer tracked in strike price
        current_strike_prices = self._state.strike_prices
        next_strike_prices = self._state.next_strike_prices

        idx = np.isin(current_strike_prices, next_strike_prices)

        call_owned = self._state.call_owned.copy()
        call_sell_actions = call_owned
        call_sell_actions[~idx] = 0

        put_owned = self._state.put_owned.copy()
        put_sell_actions = put_owned
        put_sell_actions[~idx] = 0

        call_sell_actions = self._sell(call_sell_actions, self._state.next_call_prices, "call")
        put_sell_actions = self._sell(put_sell_actions, self._state.next_put_prices, "put")

        return np.array([call_sell_actions, put_sell_actions])

    def _update_state(self):
        state = copy.deepcopy(self._state)

        state.call_prices, state.next_call_prices = self._extract_option_prices("C", self.day)
        state.put_prices, state.next_put_prices = self._extract_option_prices("P", self.day)

        state.strike_prices = self._extract_strike_price("C", self.day)
        state.next_strike_prices = self._extract_strike_price("C", self.day + 1)

        state.expiration_dates = self._extract_expiration_date("C", self.day)
        state.stock_price = self._extract_stock_price("C", self.day)
        state.next_stock_price = self._extract_stock_price("C", self.day + 1)

        return state


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = (self.hmax * action).astype(int)
        self.terminal = self.day >= (self.max_dates - self.expiration_delta)
        if self.terminal:
            if self.make_plots:
                # TODO implement make plot function
                pass

            end_total_asset = self._state.value()
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = end_total_asset - self.asset_memory[0]

            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)

            if df_total_value['daily_return'].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value['daily_return'].mean()
                    / df_total_value['daily_return'].std()
                )

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]

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
                    plt.savefig(
                        "results/account_value_{}_{}_{}.png".format(
                            self.mode, self.model_name, self.iteration
                        )
                    )
                    plt.close()
        else:
            begin_total_asset = self._state.value()

            sell_actions = action.copy()
            sell_actions[sell_actions > 0] = 0
            sell_actions = np.abs(sell_actions)

            buy_actions = action.copy()
            buy_actions[buy_actions < 0] = 0
            buy_actions = np.abs(buy_actions)

            pre_sell_num_options = self._update()
            if np.all(pre_sell_num_options != 0):
                self.action_memory.append(pre_sell_num_options.flatten())
                self.date_memory.append(self.day + 1)
                print("Auto selling options no longer tracked")

            call_sell_actions = self._sell(sell_actions[0], self._state.call_prices, type_="call")
            put_sell_actions = self._sell(sell_actions[1], self._state.put_prices, type_="put")
            sell_num_actions = np.array([call_sell_actions, put_sell_actions]) + pre_sell_num_options
            sell_num_actions *= -1

            call_buy_actions = self._buy(buy_actions[0], self._state.call_prices, type_="call")
            put_buy_actions = self._buy(buy_actions[1], self._state.put_prices, type_="put")
            buy_num_actions = np.array([call_buy_actions, put_buy_actions])

            num_actions = buy_num_actions + sell_num_actions

            self.action_memory.append(num_actions.flatten())

            self.day += 1
            self._state = self._update_state()

            end_total_asset = self._state.value()
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.day)
            self.reward = end_total_asset - begin_total_asset
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}



    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.day = 0
        self._state = self._initialize()

        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = list()
        self.action_memory = list()
        self.asset_memory = [self._state.value()]
        self.date_memory = [self.day]

        self.episodes += 1

        return self.state, {}

    def _initialize(self):
        state = StateClass(
            balance=self.initial_amount,
            ticker_list=self.ticker_list,
            strike_delta=self.strike_delta,
            expiration_delta=self.expiration_delta,
            num_indicators=3
        )

        state.put_owned = np.zeros((len(self.ticker_list), self.strike_delta, self.expiration_delta))
        state.call_owned = np.zeros((len(self.ticker_list), self.strike_delta, self.expiration_delta))

        state.call_prices, state.next_call_prices = self._extract_option_prices("C", self.day)
        state.put_prices, state.next_put_prices = self._extract_option_prices("P", self.day)

        state.strike_prices = self._extract_strike_price("C", self.day)
        state.next_strike_prices = self._extract_strike_price("C", self.day + 1)

        state.expiration_dates = self._extract_expiration_date("C", self.day)
        state.stock_price = self._extract_stock_price("C", self.day)
        state.next_stock_price = self._extract_stock_price("C", self.day + 1)

        return state

    def _sell(self, action: np.ndarray, prices, type_):
        prices = prices * 100
        owned = getattr(self._state, f"{type_}_owned")

        sell_num_options = np.minimum(np.abs(action), owned)
        earned = np.sum(prices * sell_num_options)
        self.cost += self.fee * np.sum(sell_num_options)
        self.trades += np.sum(self.fee * owned)

        self._state.balance += earned

        owned -= sell_num_options
        setattr(self._state, f"{type_}_owned", owned)

        return sell_num_options

    def _buy(self, action: np.ndarray, prices, type_):
        prices = prices * 100
        owned = getattr(self._state, f"{type_}_owned")

        buy_num_shares = np.zeros(action.shape, dtype=int)

        idxes = np.unravel_index(action.flatten().argsort()[::-1], action.shape)
        for idx in zip(*idxes):
            price = (prices[idx] - self.fee) * 100
            act = action[idx]

            if price * act <= 0:
                continue

            buy_num = self._state.balance // (price * act)
            if buy_num == 0 or buy_num >= self.hmax:
                continue

            buy_num_shares[idx] = buy_num
            self._state.balance -= buy_num * price
            self.trades += buy_num

            owned[idx] += buy_num
        setattr(self._state, f"{type_}_owned", owned)
        return buy_num_shares




    def _extract_option_prices(self, type_: str, day: int):
        current_prices_list, next_prices_list = [], []
        for ticker in self.ticker_list:
            data = (
                self.data[ticker]
                .filter(pl.col("contract_type") == pl.lit(type_))
                .sort(pl.col('date'))
            )

            price_columns = [name for name in data.columns if re.search(r"^price_", name)]
            next_price_columns = [name for name in data.columns if re.search(r"^next_price_", name)]

            current_prices = np.array(
                data
                .select(price_columns)
                .row(day)
            ).reshape(self.strike_delta, self.expiration_delta)
            current_prices_list.append(current_prices)

            next_prices = np.array(
                data.select(next_price_columns)
                .row(day)
            ).reshape(self.strike_delta, self.expiration_delta)
            next_prices_list.append(next_prices)
        return np.array(current_prices_list), np.array(next_prices_list)

    def _extract_strike_price(self, type_: str, day: int):
        strike_prices_list = list()
        for ticker in self.ticker_list:
            data = (
                self.data[ticker]
                .filter(pl.col("contract_type") == pl.lit(type_))
                .sort(pl.col("date"))
            )

            strike_columns = [name for name in data.columns if re.search(r"^strike_", name)]
            strike_prices = np.array(
                data
                .select(strike_columns)
                .row(day)
            )
            strike_prices_list.append(strike_prices)
        return np.array(strike_prices_list)

    def _extract_expiration_date(self, type_: str, day: int):
        for ticker in self.ticker_list:
            data = (
                self.data[ticker]
                .filter(pl.col("contract_type") == pl.lit(type_))
                .sort(pl.col("date"))
            )

            date_columns = [name for name in data.columns if re.search(r"^date_", name)]
            dates = np.array(
                data
                .select(date_columns)
                .row(day)
            )
            return dates

    def _extract_stock_price(self, type_: str, day: int):
        stock_prices = []
        for ticker in self.ticker_list:
            data = (
                self.data[ticker]
                .filter(pl.col("contract_type") == pl.lit(type_))
                .sort(pl.col("date"))
            )

            stock_price = data.select("stock_price").row(day, named=True)['stock_price']
            stock_prices.append(stock_price)
        return np.array(stock_prices)


    # def _extract_data_from_data(self):
    #     call_price_list = list()
    #     put_price_list = list()
    #     next_call_price_list = list()
    #     next_put_price_list = list()
    #     strike_columns_list = list()
    #     next_strike_columns_list = list()
    #     stock_price_list = list()
    #
    #     for ticker in self.ticker_list:
    #         call = (
    #             self.data[ticker]
    #             .filter(pl.col("contract_type") == pl.lit("C"))
    #             .sort(pl.col("date"))
    #         )
    #
    #         puts = (
    #             self.data[ticker]
    #             .filter(pl.col("contract_type") == pl.lit("P"))
    #             .sort(pl.col("date"))
    #         )
    #
    #         regexp = re.compile(r"^price_")
    #         price_columns = [name for name in call.columns if regexp.search(name)]
    #
    #         call_prices = np.array(
    #             call
    #             .select(price_columns)
    #             .row(self.day)
    #         ).reshape(self.strike_delta, self.expiration_delta)
    #         call_price_list.append(call_prices)
    #
    #         put_prices = np.array(
    #             puts
    #             .select(price_columns)
    #             .row(self.day)
    #         ).reshape(self.strike_delta, self.expiration_delta)
    #         put_price_list.append(put_prices)
    #
    #         regexp = re.compile(r"^next_price_")
    #         next_price_columns = [name for name in call.columns if regexp.search(name)]
    #
    #         next_call_prices = np.array(
    #             call.select(next_price_columns).row(self.day)
    #         ).reshape(self.strike_delta, self.expiration_delta)
    #         next_call_price_list.append(next_call_prices)
    #
    #         next_put_prices = np.array(
    #             puts.select(next_price_columns).row(self.day)
    #         ).reshape(self.strike_delta, self.expiration_delta)
    #         next_put_price_list.append(next_put_prices)
    #
    #         regexp = re.compile(r"^date_")
    #         date_columns = [name for name in call.columns if regexp.search(name)]
    #         expiration_date = np.array(
    #             call.select(date_columns).row(self.day)
    #         )
    #
    #         stock_price = call.select('stock_price').row(self.day)
    #         stock_price_list.append(stock_price)
    #
    #         strike_columns = [name for name in call.columns if re.search(r"^strike_", name)]
    #         strikes = np.array(call.select(strike_columns).row(self.day))
    #         strike_columns_list.append(strikes)
    #
    #         next_strike_columns = [name for name in call.columns if re.search(r"^strike_", name)]
    #         next_strikes = np.array(call.select(next_strike_columns).row(self.day + 1))
    #         next_strike_columns_list.append(next_strikes)
    #
    #     return (np.array(stock_price_list),
    #             np.array(strike_columns_list),
    #             np.array(next_strike_columns_list),
    #             expiration_date,
    #             np.array(call_price_list),
    #             np.array(put_price_list),
    #             np.array(next_call_price_list),
    #             np.array(next_put_price_list))


if __name__ == "__main__":
    env_kwargs = {
        'ticker_list': ['SPY'],
        'time_interval': "1Day",
        'hmax': 200,
        'start_date': "2023-01-01",
        "end_date": "2024-06-10",
        "initial_amount": 1e3,
        "reward_scaling": 1e-4,
        "strike_delta": 20,
        "expiration_delta": 5,
        "fee": 0.03,
        "cooldown": 1,
        'API_KEY': "SP7gdSLbEGk_UDZgaeY0V_dGBfQpVULd"

    }

    env = PolygonEnv(**env_kwargs)
    for _ in range(100):
        action = env.action_space.sample()
        start = time.time()
        env.step(action)
        print(time.time() - start)
