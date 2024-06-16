import numpy as np
import polars as pl

from typing import Optional, Dict, Any, List, Union


class StateClass:
    def __init__(self,
                 balance: float = 1e3,
                 ticker_list: List[str] = ["SPY"],
                 strike_delta: int = 10,
                 expiration_delta: int = 10,
                 num_indicators: int = 3):
        self._balance: float = balance
        self.ticker_list = ticker_list
        self.num_tickers: int = len(self.ticker_list)
        self.strike_delta: int = strike_delta
        self.expiration_delta: int = expiration_delta
        self.num_indicators: int = num_indicators

        self._stock_price: np.ndarray = np.array([])
        self._next_stock_price: np.ndarray = np.array([])
        self._call_prices: np.ndarray = np.array([])
        self._put_prices: np.ndarray = np.array([])
        self._next_call_prices: np.ndarray = np.array([])
        self._next_put_prices: np.ndarray = np.array([])
        self._call_owned: np.ndarray = np.array([])
        self._put_owned: np.ndarray = np.array([])
        self._technical_indicators: np.ndarray = np.array([])

        self._expiration_dates: Union[List[str], np.ndarray] = [""] * self.expiration_delta
        self._strike_prices: Union[List[float], np.ndarray] = np.zeros((self.num_tickers, self.strike_delta))
        self._next_strike_prices: Union[List[float], np.ndarray] = np.zeros((self.num_tickers, self.strike_delta))

    @property
    def balance(self) -> float:
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        if isinstance(value, float):
            self._balance = value
        else:
            raise TypeError("Balance must be a numeric value")

    @property
    def stock_price(self) -> np.ndarray:
        return self._stock_price

    @stock_price.setter
    def stock_price(self, value: Union[np.ndarray, list]) -> None:
        if isinstance(value, (np.ndarray, list)) and len(value) == self.num_tickers:
            self._stock_price = np.array(value)
        else:
            raise TypeError("Stock price must be a List or numpy array")

    @property
    def next_stock_price(self) -> np.ndarray:
        return self._next_stock_price

    @next_stock_price.setter
    def next_stock_price(self, value: Union[np.ndarray, list]):
        if isinstance(value, (np.ndarray, list)) and len(value) == self.num_tickers:
            self._next_stock_price = np.array(value)
        else:
            raise TypeError("Stock price must be a List or numpy array")


    @property
    def call_prices(self) -> np.ndarray:
        return self._call_prices

    @call_prices.setter
    def call_prices(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and value.shape == (
                self.num_tickers, self.strike_delta, self.expiration_delta):
            self._call_prices = value
        else:
            raise TypeError(
                f"Call Prices must be a numpy array with shape {(self.num_tickers, self.strike_delta, self.expiration_delta)}")

    @property
    def put_prices(self) -> np.ndarray:
        return self._put_prices

    @put_prices.setter
    def put_prices(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and value.shape == (
                self.num_tickers, self.strike_delta, self.expiration_delta):
            self._put_prices = value
        else:
            raise TypeError(
                f"Put Prices must be a numpy array with shape {(self.num_tickers, self.strike_delta, self.expiration_delta)}")

    @property
    def next_call_prices(self) -> np.ndarray:
        return self._next_call_prices

    @next_call_prices.setter
    def next_call_prices(self, value: np.ndarray):
        if isinstance(value, np.ndarray) and value.shape == (
                self.num_tickers, self.strike_delta, self.expiration_delta):
            self._next_call_prices = value
        else:
            raise TypeError(
                f"Next Call Prices must be a numpy array with shape {(self.num_tickers, self.strike_delta, self.expiration_delta)}")

    @property
    def next_put_prices(self):
        return self._next_put_prices

    @next_put_prices.setter
    def next_put_prices(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and value.shape == (
                self.num_tickers, self.strike_delta, self.expiration_delta):
            self._next_put_prices = value
        else:
            raise TypeError(
                f"Next Put Prices must be a numpy array with shape {(self.num_tickers, self.strike_delta, self.expiration_delta)}")

    @property
    def call_owned(self) -> np.ndarray:
        return self._call_owned

    @call_owned.setter
    def call_owned(self, value) -> None:
        if isinstance(value, np.ndarray) and value.shape == (
                self.num_tickers, self.strike_delta, self.expiration_delta):
            assert np.all(value >= 0), "Owned Values must all be greater than zero"
            self._call_owned = value
        else:
            raise TypeError(
                f"Calls Owned must be a numpy array with shape {(self.num_tickers, self.strike_delta, self.expiration_delta)}")

    @property
    def put_owned(self) -> np.ndarray:
        return self._put_owned

    @put_owned.setter
    def put_owned(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and value.shape == (
                self.num_tickers, self.strike_delta, self.expiration_delta):
            assert np.all(value >= 0), "Owned Values must be all greater than zero"
            self._put_owned = value
        else:
            raise TypeError(
                f"Puts Owned must be a numpy array with shape {(self.num_tickers, self.strike_delta, self.expiration_delta)}")

    @property
    def expiration_dates(self) -> np.ndarray:
        return self._expiration_dates

    @expiration_dates.setter
    def expiration_dates(self, value: Union[list, np.ndarray]) -> None:
        if isinstance(value, Union[list, np.ndarray]) and len(value) == self.expiration_delta:
            self._expiration_dates = value
        else:
            raise TypeError("Expiration Dates must be in a list and of size " + str(self.num_tickers))

    @property
    def strike_prices(self) -> np.ndarray:
        return self._strike_prices

    @strike_prices.setter
    def strike_prices(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and value.shape == (self.num_tickers, self.strike_delta):
            self._strike_prices = value
        else:
            raise TypeError("Strike Prices must be in a list and of size " + str((self.num_tickers, self.strike_delta)))

    @property
    def next_strike_prices(self) -> np.ndarray:
        return self._next_strike_prices

    @next_strike_prices.setter
    def next_strike_prices(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and value.shape == (self.num_tickers, self.strike_delta):
            self._next_strike_prices = value
        else:
            raise TypeError("Strike Prices must be in a list and of size " + str((self.num_tickers, self.strike_delta)))

    @property
    def technical_indicators(self) -> np.ndarray:
        return self._technical_indicators

    @technical_indicators.setter
    def technical_indicators(self, value: np.ndarray) -> None:
        if isinstance(value, np.ndarray) and value.shape == (self.num_tickers, self.num_indicators):
            self._technical_indicators = value
        else:
            raise TypeError("Technical Indicators should be of shape " + str((self.num_tickers, self.num_indicators)))

    def __call__(self):
        L = [self.balance]
        L += self.stock_price.flatten().tolist()
        L += self.call_prices.flatten().tolist()
        L + self.put_prices.flatten().tolist()
        L += self.call_owned.flatten().tolist()
        L += self.put_owned.flatten().tolist()
        L += self.technical_indicators.flatten().tolist()

        return np.array(L)

    def value(self):
        return self.balance + np.sum(self.call_prices * self.call_owned) + np.sum(self.put_prices * self.put_owned)
