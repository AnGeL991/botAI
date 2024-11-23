from pybit.unified_trading import HTTP
import gym
import numpy as np
import pandas as pd
from gym import spaces


class TradingEnv(gym.Env):
    def __init__(
        self, client: HTTP, symbol="BTCUSDT", interval="5", limit=100, leverage=10
    ):
        super(TradingEnv, self).__init__()

        # Ustawienia
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.leverage = leverage  # Dźwignia (lewarowanie)

        self.data = self.fetch_data()  # Pobieranie danych z Bybit
        self.current_step = 0
        self.balance = 1000  # Początkowy stan konta
        self.shares = 0  # Początkowa liczba kontraktów
        self.margin = 0  # Początkowy margin
        self.entry_price = 0  # Cena zakupu (lub sprzedaży)

        # Akcje: 0 - Kup, 1 - Sprzedaj, 2 - Trzymaj
        self.action_space = spaces.Discrete(3)

        # Obserwacja: cena, saldo, liczba kontraktów, margin, cena zakupu
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )

    def fetch_data(self):
        try:
            data = self.client.get_kline(
                symbol=self.symbol, interval=self.interval, limit=self.limit
            )

            if "result" not in data:
                raise ValueError("Brak wyników w odpowiedzi API.")

            df = pd.DataFrame(data["result"])

            if "list" in df.columns:
                df = pd.DataFrame(
                    df["list"].to_list(),
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "turnover",
                    ],
                )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            df = df.astype(
                {
                    "open": "float",
                    "high": "float",
                    "low": "float",
                    "close": "float",
                    "volume": "float",
                    "turnover": "float",
                }
            )

            if df.empty:
                raise ValueError("Dane z Bybit są puste.")

            return df
        except Exception as e:
            print(f"Błąd przy pobieraniu danych: {e}")
            return pd.DataFrame()  # Zwróć pusty DataFrame w przypadku błędu

    def reset(self):
        self.balance = 1000
        self.shares = 0
        self.margin = 0
        self.entry_price = 0
        self.current_step = 0
        return self.get_observation()

    def step(self, action):
        if self.data is None:
            print("Błąd: Brak danych rynkowych.")
            return (
                np.zeros(5),
                0,
                True,
                {},
            )  # Zwróć domyślne wartości, aby uniknąć błędu

        self.current_step += 1
        done = False
        if self.current_step >= len(self.data) - 1:
            done = True

        current_price = self.data["close"].iloc[self.current_step]
        prev_price = self.data["close"].iloc[self.current_step - 1]
        reward = 0

        # Dźwignia: Obliczanie, ile kontraktów można otworzyć
        leverage_amount = self.balance * self.leverage

        if action == 0:  # Kupno kontraktów
            if self.balance >= current_price:
                self.shares += leverage_amount // current_price
                self.balance -= self.shares * current_price  # Pobieramy saldo
                self.margin = self.shares * current_price  # Margin
                self.entry_price = current_price
        elif action == 1:  # Sprzedaż kontraktów
            if self.shares > 0:
                self.shares -= leverage_amount // current_price
                self.balance += self.shares * current_price  # Dodajemy do salda
                self.margin = self.shares * current_price
        elif action == 2:  # Trzymanie kontraktów
            pass

        reward = (
            self.balance + self.shares * current_price - 1000
        )  # Nagroda = saldo + kontrakty - początkowe saldo
        return self.get_observation(), reward, done, {}

    def get_observation(self):
        if self.data is None:
            print("Błąd: Brak danych rynkowych.")
            return np.zeros(5)  # Zwróć pustą tablicę, aby uniknąć błędu

        return np.array(
            [
                self.data["close"].iloc[self.current_step],
                self.balance,
                self.shares,
                self.margin,
                self.entry_price,
            ]
        )

    def render(self):
        print(
            f"Step: {self.current_step}, Balance: {self.balance}, Shares: {self.shares}, Price: {self.data['close'].iloc[self.current_step]}"
        )
