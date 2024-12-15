import gym
import numpy as np
import pandas as pd
from gym import spaces
from pybit.unified_trading import HTTP
import logging

logging.basicConfig(level=logging.INFO)


class TradingEnv(gym.Env):
    def __init__(
        self,
        client: HTTP,
        window_size=30,
        symbol="ONDOUSDT",
        interval="15",
        limit=1000,
        leverage=50,
        risk_per_trade=0.02,
        take_profit_ratio=0.05,
        stop_loss_ratio=0.02,
    ):
        super(TradingEnv, self).__init__()

        # Ustawienia
        self.client = client
        self.symbol = symbol
        self.interval = interval  # Interwał ustawiony na 5 minut
        self.window_size = window_size  # Rozmiar okna dla scalpingu
        self.limit = limit
        self.leverage = leverage  # Dźwignia (lewarowanie)

        # Zarządzanie ryzykiem
        self.risk_per_trade = risk_per_trade  # Ryzyko na transakcję jako % kapitału
        self.take_profit_ratio = take_profit_ratio  # Take profit w % (np. 5%)
        self.stop_loss_ratio = stop_loss_ratio  # Stop loss w % (np. 2%)

        # Parametry środowiska
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 5), dtype=np.float32
        )

        # Dane rynkowe i stan
        self.data = None
        self.current_step = 0
        self.balance = 1000  # Początkowy kapitał w USD
        self.position = (
            None  # Pozycja: {"type": "long", "entry_price": float, "quantity": float}
        )
        self.transaction_history = []

        self.data = self.fetch_data()  # Pobieranie danych z Bybit

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

            if df.isnull().values.any():
                raise ValueError("Pobrane dane zawierają NaN.")

            return df
        except Exception as e:
            print(f"Błąd przy pobieraniu danych: {e}")
            return pd.DataFrame()  # Zwróć pusty DataFrame w przypadku błędu

    def reset(self):
        self.balance = 1000
        self.shares = 0
        self.margin = 0
        self.entry_price = 0
        self.buy_price = None  # Cena zakupu
        self.current_step = self.window_size  # Ustawienie kroku początkowego dla okna
        self.transaction_history = []  # Resetowanie historii transakcji
        self.position = None
        return self._get_observation()

    def step(self, action):
        """
        Wykonanie akcji w środowisku.
        """
        assert self.action_space.contains(action), f"Nieprawidłowa akcja: {action}"

        # Aktualizacja stanu środowiska
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        current_price = self.get_current_price()
        # logging.info(f"Bieżąca cena: {current_price}")  # Logowanie bieżącej ceny
        reward = 0
        info = {}

        # Logika akcji
        if action == 1:  # Kupno (long)
            if self.position is None:

                self.open_position(current_price, position_type="long")
                info = {
                    "message": "Open Long",
                    "current_price": current_price,
                    "position": self.position,
                }
            elif self.position and self.position["type"] == "short":
                info = {
                    "message": "Close Position " + self.position["type"],
                    "current_price": current_price,
                    "balance": self.balance,
                    "position": self.position,
                    "net_profit": 0,
                }
                res = self.close_position(current_price)
                reward = res[0]
                info["net_profit"] = res[1]
        elif action == 2:  # Sprzedaż (short)
            if self.position is None:

                self.open_position(current_price, position_type="short")
                info = {
                    "message": "Open Short",
                    "current_price": current_price,
                    "position": self.position,
                }
            elif self.position and self.position["type"] == "long":
                info = {
                    "message": "Close Position " + self.position["type"],
                    "current_price": current_price,
                    "balance": self.balance,
                    "position": self.position,
                    "net_profit": 0,
                }
                res = self.close_position(current_price)
                reward = res[0]
                info["net_profit"] = res[1]
        elif action == 0:  # Hold
            if self.position:
                if (
                    self.position["type"] == "long"
                    and (
                        current_price >= self.position["take_profit"]
                        or current_price <= self.position["stop_loss"]
                    )
                ) or (
                    self.position["type"] == "short"
                    and (
                        current_price <= self.position["take_profit"]
                        or current_price >= self.position["stop_loss"]
                    )
                ):
                    info = {
                        "message": "Close Position " + self.position["type"],
                        "balance": self.balance,
                        "position": self.position,
                        "net_profit": 0,
                    }
                    res = self.close_position(current_price)
                    reward = res[0]
                    info["net_profit"] = res[1]
                else:
                    info = {
                        "message": "Hold",
                        "current_price": current_price,
                        "entry_price": self.position["entry_price"],
                        "position": self.position,
                    }

        # Logowanie nagrody
        if np.isnan(reward):
            logging.error(
                f"Reward is NaN for action: {action} at step: {self.current_step}"
            )

        # Dodaj logowanie dla current_price
        if np.isnan(current_price):
            logging.error(f"Current price is NaN at step: {self.current_step}")

        observation = self._get_observation()

        # Logowanie obserwacji
        if np.isnan(observation).any():
            logging.error(
                f"Observation contains NaN at step: {self.current_step}: {observation}"
            )

        # Logowanie wartości w self.position
        # if self.position is not None:
        #  logging.info(
        #      f"Current position: balance {self.balance} position {self.position}"
        # )

        return observation, reward, done, info

    def calculate_quantity_and_risk(self, current_price):
        """
        Oblicza liczbę kontraktów do otwarcia na podstawie kapitału, dźwigni i poziomu ryzyka.
        """
        if current_price == 0:
            raise ValueError("Current price cannot be zero.")

        max_risk = self.balance * self.risk_per_trade  # Maksymalna strata w USD

        quantity = (max_risk / (self.stop_loss_ratio * current_price)) * self.leverage

        # Logowanie wartości
        logging.info(
            f"Current Price: {current_price}, Max Risk: {max_risk}, Quantity: {quantity}"
        )

        return quantity, max_risk

    def calculate_position_size(
        self, balance, risk_per_trade, leverage, stop_loss_distance, value_per_unit
    ):
        """
        Oblicza maksymalną wielkość pozycji na podstawie balansu, ryzyka, dźwigni i stop lossa.
        """
        # 1. Maksymalna strata, którą akceptujemy
        max_loss = balance * risk_per_trade  # np. 2% z balansu

        # Logowanie wartości wejściowych
        # logging.info(
        #    f"Balance: {balance}, Risk per trade: {risk_per_trade}, "
        #    f"Leverage: {leverage}, Stop loss distance: {stop_loss_distance}, "
        #     f"Value per unit: {value_per_unit}"
        # )

        # 2. Sprawdzenie, czy stop_loss_distance i value_per_unit są różne od zera
        if stop_loss_distance <= 0 or value_per_unit <= 0:
            raise ValueError(
                "stop_loss_distance i value_per_unit muszą być większe od zera."
            )

        # 3. Obliczenie wielkości pozycji
        position_size = max_loss / (stop_loss_distance * value_per_unit)

        # Logowanie obliczonej wielkości pozycji
        # logging.info(f"Calculated position size: {position_size}")

        return round(position_size, 2)

    def open_position(self, current_price, position_type):
        """
        Otwieranie pozycji (long/short) z uwzględnieniem wielkości pozycji na podstawie ryzyka.
        """
        stop_loss_distance = self.calculate_stop_loss_distance(
            current_price, position_type
        )
        if stop_loss_distance == 0:
            raise ValueError("Stop loss distance cannot be zero.")

        # Oblicz wielkość pozycji
        position_size = self.calculate_position_size(
            balance=self.balance,
            risk_per_trade=self.risk_per_trade,
            leverage=self.leverage,
            stop_loss_distance=stop_loss_distance,
            value_per_unit=current_price,
        )

        # Walidacja pozycji
        if position_size <= 0:
            logging.error(f"Invalid position size: {position_size}")
            raise ValueError("Position size must be greater than zero.")

        # Aktualizacja pozycji
        self.position = {
            "type": position_type,
            "quantity": position_size,
            "entry_price": current_price,
            "stop_loss": self.calculate_stop_loss(
                current_price, stop_loss_distance, position_type
            ),
            "take_profit": self.calculate_take_profit(current_price, position_type),
        }

        # logging.info(f"Opened position: {self.position}")

        self.transaction_history.append(
            (
                position_type,
                current_price,
                position_size,
                self.balance,
            )
        )

        return position_size

    def close_position(self, current_price, take_profit=False, stop_loss=False):
        """
        Zamyka otwartą pozycję i oblicza wynik z uwzględnieniem prowizji.
        Zwraca procentowy zysk z transakcji.
        """
        if self.position is None:
            return 0

        # Oblicz podstawowe parametry
        entry_price = self.position["entry_price"]
        quantity = self.position.get("quantity", 0)
        position_size = entry_price * quantity  # Wielkość pozycji w dolarach

        # Ustawienie stawek prowizji (dostosuj do swoich stawek)
        fee_rate = 0.0006  # 0.06% dla market takers
        total_fees = position_size * fee_rate * 2  # Dla otwarcia i zamknięcia pozycji

        # Oblicz wynik brutto
        if self.position["type"] == "long":
            profit = (current_price - entry_price) * quantity
        elif self.position["type"] == "short":
            profit = (entry_price - current_price) * quantity

        # Uwzględnij prowizję w wyniku
        net_profit = profit - total_fees

        # Oblicz procentowy zysk
        percentage_profit = (
            (net_profit / position_size) * 100 if position_size > 0 else 0
        )

        # Aktualizacja balansu
        self.balance += round(net_profit, 2)

        # Dodanie do historii
        reason = (
            "Take Profit"
            if take_profit
            else (
                "Stop Loss"
                if stop_loss
                else f"Close Position {self.position.get('type')} net profit {net_profit} profit {profit} total fees {total_fees}"
            )
        )

        self.transaction_history.append(
            (
                reason,
                current_price,
                quantity,
                self.balance,
                round(net_profit, 2),
            )
        )

        self.position = None  # Reset pozycji
        return percentage_profit, net_profit  # Zwróć procentowy zysk

    def _get_observation(self):
        """
        Zwraca obserwację złożoną z ostatnich window_size kroków.
        """
        if self.current_step < self.window_size:
            padding = self.window_size - self.current_step
            window_data = np.vstack(
                [
                    np.zeros((padding, 5)),
                    self.data.iloc[: self.current_step][
                        ["open", "high", "low", "close", "volume"]
                    ].values,
                ]
            )
        else:
            window_data = self.data.iloc[
                self.current_step - self.window_size : self.current_step
            ][["open", "high", "low", "close", "volume"]].values

        return window_data

    def calculate_take_profit(self, current_price, position_type):
        """
        Oblicza poziom take profit w zależności od rodzaju pozycji.
        """
        if position_type == "long":
            return current_price * (1 + self.take_profit_ratio)
        elif position_type == "short":
            return current_price * (1 - self.take_profit_ratio)
        else:
            raise ValueError(f"Nieznany typ pozycji: {position_type}")

    def calculate_stop_loss(self, current_price, stop_loss_distance, position_type):
        """
        Oblicza poziom stop loss w zależności od rodzaju pozycji.
        """
        if position_type == "long":
            return current_price - stop_loss_distance
        elif position_type == "short":
            return current_price + stop_loss_distance
        else:
            raise ValueError(f"Nieznany typ pozycji: {position_type}")

    def calculate_stop_loss_distance(self, current_price, position_type):
        """
        Oblicza odległość stop lossa od ceny wejścia w zależności od rodzaju pozycji.
        """
        if current_price == 0:
            return 0  # Zwróć 0, aby uniknąć dzielenia przez 0
        elif position_type == "long":
            return current_price * self.stop_loss_ratio
        elif position_type == "short":
            return current_price * self.stop_loss_ratio
        else:
            raise ValueError(f"Nieznany typ pozycji: {position_type}")

    def get_current_price(self):
        """Zwróć bieżącą cenę z danych."""
        if self.data is not None:
            # Sprawdzenie, czy current_step mieści się w dostępnych danych
            if self.current_step >= len(self.data):
                raise ValueError(
                    f"current_step ({self.current_step}) przekroczył liczbę dostępnych danych ({len(self.data)})."
                )

            price = self.data["close"].iloc[self.current_step]

            # Sprawdzanie, czy cena jest NaN
            if np.isnan(price):
                raise ValueError(f"Cena dla kroku {self.current_step} jest NaN.")

            return price
        else:
            raise ValueError("Brak danych rynkowych.")

    def get_transaction_history(self):
        """Zwróć historię transakcji."""
        return self.transaction_history

    # Metody dostępowe do stanu
    def get_balance(self):
        return self.balance

    def get_profit_or_loss(self):
        return self.profit_or_loss

    def set_balance(self, amount):
        self.balance = amount

    def get_shares(self):
        return self.shares

    def set_shares(self, shares):
        self.shares = shares

    def get_buy_price(self):
        """Zwróć cenę zakupu kontraktów (jeśli zakupiono)."""
        return self.buy_price

    def set_buy_price(self, price):
        """Ustaw cenę zakupu kontraktów."""
        self.buy_price = price
