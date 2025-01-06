import gym
import numpy as np
import pandas as pd
from gym import spaces
from pybit.unified_trading import HTTP
import logging
from services.bybit_service import BybitService
from services.websocket_service import BybitWebSocketService

logging.basicConfig(level=logging.INFO)


class TradingEnv(gym.Env):
    def __init__(
        self,
        bybit_service: BybitService,
        window_size=30,
        symbol="ONDOUSDT",
        interval="15",
        limit=1000,
        leverage=50,
        risk_per_trade=0.02,
        take_profit_ratio=0.05,
        stop_loss_ratio=0.02,
        take_data_from_file=False,
    ):
        super(TradingEnv, self).__init__()

        # Ustawienia
        self.bybit_service = bybit_service
        self.symbol = symbol
        self.interval = interval  # Interwał ustawiony na 5 minut
        self.window_size = window_size  # Rozmiar okna dla scalpingu
        self.limit = limit
        self.leverage = leverage  # Dźwignia (lewarowanie)
        self.is_real_time = False

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
        self.new_data_received = True
        self.current_step =997
        self.balance = 1000  # Początkowy kapitał w USD
        self.position = (
            None  # Pozycja: {"type": "long", "entry_price": float, "quantity": float}
        )
        self.transaction_history = []

        # Inicjalizacja WebSocket
        self.websocket_service = BybitWebSocketService(
            symbol=self.symbol, on_message_callback=self.message_callback
        )

        if take_data_from_file:
            self.data = self.bybit_service.fetch_data_from_file("bybit_data.json", 1)
        else:
            self.data = self.bybit_service.fetch_data(
                self.symbol, self.interval, self.limit
            )  # Pobieranie danych z Bybit

        try:
            self.websocket_service.start()
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.websocket_service.stop()

    def start_websocket(self):
        if not self.websocket_service.is_running():
            self.websocket_service.start()
            logging.info("WebSocket started successfully.")

    def message_callback(self, message):
        """
        Obsługuje przychodzące wiadomości WebSocket.
        Dodaje dane do self.data, jeśli 'confirm' jest ustawione na True.
        """
        if "data" in message and isinstance(message["data"], list):
            for item in message["data"]:
                if item.get("confirm") is True:  # Sprawdzenie, czy 'confirm' jest True
                    new_row = {
                        "timestamp": item["timestamp"],
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "volume": float(item["volume"]),
                        "turnover": float(item["turnover"]),
                    }

                    # Dodanie nowego wiersza do self.data
                    new_row_df = pd.DataFrame(
                        [new_row]
                    )  # Tworzenie DataFrame z nowego wiersza

                    self.data = pd.concat(
                        [self.data, new_row_df], ignore_index=True
                    )  # Użycie pd.concat zamiast append

                    self.data["timestamp"] = pd.to_datetime(
                        self.data["timestamp"], unit="ms"
                    )

                    self.data.set_index("timestamp", inplace=True)
                    self.new_data_received = True

                    logging.info(f"Added new data: {new_row}")
        else:
            logging.warning("Received message does not contain valid data.")

    def set_new_data(self, step):
        self.data = self.bybit_service.fetch_data_from_file("bybit_data.json", step)

    def reset(self):
        self.balance = 1000
        self.current_step = self.window_size  # Ustawienie kroku początkowego dla okna
        self.transaction_history = []  # Resetowanie historii transakcji
        self.position = None
        self.new_data_received = True
        self.is_real_time = False
        return self._get_observation()

    def step(self, action):
        """
        Wykonanie akcji w środowisku.
        """
        assert self.action_space.contains(action), f"Nieprawidłowa akcja: {action}"

        if self.new_data_received and self.is_real_time:
            self.new_data_received = False

        # Aktualizacja stanu środowiska
        self.current_step += 1
        #done = self.current_step >= len(self.data) - 1
        done = False
        current_price = self.get_current_price()
        # logging.info(f"Bieżąca cena: {current_price}")  # Logowanie bieżącej ceny
        reward = 0
        info = {}

        reward, info = self.get_profit_or_loss(current_price)
        observation = self._get_observation()

        if reward > 0:
            return observation, reward, done, info

        # Logika akcji
        if action == 1:  # Kupno (long)
            if self.position is None:

                self.open_position(current_price, position_type="Buy")
                info = {
                    "message": "Open Long",
                    "current_price": current_price,
                    "position": self.position,
                }
            elif self.position and self.position["type"] == "Sell":
                info = {
                    "message": "Close Position " + self.position["type"],
                    "current_price": current_price,
                    "balance": self.balance,
                    "position": self.position,
                    "net_profit": 0,
                }
                res = self.close_position(current_price, position_type="Buy")
                reward = res[0]
                info["net_profit"] = res[1]
        elif action == 2:  # Sprzedaż (short)
            if self.position is None:

                self.open_position(current_price, position_type="Sell")
                info = {
                    "message": "Open Short",
                    "current_price": current_price,
                    "position": self.position,
                }
            elif self.position and self.position["type"] == "Buy":
                info = {
                    "message": "Close Position " + self.position["type"],
                    "current_price": current_price,
                    "balance": self.balance,
                    "position": self.position,
                    "net_profit": 0,
                }
                res = self.close_position(current_price, position_type="Sell")
                reward = res[0]
                info["net_profit"] = res[1]

        elif action == 0:  # Hold
            if self.position is not None:
                info = {
                    "message": "Hold",
                    "current_price": current_price,
                    "entry_price": self.position["entry_price"],
                    "position": self.position,
                }
            else:
                info = {
                    "message": "Hold",
                    "current_price": current_price,
                }

        # Logowanie nagrody
        if np.isnan(reward):
            logging.error(
                f"Reward is NaN for action: {action} at step: {self.current_step}"
            )

        # Dodaj logowanie dla current_price
        if np.isnan(current_price):
            logging.error(f"Current price is NaN at step: {self.current_step}")

        # Logowanie obserwacji
        if np.isnan(observation).any():
            logging.error(
                f"Observation contains NaN at step: {self.current_step}: {observation}"
            )

        # Logowanie wartości w self.position
        # if self.position is not None:
        #   logging.info(
        #       f"Current position: balance {self.balance} position {self.position}"
        #  )

        # print(f"self.data: {len(self.data)} {self.current_step == 998}")

        if not self.is_real_time and self.current_step >= 997:
            self.is_real_time = True
            self.position = None
            self.balance = 1000

        return observation, reward, done, info

    def open_position(self, current_price, position_type):
        """
        Otwieranie pozycji (long/short) z uwzględnieniem wielkości pozycji na podstawie ryzyka.
        """
        stop_loss_distance = self.bybit_service.calculate_stop_loss_distance(
            current_price, position_type
        )

        if stop_loss_distance == 0:
            raise ValueError("Stop loss distance cannot be zero.")

        # Oblicz wielkość pozycji
        position_size = self.bybit_service.calculate_position_size(
            balance=self.balance,
            risk_per_trade=self.risk_per_trade,
            stop_loss_distance=stop_loss_distance,
            value_per_unit=current_price,
        )

        # Walidacja pozycji
        if position_size <= 0:
            logging.error(f"Invalid position size: {position_size}")
            raise ValueError("Position size must be greater than zero.")

        stop_loss = self.bybit_service.calculate_stop_loss(
            current_price, stop_loss_distance, position_type
        )
        take_profit = self.bybit_service.calculate_take_profit(
            current_price, position_type
        )

        # Aktualizacja pozycji
        self.position = {
            "type": position_type,
            "quantity": position_size,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        if self.is_real_time and self.current_step >= 999:
            self.bybit_service.open_position(
                symbol=self.symbol,
                side=position_type,
                quantity=position_size,
                current_price=current_price,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )
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

    def close_position(
        self, current_price, position_type, take_profit=False, stop_loss=False
    ):
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
        if self.position["type"] == "Buy":
            profit = (current_price - entry_price) * quantity
        elif self.position["type"] == "Sell":
            profit = (entry_price - current_price) * quantity

        # Uwzględnij prowizję w wyniku
        net_profit = profit - total_fees

        # Oblicz procentowy zysk
        percentage_profit = (
            (net_profit / position_size) * 100 if position_size > 0 else 0
        )

        # Aktualizacja balansu
        self.balance += round(net_profit, 2)

        if self.is_real_time and self.current_step >= 999:
            self.bybit_service.close_position(
                symbol=self.symbol,
                side=position_type,
                quantity=self.position["quantity"],
                price=current_price,
            )

        # Dodanie do historii
        if stop_loss:
            reason = f"Stop loss {self.position.get('type')} net profit {net_profit} profit {profit} total fees {total_fees}"
        elif take_profit:
            reason = f"Take profit {self.position.get('type')} net profit {net_profit} profit {profit} total fees {total_fees}"
        else:
            reason = f"Close Position {self.position.get('type')} net profit {net_profit} profit {profit} total fees {total_fees}"

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

    def get_profit_or_loss(self, current_price):
        info = {}
        if self.position is None:
            return 0, info
        elif (
            self.position
            and self.position["type"] == "Buy"
            and (current_price >= self.position["take_profit"])
        ) or (
            self.position
            and self.position["type"] == "Sell"
            and (current_price <= self.position["take_profit"])
        ):
            position_type = "Sell" if self.position["type"] == "Buy" else "Buy"
            info = {
                "message": "Close Position with take profit" + self.position["type"],
                "balance": self.balance,
                "position": self.position,
                "net_profit": 0,
            }
            res = self.close_position(
                self.position["take_profit"], position_type, take_profit=True
            )
            reward = res[0]
            info["net_profit"] = res[1]
            self.position = None
            return reward, info
        elif (
            self.position
            and self.position["type"] == "Buy"
            and (current_price <= self.position["stop_loss"])
        ) or (
            self.position
            and self.position["type"] == "Sell"
            and (current_price >= self.position["stop_loss"])
        ):
            position_type = "Sell" if self.position["type"] == "Buy" else "Buy"
            info = {
                "message": "Close Position with stop loss" + self.position["type"],
                "balance": self.balance,
                "position": self.position,
                "net_profit": 0,
            }
            res = self.close_position(
                self.position["stop_loss"], position_type, stop_loss=True
            )
            reward = res[0]
            info["net_profit"] = res[1]
            self.position = None
            return reward, info
        else:
            return 0, info

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
