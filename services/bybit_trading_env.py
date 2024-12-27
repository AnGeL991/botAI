import gym
import numpy as np
import pandas as pd
from gym import spaces
from pybit.unified_trading import HTTP
import logging
from services.bybit_service import BybitService
from services.websocket_service import BybitWebSocketService

logging.basicConfig(level=logging.INFO)


class BybitTradingEnv(gym.Env):
    def __init__(
        self,
        api_key,
        api_secret,
        window_size=30,
        symbol="ONDOUSDT",
        interval="15",
        limit=1000,
        leverage=50,
        risk_per_trade=0.02,
        take_profit_ratio=0.05,
        stop_loss_ratio=0.02,
    ):
        super(BybitTradingEnv, self).__init__()

        # Bybit service
        self.bybit_service = BybitService(
            api_key,
            api_secret,
            demo=True,
            leverage=leverage,
            risk_per_trade=risk_per_trade,
            take_profit_ratio=take_profit_ratio,
            stop_loss_ratio=stop_loss_ratio,
        )

        self.symbol = symbol
        self.interval = interval
        self.window_size = window_size
        self.limit = limit
        self.leverage = leverage

        # Zarządzanie ryzykiem
        self.risk_per_trade = risk_per_trade
        self.take_profit_ratio = take_profit_ratio
        self.stop_loss_ratio = stop_loss_ratio

        # Parametry środowiska
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 5), dtype=np.float32
        )

        # Dane rynkowe i stan
        self.data = self.bybit_service.fetch_data(symbol, interval, limit)
        self.new_data_received = False
        self.current_step = 999
        self.balance = 1000
        self.position = (
            None  # Pozycja: {"type": "long", "entry_price": float, "quantity": float}
        )
        self.transaction_history = []

        # Inicjalizacja WebSocket
        self.websocket_service = BybitWebSocketService(
            symbol=self.symbol, on_message_callback=self.message_callback
        )
        try:
            self.websocket_service.start()
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.websocket_service.stop()

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

    def stop(self):
        """Zatrzymuje działanie serwisu WebSocket i środowiska."""
        self.websocket_service.stop()  # Zatrzymanie WebSocket
        logging.info("BybitTradingEnv stopped.")

    def add_data(self, data):
        self.data = data

    def add_new_data(self, new_data):
        """
        Dodaje nowe dane do self.data.
        new_data powinno być DataFrame z odpowiednim formatem.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("new_data musi być typu DataFrame.")

        # Upewnij się, że indeks jest ustawiony na 'timestamp'
        new_data["timestamp"] = pd.to_datetime(new_data["timestamp"], unit="ms")
        new_data.set_index("timestamp", inplace=True)

        # Łączenie z istniejącymi danymi
        self.data = (
            pd.concat([self.data, new_data]).drop_duplicates().reset_index(drop=False)
        )
        logging.info(f"Added new data to self.data. Current size: {len(self.data)}")

    def step(self, action):
        """
        Wykonanie akcji w środowisku.
        """
        assert self.action_space.contains(action), f"Nieprawidłowa akcja: {action}"
        print("data", len(self.data), self.new_data_received)
        if self.new_data_received:
            self.new_data_received = False

        # Aktualizacja stanu środowiska
        self.current_step += 1

        current_price = self.bybit_service.get_current_price(self.symbol)
        logging.info(f"Bieżąca cena: {current_price}")

        # Dodaj logowanie, aby sprawdzić, czy current_price jest poprawne
        if current_price is None or not isinstance(current_price, (int, float)):
            logging.error("Bieżąca cena jest None lub nie jest typu numerycznego.")
        elif np.isnan(current_price):
            logging.error("Bieżąca cena jest NaN.")

        reward = 0
        info = {}

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
                res = self.close_position(current_price, "Buy")
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
                res = self.close_position(current_price, "Sell")
                reward = res[0]
                info["net_profit"] = res[1]
        elif action == 0:  # Hold
            if self.position:
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

        observation = self._get_observation()

        # Logowanie obserwacji
        if np.isnan(observation).any():
            logging.error(
                f"Observation contains NaN at step: {self.current_step}: {observation}"
            )

        # Logowanie wartości w self.position
        if self.position is not None:
            logging.info(
                f"Current position: balance {self.balance} position {self.position}"
            )

        return observation, reward, info

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
        print(f"Position size: {position_size}")
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

        self.bybit_service.open_position(
            symbol=self.symbol,
            side=position_type,
            quantity=position_size,
            current_price=current_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )
        logging.info(f"Opened position: {self.position}")

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

        self.bybit_service.close_position(
            symbol=self.symbol,
            side=position_type,
            quantity=self.position["quantity"],
            price=current_price,
        )

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
        logging.info(f"Closed position: {reason}")

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

        # Upewnij się, że kształt jest zawsze (window_size, 5)
        if window_data.shape[0] < self.window_size:
            padding = self.window_size - window_data.shape[0]
            window_data = np.vstack([np.zeros((padding, 5)), window_data])

        return window_data

    def get_current_price(self):
        """Zwróć bieżącą cenę z danych."""
        if self.data is not None:
            price = self.data["close"].iloc[self.current_step]

            # Sprawdzanie, czy cena jest NaN
            if np.isnan(price):
                raise ValueError(f"Cena dla kroku {self.current_step} jest NaN.")

            return price
        else:
            raise ValueError("Brak danych rynkowych.")
