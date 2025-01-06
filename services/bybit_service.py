import gym
import numpy as np
import pandas as pd
from gym import spaces
from pybit.unified_trading import HTTP
import logging
import json


logging.basicConfig(level=logging.INFO)


class BybitService:
    def __init__(
        self,
        api_key,
        api_secret,
        demo=True,
        stop_loss_ratio=0.02,
        take_profit_ratio=0.1,
        risk_per_trade=0.02,
        leverage=50,
    ):
        self.client = HTTP(api_key=api_key, api_secret=api_secret, demo=demo)
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.balance = self.get_account_balance()
        self.data = self.fetch_data()

    def test_connection(self):
        try:
            # Sprawdzamy aktualną cenę rynkową
            self.client.get_tickers(category="linear", symbol="BTCUSDT")
            print("Connection test successful")
        except Exception as e:
            print("Connection test failed:", e)

    def fetch_data_from_file(self, file_path, step):
        """
        Pobiera dane z pliku bybit_data w formacie JSON, formatuje je jak w fetch_data,
        a następnie zwraca odpowiednią część danych w zależności od wartości step.
        """
        try:
            with open(file_path, "r") as file:
                json_data = json.load(file)  # Wczytanie danych z pliku JSON
                data = json_data["data"]  # Uzyskanie dostępu do tablicy 'data'
                total_records = json_data[
                    "total"
                ]  # Uzyskanie całkowitej liczby rekordów

            # Tworzenie DataFrame z danych
            df = pd.DataFrame(
                data,
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
            logging.info(f"Dostępne kolumny w DataFrame: {df.columns.tolist()}")
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
                raise ValueError("Dane z pliku są puste.")

            if df.isnull().values.any():
                raise ValueError("Pobrane dane zawierają NaN.")

            # Dzielimy DataFrame na części po 1000 rekordów
            chunk_size = 1000
            total_chunks = total_records // chunk_size + (
                1 if total_records % chunk_size > 0 else 0
            )

            if step < 1 or step > total_chunks:
                raise ValueError(f"Step musi być w zakresie od 1 do {total_chunks}.")

            # Zwracamy odpowiednią część danych
            if step == 1:
                start_index = 0
                end_index = chunk_size
            else:
                start_index = (step - 1) * chunk_size - 400
                end_index = start_index + chunk_size

            # Upewniamy się, że indeksy są w odpowiednich granicach
            start_index = max(start_index, 0)
            end_index = min(end_index, total_records)

            print(f"step: {step}, start_index: {start_index}, end_index: {end_index}")
            return df.iloc[start_index:end_index]
        except Exception as e:
            logging.error(f"Błąd przy pobieraniu danych z pliku: {e}")
            return pd.DataFrame()

    def fetch_data(
        self, symbol="BTCUSDT", interval="5", limit=500, start=None, end=None
    ):
        try:
            data = self.client.get_kline(
                symbol=symbol, interval=interval, limit=limit, start=start, end=end
            )

            if "result" not in data:
                raise ValueError("Brak wyników w odpowiedzi API.")

            df = pd.DataFrame(data["result"])

            if "list" in df.columns:
                reversed_list = df["list"].to_list()[::-1]  # Odwrócenie listy
                reversed_list.pop()
                df = pd.DataFrame(
                    reversed_list,
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
            return pd.DataFrame()

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
                    self.data = self.data.append(new_row, ignore_index=True)
                    self.data["timestamp"] = pd.to_datetime(
                        self.data["timestamp"], unit="ms"
                    )
                    self.data.set_index("timestamp", inplace=True)
                    # logging.info(f"Added new data: {new_row}")
        else:
            logging.warning("Received message does not contain valid data.")

    def open_position(
        self,
        symbol,
        side,
        quantity,
        current_price=None,
        leverage=50,
        stop_loss_price=None,
        take_profit_price=None,
    ):
        """
        Otwiera pozycję (long/short) na danym symbolu.
        """
        try:
            order = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=quantity,
                price=current_price,
                isLeverage=leverage,
                timeInForce="PostOnly",
                takeProfit=take_profit_price,
                stopLoss=stop_loss_price,
            )

            return order
        except Exception as e:
            logging.error(f"Błąd przy otwieraniu pozycji: {e}")
            return None

    def close_position(self, symbol, side, quantity, price):
        """
        Zamyka otwartą pozycję na danym symbolu.
        """
        try:
            order = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=quantity,
                price=price,
            )
            return order
        except Exception as e:
            logging.error(f"Błąd przy zamykaniu pozycji: {e}")
            return None

    def get_account_balance(self):
        """
        Zwraca saldo konta.
        """
        try:
            balance_info = self.client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT",
            )
            return balance_info
        except Exception as e:
            logging.error(f"Błąd przy pobieraniu salda konta: {e}")
            return None

    def get_open_positions(self, symbol):
        """
        Zwraca otwarte pozycje dla danego symbolu.
        """
        try:
            positions = self.client.get_open_orders(symbol=symbol)
            return positions
        except Exception as e:
            logging.error(f"Błąd przy pobieraniu otwartych pozycji: {e}")
            return None

    def get_orders(self, symbol):
        """
        Pobiera listę zamówień dla danego symbolu.
        """
        try:
            orders = self.client.get_open_orders(
                category="linear", limit=1, symbol=symbol
            )
            if "result" not in orders:
                raise ValueError("Brak wyników w odpowiedzi API.")

            logging.info(f"Pobrano zamówienia dla symbolu {symbol}: {orders['result']}")
            return orders["result"]
        except Exception as e:
            logging.error(f"Błąd przy pobieraniu zamówień dla symbolu {symbol}: {e}")
            return None

    def get_current_price(self, symbol):
        """
        Zwraca bieżącą cenę rynkową dla danego symbolu.
        """
        ticker = self.client.get_tickers(category="linear", symbol=symbol)
        print(ticker)
        return float(ticker["result"]["list"][0]["lastPrice"])

    def calculate_take_profit(self, current_price, position_type):
        """
        Oblicza poziom take profit w zależności od rodzaju pozycji.
        """
        if position_type == "Buy":
            return current_price * (1 + self.take_profit_ratio)
        elif position_type == "Sell":
            return current_price * (1 - self.take_profit_ratio)
        else:
            raise ValueError(f"Nieznany typ pozycji: {position_type}")

    def calculate_stop_loss(self, current_price, stop_loss_distance, position_type):
        """
        Oblicza poziom stop loss w zależności od rodzaju pozycji.
        """
        if position_type == "Buy":
            return current_price - stop_loss_distance
        elif position_type == "Sell":
            return current_price + stop_loss_distance
        else:
            raise ValueError(f"Nieznany typ pozycji: {position_type}")

    def calculate_stop_loss_distance(self, current_price, position_type):
        """
        Oblicza odległość stop lossa od ceny wejścia w zależności od rodzaju pozycji.
        """
        current_price = float(current_price)
        if current_price == 0:
            return 0
        elif position_type == "Buy":
            return current_price * self.stop_loss_ratio
        elif position_type == "Sell":
            return current_price * self.stop_loss_ratio
        else:
            raise ValueError(f"Nieznany typ pozycji: {position_type}")

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
        self, balance, risk_per_trade, stop_loss_distance, value_per_unit
    ):
        """
        Oblicza maksymalną wielkość pozycji na podstawie balansu, ryzyka, i stop lossa.
        """
        max_loss = balance * risk_per_trade

        if stop_loss_distance <= 0 or value_per_unit <= 0:
            raise ValueError(
                "stop_loss_distance i value_per_unit muszą być większe od zera."
            )

        position_size = max_loss / (stop_loss_distance * value_per_unit)

        if position_size >= 10000:
            position_size = 10000

        return round(position_size, 0)

    def get_shares(self):
        return self.shares
