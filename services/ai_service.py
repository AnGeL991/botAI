from pybit.unified_trading import HTTP
import pandas as pd


class AIStrategy:
    def __init__(self, client: HTTP):
        self.client = client

    def fetch_and_process_data(self):
        """
        Pobieranie danych historycznych z Bybit i obliczanie średnich kroczących.
        """
        # Zamiast fetch_ticker z ccxt, używamy pybit
        data = self.client.get_kline(
            symbol="BTCUSDT", interval="5", limit=100  # Ostatnie 100 świec
        )

        print(data)  # Dodano do debugowania

        # Dodano logowanie, aby sprawdzić, co zawiera 'data'
        if "result" not in data or "list" not in data["result"]:
            raise ValueError("Otrzymano nieprawidłowe dane: {}".format(data))

        ohlcv = pd.DataFrame(
            data["result"]["list"],
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quoteVolume",
            ],
        )

        # Sprawdzenie, czy kolumna 'timestamp' istnieje
        if "timestamp" not in ohlcv.columns:
            raise ValueError("Brak kolumny 'timestamp' w danych")

        ohlcv["timestamp"] = pd.to_datetime(
            ohlcv["timestamp"], unit="ms"
        )  # Zmiana na 'ms' dla poprawnego przetworzenia
        ohlcv.set_index("timestamp", inplace=True)

        # Obliczanie średnich kroczących
        ohlcv["SMA_20"] = ohlcv["close"].astype(float).rolling(window=20).mean()
        ohlcv["SMA_50"] = ohlcv["close"].astype(float).rolling(window=50).mean()

        return ohlcv

    def run_strategy(self):
        """
        Uruchamianie strategii i generowanie sygnałów 'BUY', 'SELL' lub 'HOLD'.
        """
        data = self.fetch_and_process_data()

        # Generowanie sygnałów (buy/sell)
        if data["SMA_20"].iloc[-1] > data["SMA_50"].iloc[-1]:
            return "BUY"
        elif data["SMA_20"].iloc[-1] < data["SMA_50"].iloc[-1]:
            return "SELL"
        else:
            return "HOLD"
