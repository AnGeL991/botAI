from pybit.unified_trading import HTTP


class DataService:
    def __init__(self, client: HTTP):
        """
        Inicjalizuje serwis z przekazanym klientem Bybit.
        """
        self.client = client

    def test_connection(self):
        try:
            # Sprawdzamy aktualną cenę rynkową
            self.client.get_tickers(category="linear", symbol="BTCUSDT")
            print("Connection test successful")
        except Exception as e:
            print("Connection test failed:", e)

    def fetch_historical_data(self, symbol="BTCUSDT", interval="5", limit=500):
        """
        Pobiera dane historyczne z testnetu Bybit w postaci świec OHLCV
        """
        try:
            # Pobranie danych (historia świec)
            ohlcv = self.client.query_kline(
                symbol=symbol, interval=interval, limit=limit
            )
            ohlcv_data = ohlcv["result"]

            # Konwersja do DataFrame
            df = pd.DataFrame(ohlcv_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)

            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
