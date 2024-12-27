import logging
import threading
import time
from pybit.unified_trading import WebSocket


class BybitWebSocketService:
    def __init__(self, symbol, on_message_callback, reconnect_interval=5):
        """
        Inicjalizacja serwisu WebSocket.

        :param symbol: Symbol tradingowy, np. "ONDOUSDT".
        :param on_message_callback: Funkcja wywoływana przy odbiorze wiadomości.
        :param reconnect_interval: Czas (w sekundach) między próbami ponownego połączenia.
        """
        self.symbol = symbol
        self.on_message_callback = on_message_callback
        self.reconnect_interval = reconnect_interval
        self.ws = None
        self.running = False
        self.thread = None

    def start(self):
        """Rozpoczyna działanie serwisu WebSocket w tle."""
        logging.info("Starting WebSocket service...")
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logging.info(f"WebSocket service started for symbol: {self.symbol}")

    def _run(self):
        """Główna pętla serwisu WebSocket z automatycznym wznawianiem połączenia."""
        while self.running:
            try:
                self.ws = WebSocket(channel_type="linear", testnet=False)
                # Sprawdzenie, czy subskrypcja jest poprawna
                self.ws.kline_stream("1", self.symbol, self.on_message_callback)
                logging.info(f"WebSocket connected for symbol: {self.symbol}")

                # Pozostawienie połączenia aktywnego
                while self.running:
                    time.sleep(1)  # Utrzymanie pętli
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                logging.info(
                    f"Attempting to reconnect in {self.reconnect_interval} seconds..."
                )
                time.sleep(self.reconnect_interval)
            finally:
                self._close_websocket()

    def _close_websocket(self):
        """Zamyka bieżące połączenie WebSocket."""
        if self.ws:
            try:
                self.ws.close()
                logging.info("WebSocket connection closed.")
            except Exception as e:
                logging.error(f"Error while closing WebSocket: {e}")
            finally:
                self.ws = None

    def stop(self):
        """Zatrzymuje działanie serwisu WebSocket."""
        logging.info("Stopping WebSocket service...")
        self.running = False
        if self.thread:
            self.thread.join()
        self._close_websocket()
        logging.info("WebSocket service stopped.")


# Przykładowa funkcja callback do obsługi wiadomości
def on_message(msg):
    logging.info(f"Received message: {msg}")


# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
