from pybit.unified_trading import HTTP
import logging

logging.basicConfig(level=logging.INFO)


class TradingService:
    def __init__(self, client: HTTP):
        self.client = client

    def place_order(self, action):
        """
        Składa zlecenie na podstawie decyzji strategii.
        """
        request = []

        if action == "BUY":
            request.append(
                {
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Market",
                    "isLeverage": 0,
                    "qty": "0.01",  # Ilość kontraktów jako string
                    "timeInForce": "GTC",
                    "orderLinkId": "spot-btc-01",  # Unikalny identyfikator
                }
            )
        elif action == "SELL":
            request.append(
                {
                    "symbol": "BTCUSDT",
                    "side": "Sell",
                    "orderType": "Market",
                    "isLeverage": 0,
                    "qty": "0.01",  # Ilość kontraktów jako string
                    "timeInForce": "GTC",
                    "orderLinkId": "spot-btc-02",  # Unikalny identyfikator
                }
            )
        else:
            return None

        # Składanie zlecenia
        order = self.client.place_batch_order(category="spot", request=request)

        return order

    def get_balance(self):
        """
        Pobiera saldo konta.
        """
        return self.client.get_balance()

    def get_open_orders(self):
        """
        Pobiera otwarte zlecenia.
        """
        return self.client.get_open_orders()

    def cancel_order(self, order_id):
        """
        Anuluje zlecenie na podstawie jego identyfikatora.
        """
        return self.client.cancel_order(order_id)

    def get_market_data(self, symbol):
        """
        Pobiera dane rynkowe dla danego symbolu.
        """
        return self.client.get_market_data(symbol)

    def calculate_stop_loss(self):
        """
        Obliczanie ceny stop-loss na podstawie ceny rynkowej i procentu.
        """
        current_price = self.client.get_ticker(symbol="BTCUSDT")["last"]
        stop_loss_price = float(current_price) * (1 - 0.02)  # Załóżmy 2% stop loss
        return stop_loss_price

    def calculate_take_profit(self):
        """
        Obliczanie ceny take-profit na podstawie ceny rynkowej i procentu.
        """
        current_price = self.client.get_ticker(symbol="BTCUSDT")["last"]
        take_profit_price = float(current_price) * (1 + 0.05)  # Załóżmy 5% take profit
        return take_profit_price

    def set_stop_loss(self, price):
        """
        Ustawienie stop-loss.
        """
        print(f"Setting stop loss at {price}")
        try:
            stop_loss_order = self.client.place_batch_order(
                symbol="BTCUSDT",
                side="Sell",
                order_type="StopMarket",
                qty=0.01,
                stop_px=price,
                time_in_force="GoodTillCancel",
            )
            print(f"Stop-loss order placed: {stop_loss_order}")
        except Exception as e:
            print(f"Error placing stop-loss order: {e}")

    def set_take_profit(self, price):
        """
        Ustawienie take-profit.
        """
        print(f"Setting take profit at {price}")
        try:
            take_profit_order = self.client.place_batch_order(
                symbol="BTCUSDT",
                side="Sell",
                order_type="Limit",
                qty=0.01,
                price=price,
                time_in_force="GoodTillCancel",
            )
            print(f"Take-profit order placed: {take_profit_order}")
        except Exception as e:
            print(f"Error placing take-profit order: {e}")
