class Backtester:
    def __init__(self, strategy, initial_balance=10000):
        """
        Inicjalizacja Backtestera
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []

    def execute_backtest(self):
        """
        Przeprowadzenie backtestu na danych historycznych
        """
        data = self.strategy.fetch_and_process_data()
        signals = self.strategy.run_strategy()

        # Logika wykonywania transakcji
        for i in range(len(data) - 1):
            if data["SMA_20"].iloc[i] > data["SMA_50"].iloc[i] and signals == "BUY":
                self.buy(data["close"].iloc[i])
            elif data["SMA_20"].iloc[i] < data["SMA_50"].iloc[i] and signals == "SELL":
                self.sell(data["close"].iloc[i])

        return self.balance

    def buy(self, price):
        """
        Wykonanie transakcji kupna
        """
        self.positions.append({"type": "buy", "price": price})
        print(f"Buying at {price}")
        self.balance -= price  # Zakłada, że kupujemy za całą kwotę (można dodać bardziej złożoną logikę)

    def sell(self, price):
        """
        Wykonanie transakcji sprzedaży
        """
        if self.positions:
            last_position = self.positions.pop()
            profit = price - last_position["price"]
            self.balance += price  # Dodajemy kwotę sprzedaży do salda
            print(f"Selling at {price}. Profit: {profit}")
        else:
            print("No position to sell!")
