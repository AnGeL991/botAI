# Klucze API i konfiguracje
API_KEY = "6uKg9hTwat6spGnRqA"
API_SECRET = "wYFflxJ6cMaEVOo5dertWFbbiNKNFpsqQeZB"
API_PASSWORD = "YOUR_API_PASSWORD"  # Dodatkowe pole, np. dla Bybit

# Ustawienia giełdy
EXCHANGE = "bybit"  # Możesz zmienić na inny, np. "binance"
TRADE_SYMBOL = "BTC/USDT"
TRADE_QUANTITY = 0.01  # Ilość do handlu (np. BTC, ETH)
ORDER_TYPE = "market"  # Typ zlecenia (market, limit)
SLIPPAGE = 0.05  # Akceptowany slippage w procentach

# Stop-loss i Take-profit
STOP_LOSS_PERCENT = 0.02  # 2% stop-loss
TAKE_PROFIT_PERCENT = 0.05  # 5% take-profit

# Zarządzanie ryzykiem
MAX_RISK_PERCENT = 0.1  # Maksymalny procent kapitału na jeden trade
RISK_REWARD_RATIO = 2.0  # Stosunek ryzyka do zysku (np. 2:1)

# Konfiguracja transakcji
LEVERAGE = 10  # Dźwignia (np. 10x)
POSITION_SIZE = 0.01  # Wielkość pozycji (np. w BTC)
USE_STOP_LOSS = True  # Używać stop loss?
USE_TAKE_PROFIT = True  # Używać take profit?

# Ustawienia danych historycznych
DATA_FETCH_LIMIT = 500  # Liczba danych OHLCV do pobrania
TIMEFRAME = "1m"  # Czasowe ramy dla danych (np. "1m", "5m", "1h")

# Sygnały AI i modelowanie
USE_AI = True  # Czy używać sztucznej inteligencji do podejmowania decyzji?
AI_MODEL = (
    "linear_regression"  # Typ modelu AI, np. "linear_regression", "neural_network"
)
AI_TRAINING_LIMIT = 200  # Liczba punktów danych do treningu modelu AI

# Backtesting
USE_BACKTESTING = True  # Czy uruchomić backtest?
BACKTEST_START_DATE = "2023-01-01"  # Data początkowa do backtestingu
BACKTEST_END_DATE = "2023-12-31"  # Data końcowa do backtestingu

# Ustawienia logów
LOG_FILE = "trading_bot.log"  # Plik logów
LOG_LEVEL = (
    "INFO"  # Poziom logowania (np. "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
)

# Inne ustawienia
NOTIFY_ON_ORDER = True  # Powiadomienie po złożeniu zlecenia
NOTIFY_ON_ERROR = True  # Powiadomienie o błędach
ALERT_EMAIL = "your_email@example.com"  # Email do powiadomień
