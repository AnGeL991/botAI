from pybit.unified_trading import HTTP
from config.settings import API_KEY, API_SECRET
from services.data_service import DataService
from services.trading_service import TradingService
from services.ai_service import AIStrategy
from services.backtest_service import Backtester
from services.rl_agent import RLAgent
import logging
import json

logging.basicConfig(filename="trading_bot.log", level=logging.INFO)


# Inicjalizacja klienta Bybit z pybit
client = HTTP(
    api_key="6uKg9hTwat6spGnRqA",
    api_secret="wYFflxJ6cMaEVOo5dertWFbbiNKNFpsqQeZB",
    demo=True,
)

# help(client)  # Wyświetli wszystkie dostępne metody i atrybuty obiektu client


def load_results(filepath="results.json"):
    """Funkcja do wczytania wyników agenta z pliku JSON."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Results file {filepath} not found. Starting fresh.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file {filepath}. Starting fresh.")
        return {}


def save_results(results, filepath="results.json"):
    """Funkcja do zapisu wyników agenta do pliku JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)


def main():
    # Inicjalizacja serwisów, klient jest przekazywany jako zależność
    data_service = DataService(client)
    trading_service = TradingService(client)

    # Test połączenia z API Bybit
    data_service.test_connection()

    # Dodanie agenta RL
    rl_agent = RLAgent(client)

    # Pobranie danych i wyświetlenie ich raz
    logging.info("Fetching market data for the agent...")
    env_data = rl_agent.env.data  # Pobranie danych z środowiska agenta

    if env_data is not None and not env_data.empty:
        print("Dane rynkowe używane przez agenta:")
        print(env_data.head())  # Wyświetlenie pierwszych wierszy danych
    else:
        print("Brak danych rynkowych! Upewnij się, że API zwraca poprawne dane.")


    # Wczytanie wyników
    results_filepath = "results.json"
    model_filepath = "ppo_model.zip"
    results = load_results(results_filepath)

    # Sprawdzenie, czy istnieje zapisany model
    try:
        rl_agent.load_model(model_filepath)
    except FileNotFoundError:
        logging.warning("Model file not found. Starting with a new model.")

    # Trening agenta
    logging.info("Starting RL agent training...")
    rl_agent.train(timesteps=20000)
    logging.info("RL agent training complete.")

    # Ewaluacja agenta
    total_reward = rl_agent.evaluate()
    logging.info(f"RL agent evaluation total reward: {total_reward}")

    # Aktualizacja wyników
    # results["last_training"] = {
    #     "timesteps": 20000,
    #     "total_reward": total_reward,
    # }
    save_results(results, results_filepath)


if __name__ == "__main__":
    main()
