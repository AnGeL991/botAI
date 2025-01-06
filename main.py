from pybit.unified_trading import HTTP
from config.settings import API_KEY, API_SECRET
from services.data_service import DataService
from services.trading_service import TradingService

from services.rl_agent import RLAgent
import logging
import json
from services.bybit_service import BybitService
from services.websocket_service import BybitWebSocketService
import time
import threading

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
    # Inicjowanie serwisu Bybit
    bybit_service = BybitService(API_KEY, API_SECRET)

    # Dodanie agenta RL
    rl_agent = RLAgent(client, bybit_service, window_size=60)

    logging.info("Fetching market data for the agent...")
    env_data = rl_agent.env.data  # Pobranie danych z środowiska agenta

    if env_data is not None and not env_data.empty:
        print("Dane rynkowe używane przez agenta:")
        print(env_data.head())  # Wyświetlenie pierwszych wierszy danych

    else:
        print("Brak danych rynkowych! Upewnij się, że API zwraca poprawne dane.")

    # Wczytanie wyników

    model_filepath = "./logs/best_model.zip"

    # Sprawdzenie, czy istnieje zapisany model
    try:
        rl_agent.load_model(model_filepath)
    except FileNotFoundError:
        logging.warning("Model file not found. Starting with a new model.")

    # Trening agenta
    #logging.info("Starting RL agent training...")
    #results = []
    #total_steps = 43  # Ustal maksymalną liczbę kroków, np. 10
    #for step in range(1, total_steps + 1):
    #    logging.info(f"Training step: {step}")
    #    rl_agent.load_model(model_filepath)
    #    rl_agent.train_on_file_data(timesteps=200000, eval_freq=25000, step=step)
    #    total_reward = rl_agent.evaluate()
    #    logging.info(f"RL agent evaluation total reward: {total_reward}")
    #    results.append(
    #        {
    #            "step": step,
    #            "total_reward": total_reward,
    #        }
    #    )
    # save_results(results)
    # logging.info("RL agent training complete.")

    # rl_agent.eval_env.set_new_data(43)
    total_reward = rl_agent.real_time_evaluate()

    logging.info(f"RL agent evaluation total reward: {total_reward}")

    # total_reward = rl_agent.evaluate()
    # logging.info(f"RL agent evaluation total reward: {total_reward}")


#  results.append(
#   {
#       "step": step,
#       "total_reward": total_reward,
#   }
#  )


# if __name__ == "__main__":
#   step = 0
#  results_filepath = "results.json"
#  results = []
#
# for _ in range(50):
#   main(results, step)
#     step += 1


if __name__ == "__main__":
    main()
