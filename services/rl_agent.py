import os
import logging
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from services.rl_trading_env import TradingEnv
import torch
import re  # Dodaj import na początku pliku
import random  # Dodaj import na początku pliku


logging.basicConfig(filename="trading_bot.log", level=logging.INFO)


class RLAgent:
    def __init__(self, client, window_size=30, learning_rate=0.0003, gamma=0.99):
        """
        Inicjalizacja agenta RL z użyciem Stable-Baselines3.
        """
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.client = client

        # Tworzenie środowiska
        self.env = TradingEnv(client=client, window_size=window_size, interval="15")

        self.eval_env = TradingEnv(
            client=client,
            window_size=self.window_size,
            interval="15",
        )
        # Tworzenie modelu PPO z Stable-Baselines3
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            verbose=1,
        )

        # Inicjalizacja do wizualizacji
        self.reward_history = []
        self.evaluation_rewards = []
        self.transaction_history = []

    def train(self, timesteps=10000, eval_freq=5000):
        """
        Trening modelu przez określoną liczbę kroków.
        """
        # Callback do ewaluacji modelu podczas treningu
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path="./logs/",
            log_path="./logs/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

        # Rozpoczcie treningu
        self.model.learn(total_timesteps=timesteps, callback=eval_callback)

        # Po zakończeniu treningu zapisujemy model i wizualizujemy wyniki
        self.save_model()
        self.plot_results()
        logging.info("Model został wytrenowany i zapisany.")

    def evaluate(self, episodes=1):
        """
        Ewaluacja agenta na podstawie średniego zysku z kilku epizodów.
        """
        total_rewards = []
        symbols = [
            "ONDOUSDT",
        ]  # Lista symboli do wyboru

        # Sprawdzenie, czy liczba epizodów nie przekracza liczby dostępnych symboli
        if episodes > len(symbols):
            raise ValueError(
                "Liczba epizodów nie może przekraczać liczby dostępnych symboli."
            )

        selected_symbols = random.sample(
            symbols, episodes
        )  # Losowy wybór unikalnych symboli

        for episode in range(episodes):
            symbol = selected_symbols[
                episode
            ]  # Wybór symbolu z listy unikalnych symboli
            self.eval_env = TradingEnv(
                client=self.client,
                window_size=self.window_size,
                interval="15",
                symbol=symbol,
            )
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                )
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                print(
                  f"Action: {action}, Reward: {reward}, Info: {info}"
                 )  # Dodaj logowanie akcji i nagrody

            total_rewards.append(episode_reward)
            logging.info(
                f"Episod {episode + 1} z symbolem {symbol}: nagroda {episode_reward}"
            )

        avg_reward = sum(total_rewards) / len(total_rewards)
        self.evaluation_rewards.append(avg_reward)
        logging.info(f"total_rewards: {total_rewards}")
        logging.info(f"Średnia nagroda w ewaluacji: {avg_reward}")
        return avg_reward

    def save_model(self, filepath="ppo_model"):
        """
        Zapis modelu do pliku.
        """
        self.model.save(filepath)
        logging.info(f"Model zapisano do pliku {filepath}")

    def load_model(self, filepath="rl_model"):
        current_directory = os.getcwd()  # Bieżący katalog roboczy
        full_filepath = os.path.join(
            current_directory, filepath
        )  # Pełna ścieżka do pliku

        print(f"Bieżący katalog roboczy: {current_directory}")
        print(f"Pełna ścieżka pliku modelu: {full_filepath}")

        if os.path.exists(full_filepath):
            self.model = PPO.load(full_filepath, env=self.env)
            print(f"Model załadowany z {full_filepath}")
        else:
            print(f"Model {full_filepath} nie istnieje!")

    def plot_results(self):
        """
        Wizualizacja wyników treningu i ewaluacji.
        """
        # Wizualizacja nagród z ewaluacji
        plt.figure(figsize=(12, 6))

        if self.evaluation_rewards:
            plt.subplot(1, 2, 1)
            plt.plot(self.evaluation_rewards, label="Evaluation Rewards")
            plt.title("Evaluation Rewards History")
            plt.xlabel("Evaluation Steps")
            plt.ylabel("Average Reward")
            plt.legend()

        # Wyświetlanie historii transakcji
        self.transaction_history = self.env.get_transaction_history()
        if self.transaction_history:
            plt.subplot(1, 2, 2)
            profits = [
                t[4] if len(t) == 5 else 0 for t in self.transaction_history
            ]  # Wyciąganie zysków/strat z transakcji
            plt.plot(profits, label="Transaction Profits")
            plt.title("Transaction Profits")
            plt.xlabel("Transactions")
            plt.ylabel("Profit/Loss")
            plt.legend()

        plt.tight_layout()
        plt.savefig("training_results.png")
        plt.close()

        # Logowanie transakcji
        logging.info(f"Transaction History ({len(self.transaction_history)}):")
        for transaction in self.transaction_history:
            if transaction[0] == "long":  # Kupno (otwarcie pozycji long)
                logging.info(
                    f"Open long at {transaction[1]} for quantity {transaction[2]}, Balance: {transaction[3]}"
                )
            elif transaction[0] == "short":  # Sprzedaż (otwarcie short)
                logging.info(
                    f" Open short at {transaction[1]} for quantity {transaction[2]}, Balance: {transaction[3]}"
                )
            elif re.match(r"Close Position", transaction[0]):  # Zamknięcie pozycji long
                if len(transaction) == 5:  # Sprawdzamy, czy mamy zysk/stratę
                    logging.info(
                        f"{transaction[0]} at {transaction[1]} for quantity {transaction[2]}, "
                        f"Balance: {transaction[3]}, Profit/Loss: {transaction[4]}"
                    )
                else:
                    logging.warning(f"Invalid transaction format: {transaction}")
            else:
                logging.warning(f"Invalid transaction format: {transaction}")
