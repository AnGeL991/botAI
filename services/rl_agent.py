import os
import logging
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from services.rl_trading_env import TradingEnv

import re
from services.bybit_trading_env import BybitTradingEnv

logging.basicConfig(filename="trading_bot.log", level=logging.INFO)


class RLAgent:
    def __init__(
        self,
        client,
        bybit_service,
        window_size=60,
        learning_rate=0.001,
        gamma=0.95,
        max_steps=2,
    ):
        """
        Inicjalizacja agenta RL z użyciem Stable-Baselines3.
        """
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.client = client
        self.bybit_service = bybit_service
        self.max_steps = max_steps
        # Tworzenie środowiska
        self.env = TradingEnv(
            bybit_service=bybit_service,
            window_size=window_size,
            interval="5",
            take_data_from_file=False,
        )

        self.eval_env = TradingEnv(
            bybit_service=bybit_service,
            window_size=self.window_size,
            interval="5",
            take_data_from_file=False,
        )
        # Tworzenie modelu PPO z Stable-Baselines3
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
        )

        # Inicjalizacja do wizualizacji
        self.reward_history = []
        self.evaluation_rewards = []
        self.transaction_history = []

        self.waiting_for_new_data = False

    def train_on_file_data(self, timesteps=10000, eval_freq=10000, step=1):
        """
        Trening modelu przez określoną liczbę kroków.
        """
        self.env.set_new_data(step)
        self.eval_env.set_new_data(step)
        self.train(timesteps=timesteps, eval_freq=eval_freq, step=step)
        logging.info("Model został wytrenowany i zapisany.")

    def train(self, timesteps=10000, eval_freq=5000, step=1):
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
        self.plot_results(step)
        logging.info("Model został wytrenowany i zapisany.")

    def real_time_evaluate(self, episodes=1):
        """
        Ewaluacja agenta na podstawie średniego zysku z kilku epizodów.
        """
        try:
            total_rewards = []

            step = 0

            bybit_env = TradingEnv(
                bybit_service=self.bybit_service,
                window_size=self.window_size,
                interval="5",
                take_data_from_file=False,
            )
            # bybit_env.set_new_data(43)
            # bybit_env = BybitTradingEnv(
            #    api_key="6uKg9hTwat6spGnRqA",
            #  api_secret="wYFflxJ6cMaEVOo5dertWFbbiNKNFpsqQeZB",
            #   interval="1",
            #  symbol="ONDOUSDT",
            #  leverage=50,
            #  risk_per_trade=0.02,
            # take_profit_ratio=0.1,
            # stop_loss_ratio=0.02,
            # )

            for episode in range(episodes):

                main_obs = bybit_env.reset()
                done = False
                episode_reward = 0
                while not done:
                    # Sprawdzenie, czy nowe dane są dostępne
                    if bybit_env.new_data_received:
                        action, _ = self.model.predict(main_obs, deterministic=True)
                        obs, reward, done, info = bybit_env.step(action)
                        main_obs = obs
                        episode_reward += reward
                        step += 1
                        print(
                            f"Action: {action}, Reward: {reward}, Info: {info}"
                        )  # Dodaj logowanie akcji i nagrody

                total_rewards.append(episode_reward)
                logging.info(f"Episod {episode + 1}: nagroda {episode_reward}")

            avg_reward = sum(total_rewards) / len(total_rewards)
            # self.evaluation_rewards.append(avg_reward)
            logging.info(f"total_rewards: {total_rewards}")
            logging.info(f"Średnia nagroda w ewaluacji: {avg_reward}")
            return avg_reward
        except KeyboardInterrupt:
            bybit_env.stop()
            logging.info("Shutting down...")

    def evaluate(self, episodes=1):
        """
        Ewaluacja agenta na podstawie średniego zysku z kilku epizodów.
        """
        total_rewards = []

        for episode in range(episodes):
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
            logging.info(f"Episod {episode + 1}: nagroda {episode_reward}")

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

    def plot_results(self, step):
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
        plt.savefig(f"results_png/training_results_{step}.png")
        plt.close()

        # Logowanie transakcji
        logging.info(
            f"Transaction History step {step} ({len(self.transaction_history)}):"
        )
        for transaction in self.transaction_history:
            if transaction[0] == "Buy":  # Kupno (otwarcie pozycji long)
                logging.info(
                    f"Open long at {transaction[1]} for quantity {transaction[2]}, Balance: {transaction[3]}"
                )
            elif transaction[0] == "Sell":  # Sprzedaż (otwarcie short)
                logging.info(
                    f" Open short at {transaction[1]} for quantity {transaction[2]}, Balance: {transaction[3]}"
                )
            elif (
                re.match(r"Close Position", transaction[0])
                or re.match(r"Take profit", transaction[0])
                or re.match(r"Stop loss", transaction[0])
            ):  # Zamknięcie pozycji long
                if len(transaction) == 5:  # Sprawdzamy, czy mamy zysk/stratę
                    logging.info(
                        f"{transaction[0]} at {transaction[1]} for quantity {transaction[2]}, "
                        f"Balance: {transaction[3]}, Profit/Loss: {transaction[4]}"
                    )
                else:
                    logging.warning(f"Invalid transaction format: {transaction}")
            else:
                logging.warning(f"Invalid transaction format: {transaction}")
