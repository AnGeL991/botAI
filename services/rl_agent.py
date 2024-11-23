import os
import json
from stable_baselines3 import PPO
from services.rl_trading_env import TradingEnv


class RLAgent:
    def __init__(
        self, client, model_filepath="rl_model", results_filepath="results.json"
    ):
        self.client = client
        self.env = TradingEnv(client)
        self.model_filepath = model_filepath
        self.results_filepath = results_filepath

        # Jeśli istnieje zapisany model, wczytaj go
        if os.path.exists(self.model_filepath + ".zip"):
            self.load_model(self.model_filepath)
            print(f"Wczytano istniejący model z {self.model_filepath}")
        else:
            # Twórz nowy model, jeśli nie istnieje zapisany
            self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, timesteps=20000):
        # Sprawdź, czy środowisko jest prawidłowo zainicjalizowane
        if self.env is None:
            print("Błąd: Środowisko nie zostało zainicjalizowane.")
            return
        self.model.learn(total_timesteps=timesteps)
        # Zapisz model po treningu
        self.save_model(self.model_filepath)

    def evaluate(self, save_results=False):
        obs = self.env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = self.model.predict(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            steps += 1

        if save_results:
            self.save_evaluation_results(
                {
                    "total_reward": total_reward,
                    "steps": steps,
                    "parameters": {
                        "timesteps": 20000,  # Możesz tu dodać dynamiczne parametry
                    },
                }
            )

        return total_reward

    def save_model(self, filepath="rl_model"):
        self.model.save(filepath)

    def load_model(self, filepath="rl_model"):
        current_directory = os.getcwd()  # Bieżący katalog roboczy
        full_filepath = os.path.join(
            current_directory, filepath + ".zip"
        )  # Pełna ścieżka do pliku

        print(f"Bieżący katalog roboczy: {current_directory}")
        print(f"Pełna ścieżka pliku modelu: {full_filepath}")

        if os.path.exists(full_filepath):
            self.model = PPO.load(full_filepath)
            print(f"Model załadowany z {full_filepath}")
        else:
            print(f"Model {full_filepath} nie istnieje!")

    def save_evaluation_results(self, results):
        try:
            # Jeśli plik istnieje, odczytaj istniejące wyniki
            try:
                with open(self.results_filepath, "r") as f:
                    existing_results = json.load(f)
            except FileNotFoundError:
                existing_results = []

            # Dodaj nowe wyniki do istniejących
            existing_results.append(results)

            # Zapisz całość do pliku JSON
            with open(self.results_filepath, "w") as f:
                json.dump(existing_results, f, indent=4)
            print(f"Wyniki ewaluacji zapisano w {self.results_filepath}")
        except Exception as e:
            print(f"Błąd zapisu wyników: {e}")

    def load_evaluation_results(self):
        try:
            with open(self.results_filepath, "r") as f:
                results = json.load(f)
            print(f"Wczytano wyniki: {results}")
            return results
        except FileNotFoundError:
            print(f"Plik {self.results_filepath} nie istnieje.")
            return []
        except Exception as e:
            print(f"Błąd odczytu wyników: {e}")
            return []
