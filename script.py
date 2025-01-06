import requests
import time
import json
from services.bybit_service import BybitService
from config.settings import API_KEY, API_SECRET

# Parametry początkowe
BASE_URL = "https://api-testnet.bybit.com/v5/market/kline"
SYMBOL = "ONDOUSDT"
CATEGORY = "linear"
INTERVAL = "5"  # 1-minutowe interwały
LIMIT = 1000  # Maksymalna liczba rekordów na żądanie


# Funkcja do pobierania danych z zakresu czasowego
def fetch_data(start_ts, end_ts):
    bybit_service = BybitService(API_KEY, API_SECRET)
    url = f"{BASE_URL}?category={CATEGORY}&symbol={SYMBOL}&interval={INTERVAL}&start={start_ts}&end={end_ts}&limit={LIMIT}"

    data = bybit_service.client.get_kline(
        symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, start=start_ts, end=end_ts
    )
    if "result" not in data:
        raise ValueError("Brak wyników w odpowiedzi API.")

    return list(reversed(data.get("result", {}).get("list", [])))


# Funkcja do uzyskania timestampów dla zakresu 3 miesięcy
def get_three_months_timestamps():
    end_time = int(time.time() * 1000)  # Aktualny czas w milisekundach
    three_months_ms = 150 * 24 * 60 * 60 * 1000  # 90 dni w milisekundach
    start_time = end_time - three_months_ms
    return start_time, end_time


# Funkcja główna
def fetch_last_three_months_data():
    start_time, end_time = get_three_months_timestamps()
    all_data = {
        "total": 0,
        "data": [],
    }
    current_start = start_time
    print(f"all_data: {all_data}")
    while current_start < end_time:
        # Ustaw koniec paczki na max 1000 minut od startu lub end_time
        current_end = current_start + (LIMIT * 300 * 1000)
        if current_end > end_time:
            current_end = (
                end_time  # Zabezpieczenie przed przekroczeniem końcowego czasu
            )

        print(f"Pobieranie danych od {current_start} do {current_end}")
        batch_data = fetch_data(current_start, current_end)
        print(f"Pobrano {len(batch_data)} rekordów")
        if batch_data:
            all_data["total"] += len(batch_data)
            all_data["data"].extend(batch_data)

            current_start = current_end
        else:
            print("Brak danych w aktualnej partii lub błąd. Przerywanie.")
            break

    # Zapisz dane do pliku JSON
    with open("bybit_data.json", "w") as file:
        json.dump(all_data, file, indent=4)

    print(f"Pobrano {len(all_data)} rekordów i zapisano do pliku 'bybit_data.json'.")


# Uruchom skrypt
if __name__ == "__main__":
    fetch_last_three_months_data()
