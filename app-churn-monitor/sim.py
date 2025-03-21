import requests
import json
import time
import random

# URL de l'API Flask
url = "http://localhost:8000/predict"
headers = {"Content-Type": "application/json"}

# Fonction pour générer des données aléatoires de prédiction
def generate_random_data():
    return {
        "Age": random.randint(18, 70),  # Âge entre 18 et 70 ans
        "Total_Purchase": round(random.uniform(1000, 15000), 2),  # Achat total entre 1000 et 15000
        "Account_Manager": random.randint(0, 1),  # 0 ou 1
        "Years": round(random.uniform(0, 10), 2),  # Entre 0 et 10 années
        "Num_Sites": random.randint(1, 20)  # Nombre de sites entre 1 et 10
    }

# Fonction pour envoyer une requête à l'API Flask
def send_prediction(data):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            print(f"Réponse de l'API : {response.json()}")
        else:
            print(f"Erreur : {response.status_code}")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

# Simuler l'envoi de prédictions avec des données différentes toutes les 10 secondes
while True:
    data = generate_random_data()  # Générer des données aléatoires
    print(f"Envoyer des données : {data}")
    send_prediction(data)
    time.sleep(10)  # Intervalle de 10 secondes entre chaque requête
