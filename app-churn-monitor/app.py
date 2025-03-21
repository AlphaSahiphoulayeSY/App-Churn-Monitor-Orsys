from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from prometheus_client import Counter, generate_latest, Summary
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.debug("Démarrage de l'application Flask")

# Charger le modèle de churn
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.debug("Modèle chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    model = None

# Métriques Prometheus détaillées
TOTAL_API_CALLS = Counter('api_calls_total', 'Nombre total des appels API')
PREDICTION_COUNTER = Counter('predictions_total', 'Nombre de prédictions par classe', ['type'])
REQUEST_TIME = Summary('request_processing_seconds', 'Temps de traitement des requêtes')

@app.route('/predict', methods=['POST'])
@REQUEST_TIME.time()
def predict():
    if model is None:
        logger.error("Tentative de prédiction sans modèle chargé")
        return jsonify({"error": "Modèle indisponible"}), 500

    try:
        data = request.get_json()
        logger.debug(f"Données reçues : {data}")

        features = pd.DataFrame([data])
        prediction = model.predict(features)[0]

        # Incrémenter les métriques
        TOTAL_API_CALLS.inc()
        pred_label = 'churn' if prediction == 1 else 'no_churn'
        PREDICTION_COUNTER.labels(type=pred_label).inc()

        result = {"prediction": pred_label}
        logger.debug(f"Prédiction effectuée : {result}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    logger.debug("Requête reçue sur /metrics")
    return generate_latest(), 200

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

if __name__ == '__main__':
    logger.debug("Démarrage du serveur Flask")
    app.run(host='0.0.0.0', port=8000)
