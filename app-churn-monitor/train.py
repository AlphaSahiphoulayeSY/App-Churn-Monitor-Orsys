import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Charger le fichier CSV
df = pd.read_csv('customer_churn.csv')

# Sélectionner les cinq features numériques et la cible
features = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']
X = df[features]
y = df['Churn']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner un modèle RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarder le modèle dans un fichier .pkl
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé avec succès.")
