from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Charger les modèles
model = joblib.load('modele_xgboost.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return jsonify({'status': 'ok', 'message': 'API Darknet'})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Récupérer les données
        data = request.get_json()
        ligne_csv = data.get('csv_line', '')
        
        if not ligne_csv:
            return jsonify({'error': 'csv_line manquant'}), 400
        
        # Nettoyer la ligne : remplacer les retours à la ligne et espaces
        ligne_csv = ligne_csv.replace('\n', '').replace('\r', '').strip()
        
        # Extraire toutes les valeurs
        valeurs = ligne_csv.split(',')
        
        # Convertir en nombres (ignorer les non-numériques)
        features = []
        for v in valeurs:
            v = v.strip()
            try:
                features.append(float(v))
            except ValueError:
                continue  # Ignorer les valeurs non-numériques (IP, dates, texte)
        
        # === NOUVEAU : Compléter avec des zéros si moins de 80 features ===
        if len(features) < 80:
            print(f"⚠️ {len(features)} features reçues, complétion avec des zéros jusqu'à 80")
            features = features + [0.0] * (80 - len(features))
        
        # Si plus de 80, garder les 80 premières
        if len(features) > 80:
            features = features[:80]
        
        # Vérifier qu'on a bien 80 features
        if len(features) != 80:
            return jsonify({'error': f'Problème: {len(features)} features, impossible de traiter'}), 400
        
        # Normaliser et prédire
        features_norm = scaler.transform([features])
        prediction = model.predict(features_norm)[0]
        classe = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({
            'success': True,
            'classe': classe,
            'code': int(prediction)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)