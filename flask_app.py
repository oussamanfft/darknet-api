from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Chemins relatifs
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
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON invalide'}), 400
            
        ligne_csv = data.get('csv_line', '')
        if not ligne_csv:
            return jsonify({'error': 'csv_line manquant'}), 400
        
        valeurs = ligne_csv.split(',')
        features = []
        for v in valeurs:
            try:
                features.append(float(v))
            except:
                continue
        
        features = features[:80]
        
        if len(features) != 80:
            return jsonify({'error': f'80 features requises, {len(features)} reçues'}), 400
        
        features_norm = scaler.transform([features])
        prediction = model.predict(features_norm)[0]
        classe = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({'success': True, 'classe': classe})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
