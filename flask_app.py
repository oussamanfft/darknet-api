from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Charger les modèles
model = joblib.load('modele_xgboost.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Le scaler attend 76 features
NB_FEATURES = 76

@app.route('/')
def home():
    return jsonify({'status': 'ok', 'message': 'API Darknet'})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        ligne_csv = data.get('csv_line', '')
        
        if not ligne_csv:
            return jsonify({'error': 'csv_line manquant'}), 400
        
        # Nettoyer la ligne
        ligne_csv = ligne_csv.replace('\n', '').replace('\r', '').strip()
        valeurs = ligne_csv.split(',')
        
        # Extraire les nombres (ignorer texte, dates, IP)
        features = []
        for v in valeurs:
            v = v.strip()
            try:
                features.append(float(v))
            except ValueError:
                continue
        
        # Compléter avec des zéros si moins de 76
        if len(features) < NB_FEATURES:
            print(f"⚠️ {len(features)} features reçues, complétion avec {NB_FEATURES - len(features)} zéros")
            features = features + [0.0] * (NB_FEATURES - len(features))
        
        # Tronquer si trop de features
        if len(features) > NB_FEATURES:
            features = features[:NB_FEATURES]
        
        # Vérification finale
        if len(features) != NB_FEATURES:
            return jsonify({'error': f'Problème: {len(features)} features, attendu {NB_FEATURES}'}), 400
        
        # Normaliser et prédire
        features_norm = scaler.transform([features])
        prediction = model.predict(features_norm)[0]
        classe = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({'success': True, 'classe': classe})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)