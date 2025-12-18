flask_frontend = '''#!/usr/bin/env python3
"""
Flask Web Application: Enhanced Liver Disease Diagnostic System
Alcoholic vs Non-Alcoholic Fatty Liver Differentiation with Food and Lifestyle Recommendations
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load pre-trained model and imputer
model = joblib.load('liver_disease_model.pkl')
imputer = joblib.load('liver_disease_imputer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract form data
    patient_data = {
        'Age': float(data.get('age', 0)),
        'Gender (Female=1, Male=2)': float(data.get('gender', 1)),
        'Body Mass Index': float(data.get('bmi', 0)),
        'AST': float(data.get('ast', 0)),
        'ALT': float(data.get('alt', 0)),
        'GGT': float(data.get('ggt', 0)),
        'Glucose': float(data.get('glucose', 0))
    }

    ast, alt, ggt = patient_data['AST'], patient_data['ALT'], patient_data['GGT']
    ast_alt_ratio = ast / alt if alt != 0 else 0

    if ast_alt_ratio >= 2 or ggt > 50:
        disease_type = 'Alcoholic Fatty Liver Disease'
    else:
        disease_type = 'Non-Alcoholic Fatty Liver Disease'

    input_array = np.array([[patient_data[f] for f in patient_data]])
    input_df = pd.DataFrame(input_array, columns=patient_data.keys())
    input_df = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    prediction = model.predict(input_df)[0]
    condition = 'NASH (Severe Fatty Liver)' if prediction == 1 else 'NAFL (Mild Fatty Liver)'

    recommendations = get_recommendations(disease_type)

    response = {
        'disease_type': disease_type,
        'condition': condition,
        'ast_alt_ratio': round(ast_alt_ratio, 2),
        'GGT': ggt,
        'recommendations': recommendations
    }
    return jsonify(response)

def get_recommendations(disease_type):
    if 'Alcoholic' in disease_type:
        return {
            'diet': [
                '🍵 Complete alcohol cessation immediately.',
                '🥗 Consume high-protein, low-fat meals.',
                '🥦 Include antioxidant-rich foods like broccoli and spinach.',
                '💧 Stay hydrated with 2.5–3L water daily.',
                '🍊 Add citrus fruits and vitamin-C-rich foods.',
                '🥛 Avoid sugary and carbonated drinks.'
            ],
            'lifestyle': [
                '🚭 Stop smoking if applicable.',
                '🏃 Begin light exercise routines to restore metabolism.',
                '🧘 Practice mindfulness and sleep 7–9 hours daily.',
                '🏥 Schedule hepatologist check-ups within 2 weeks.',
                '💊 Take prescribed B-complex and liver detox supplements.'
            ]
        }
    else:
        return {
            'diet': [
                '🥗 Follow the Mediterranean diet (olive oil, fish, legumes).',
                '🚫 Reduce refined sugar and carbohydrate intake.',
                '🥬 Eat high-fiber vegetables daily.',
                '🐟 Include omega-3-rich fish twice per week.',
                '🍞 Choose whole grains over processed cereals.',
                '🍎 Consume 2–3 servings of fruits daily.'
            ],
            'lifestyle': [
                '🚶 Engage in 150+ minutes of moderate exercise weekly.',
                '⚖️ Aim for 7–10% weight reduction if overweight.',
                '💤 Maintain at least 7 hours of sound sleep.',
                '📅 Regular monthly monitoring of liver enzymes.',
                '🧘 Practice yoga or meditation for stress reduction.'
            ]
        }

if __name__ == '__main__':
    app.run(debug=True)
'''

html_index = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Liver Disease Diagnosis</title>
<style>
body{font-family:Arial;background:#f6fbff;margin:0;padding:0;color:#222;}
.container{width:80%;margin:40px auto;background:white;padding:30px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.2);}
h1{text-align:center;color:#0056b3;}
form{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:20px;}
label{font-weight:600;}
input{padding:8px;border-radius:5px;border:1px solid #ccc;}
button{grid-column:span 2;padding:12px;background:#007bff;color:white;font-size:16px;border:none;border-radius:5px;cursor:pointer;}
button:hover{background:#0056b3;}
.result{margin-top:30px;padding:20px;border-radius:10px;}
.healthy{background:#d4edda;color:#155724;}
.warning{background:#fff3cd;color:#856404;}
.danger{background:#f8d7da;color:#721c24;}
</style>
</head>
<body>
<div class="container">
<h1>Enhanced Liver Disease Assessment</h1>
<form id="patient-form">
<label>Age:</label><input type="number" name="age" required>
<label>Gender (1=Female, 2=Male):</label><input type="number" name="gender" required>
<label>Body Mass Index:</label><input type="number" step="0.1" name="bmi" required>
<label>AST:</label><input type="number" name="ast" required>
<label>ALT:</label><input type="number" name="alt" required>
<label>GGT:</label><input type="number" name="ggt" required>
<label>Glucose:</label><input type="number" name="glucose" required>
<button type="submit">Analyze</button>
</form>
<div id="output" class="result"></div>
</div>
<script>
document.getElementById('patient-form').addEventListener('submit', async (e) => {
e.preventDefault();
const formData = new FormData(e.target);
const response = await fetch('/predict', {method: 'POST', body: formData});
const data = await response.json();
let colorClass = data.disease_type.includes('Alcoholic') ? 'danger' : 'warning';
const resultDiv = document.getElementById('output');
resultDiv.className = 'result ' + colorClass;
resultDiv.innerHTML = `<h2>${data.disease_type}</h2>
<p><b>Condition:</b> ${data.condition}</p>
<p><b>AST/ALT Ratio:</b> ${data.ast_alt_ratio}, <b>GGT:</b> ${data.GGT}</p>
<h3>Dietary Recommendations:</h3><ul>${data.recommendations.diet.map(i => `<li>${i}</li>`).join('')}</ul>
<h3>Lifestyle Recommendations:</h3><ul>${data.recommendations.lifestyle.map(i => `<li>${i}</li>`).join('')}</ul>`;
});
</script>
</body>
</html>
'''

# Write Flask app and front-end template
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(flask_frontend)

import os
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html_index)

print("✅ Flask front-end app created successfully.")
print("Run using: python app.py")