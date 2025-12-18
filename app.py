#!/usr/bin/env python3
"""
Flask Web Application: Enhanced Liver Disease Diagnostic System
Alcoholic vs Non-Alcoholic Fatty Liver Differentiation with Food and Lifestyle Recommendations
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import traceback

# Use the existing high-level clinical system which handles features/imputation
from Enhanced_Liver_Disease_System import LiverDiseaseClinicalDecisionSupport
import numpy as np

app = Flask(__name__)

# Instantiate the system once
liver_system = LiverDiseaseClinicalDecisionSupport()

def prepare_model():
    # Train or load the model so it's ready for /predict requests
    try:
        liver_system.load_and_train_model('NAFLD.csv')
    except Exception as e:
        print('[app] Model preparation failed:', e)
        traceback.print_exc()

@app.route('/')
def home():
    # Serve index.html from the project root so the file can remain in workspace root
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        patient_data = {
            'Age': float(data.get('age', np.nan)),
            'Gender (Female=1, Male=2)': int(data.get('gender', 1)),
            'AST': float(data.get('ast', np.nan)),
            'ALT': float(data.get('alt', np.nan)),
            'GGT': float(data.get('ggt', np.nan)),
            'Glucose': float(data.get('glucose', np.nan)),
            'BMI': float(data.get('bmi', np.nan))
        }

        assessment = liver_system.comprehensive_patient_assessment(patient_data)
        if isinstance(assessment, dict) and 'error' in assessment:
            return jsonify({'error': assessment.get('error')}), 500

        # Compute AST/ALT ratio safely
        ast = patient_data.get('AST') or 0
        alt = patient_data.get('ALT') or 0
        ast_alt_ratio = None
        try:
            ast_alt_ratio = round(float(ast) / float(alt), 2) if float(alt) != 0 else None
        except Exception:
            ast_alt_ratio = None

        # Define function for custom recommendations based on disease type
        def get_recommendations(disease_type):
            if 'Alcoholic' in disease_type:
                return {
                    'diet': [
                        'Avoid alcohol completely',
                        'High protein, moderate carbohydrate diet',
                        'Increase B-vitamin rich foods',
                        'Stay hydrated with at least 2L water daily'
                    ],
                    'lifestyle': [
                        'Complete alcohol abstinence',
                        'Regular liver function monitoring',
                        'Join support groups for alcohol cessation',
                        'Stress management techniques'
                    ]
                }
            else:  # Non-alcoholic recommendations
                return {
                    'diet': [
                        'Mediterranean diet rich in olive oil',
                        'Avoid processed foods and added sugars',
                        'Increase fiber intake with vegetables and whole grains',
                        'Limit saturated fats and red meat'
                    ],
                    'lifestyle': [
                        'Regular exercise (150+ minutes/week)',
                        'Weight management program',
                        'Blood sugar monitoring',
                        'Adequate sleep (7-8 hours)'
                    ]
                }
        
        # Get custom recommendations based on disease type
        custom_recommendations = get_recommendations(assessment.get('disease_type', ''))
        
        # Get risk factors data
        risk_factors = get_risk_factors()
        
        # Get medical recommendations
        medical_recommendations = get_medical_recommendations(assessment.get('disease_type', ''))
        
        resp = {
            'disease_type': assessment.get('disease_type'),
            'condition': assessment.get('nash_nafl_prediction'),
            'ast_alt_ratio': ast_alt_ratio,
            'GGT': patient_data.get('GGT'),
            'recommendations': {
                'diet': custom_recommendations['diet'],
                'lifestyle': custom_recommendations['lifestyle']
            },
            'overall_risk_level': assessment.get('overall_risk_level'),
            'risk_factors': risk_factors,
            'medical_recommendations': medical_recommendations
        }

        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_risk_factors():
    # This would ideally come from a model or database
    # For now, returning static data to match the frontend requirements
    return {
        'factors': [
            {'name': 'Total Protein', 'score': 7.8, 'severity': 'high'},
            {'name': 'Glucose', 'score': 6.9, 'severity': 'high'},
            {'name': 'Total Cholesterol', 'score': 6.2, 'severity': 'medium'},
            {'name': 'Direct Bilirubin', 'score': 5.8, 'severity': 'medium'},
            {'name': 'Body Mass Index', 'score': 5.5, 'severity': 'medium'},
            {'name': 'ALT', 'score': 5.4, 'severity': 'medium'},
            {'name': 'Total Bilirubin', 'score': 5.0, 'severity': 'low'},
            {'name': 'LDL', 'score': 4.8, 'severity': 'low'}
        ],
        'title': 'Key Risk Factors for NAFLD',
        'description': 'These factors contribute to your overall liver health assessment.'
    }

def get_medical_recommendations(disease_type):
    if 'Alcoholic' in disease_type:
        return {
            'immediate': [
                '⚠️ Immediate hepatologist consultation required',
                '🔍 Consider liver biopsy for definitive diagnosis',
                '🧪 Evaluate for clinical trial enrollment',
                '🩺 Monitor for portal hypertension signs'
            ],
            'lifestyle': [
                '🚫 Complete alcohol abstinence',
                '🥦 Anti-inflammatory diet',
                '💊 Vitamin B supplementation',
                '🧘 Stress management program'
            ],
            'dietary': [
                '🍲 High-protein, low-fat diet',
                '🥗 Increase antioxidant intake',
                '💧 3L water daily minimum',
                '🚫 Avoid processed foods'
            ],
            'exercise': [
                '🚶 Start with 10-minute daily walks',
                '🧘 Gentle yoga or stretching',
                '⏱️ Gradually increase activity duration',
                '💪 Light resistance training when approved'
            ],
            'monitoring': [
                '📊 Weekly liver function tests',
                '👨‍⚕️ Bi-weekly hepatologist visits',
                '🔄 Monthly ultrasound assessment',
                '🧠 Cognitive function monitoring'
            ]
        }
    else:  # Non-alcoholic recommendations
        return {
            'immediate': [
                '⚠️ Immediate hepatologist consultation required',
                '🔍 Consider liver biopsy for definitive diagnosis',
                '🧪 Evaluate for clinical trial enrollment',
                '🩺 Monitor for portal hypertension signs'
            ],
            'lifestyle': [
                '⚖️ Weight management program',
                '🛌 Sleep hygiene improvement',
                '🧘 Stress reduction techniques',
                '📱 Digital health monitoring'
            ],
            'dietary': [
                '🥗 Mediterranean diet plan',
                '🚫 Eliminate processed sugars',
                '🌾 Complex carbohydrates only',
                '🥑 Increase healthy fats'
            ],
            'exercise': [
                '🏋️ Resistance training 3x weekly',
                '🏃 30-minute cardio 5x weekly',
                '🧘 Flexibility exercises',
                '⏱️ Active breaks every hour'
            ],
            'monitoring': [
                '📊 Monthly liver enzyme tests',
                '📏 Waist circumference tracking',
                '🩸 Quarterly metabolic panel',
                '🔄 Semi-annual imaging'
            ]
        }

if __name__ == '__main__':
    # Prepare model at startup (avoid relying on before_first_request decorator)
    prepare_model()
    app.run(debug=True)
