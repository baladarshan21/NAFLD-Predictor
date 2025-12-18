#!/usr/bin/env python3
"""
Enhanced Liver Disease Clinical Decision Support System
Alcoholic vs Non-Alcoholic Fatty Liver Disease Classification with Recommendations

Complete End-to-End Implementation with Medical Advisory System
Author: AI Assistant
Date: October 2025
"""

import os
import glob
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class LiverDiseaseClinicalDecisionSupport:
    """
    Comprehensive Liver Disease Clinical Decision Support System
    Combines ML prediction with medical recommendations and alcohol-related assessment
    """

    def __init__(self):
        self.model = None
        self.alcohol_model = None
        self.imputer = None
        self.feature_names = [
            'Age',
            'Gender (Female=1, Male=2)',
            'Body Mass Index',
            'Waist Circumference',
            'Systolic Blood Pressure',
            'Diastolic Blood Pressure',
            'Diyabetes Mellitus (No=0, Yes=1)',
            'Hypertension (No=0, Yes=1)',
            'Hyperlipidemia (No=0, Yes=1)',
            'Metabolic syndrome (No=0, Yes=1)',
            'AST',
            'ALT',
            'ALP',
            'GGT',
            'Total Cholesterol',
            'Triglycerides',
            'HDL',
            'LDL',
            'Glucose'
        ]

        # Enhanced features including alcohol-related markers
        self.extended_features = self.feature_names + [
            'Smoking Status (Never Smoked=1, Left Smoking=2, Smoking=3)'
        ]

        # Normal reference ranges
        self.normal_ranges = {
            'AST': (10, 40),
            'ALT': (7, 56),
            'ALP': (44, 147),
            'GGT': (9, 48),
            'Total Cholesterol': (125, 200),
            'Triglycerides': (50, 150),
            'HDL': {'Male': (40, 60), 'Female': (50, 70)},
            'LDL': (50, 100),
            'Glucose': (70, 100),
            'Body Mass Index': (18.5, 24.9),
            'Waist Circumference': {'Male': (80, 94), 'Female': (70, 80)},
            'Systolic Blood Pressure': (90, 120),
            'Diastolic Blood Pressure': (60, 80)
        }

        # Medical recommendations for different conditions
        self.recommendations = {
            'alcoholic_high_risk': {
                'lifestyle': [
                    "🚨 IMMEDIATE AND COMPLETE alcohol cessation is CRITICAL",
                    "🏥 Urgent addiction counseling and support group participation",
                    "💊 Consider medically supervised alcohol withdrawal if needed",
                    "⚖️ Target weight reduction of 7-10% through supervised caloric restriction",
                    "🏃 Implement structured aerobic exercise program (150+ minutes/week)",
                    "😴 Prioritize stress management and adequate sleep (7-9 hours nightly)",
                    "💧 Ensure adequate hydration and nutritional supplementation"
                ],
                'dietary': [
                    "🥗 Adopt Mediterranean diet with emphasis on liver-protective foods",
                    "🍬 Eliminate refined carbohydrates and added sugars completely",
                    "🌾 Increase soluble fiber intake to 25-35g daily",
                    "🐟 Emphasize omega-3 rich foods (fatty fish, walnuts, flax)",
                    "🥤 NO alcohol consumption - zero tolerance",
                    "💊 Consider milk thistle, vitamin B complex, and folic acid supplements",
                    "🧂 Reduce sodium intake to prevent fluid retention"
                ],
                'medical': [
                    "⚠️ URGENT: Hepatologist consultation within 1-2 weeks",
                    "🔬 Immediate liver biopsy or advanced imaging (MRI/FibroScan)",
                    "💉 Monitor for withdrawal symptoms and provide medical support",
                    "🩺 Screen for hepatitis B/C and other liver conditions",
                    "📅 Intensive monitoring every 2-4 weeks initially",
                    "❤️ Comprehensive cardiovascular assessment",
                    "🧠 Neurological evaluation for hepatic encephalopathy risk"
                ]
            },
            'alcoholic_medium_risk': {
                'lifestyle': [
                    "🚫 COMPLETE alcohol cessation is essential",
                    "👥 Join alcohol support groups (AA, SMART Recovery)",
                    "📉 Target moderate weight loss of 5-7% if overweight",
                    "🚶 Establish regular physical activity routine",
                    "💤 Maintain consistent sleep schedule",
                    "🚭 Smoking cessation if applicable"
                ],
                'dietary': [
                    "🍽️ Implement balanced, liver-friendly nutrition",
                    "🥬 Increase daily vegetable and fruit intake",
                    "🐟 Choose lean proteins and plant-based sources",
                    "💊 B-complex vitamins and thiamine supplementation",
                    "🚫 Absolutely no alcohol consumption"
                ],
                'medical': [
                    "👩⚕️ Hepatologist consultation within 4-6 weeks",
                    "📊 Liver function monitoring every 6-8 weeks",
                    "🧭 Addiction medicine specialist referral",
                    "🔄 Monitor for complications and disease progression"
                ]
            },
            'nonalcoholic_high_risk': {
                'lifestyle': [
                    "🚨 IMMEDIATE dietary modifications to reduce hepatic fat",
                    "⚖️ Target weight reduction of 7-10% through caloric restriction",
                    "🏃 Implement structured aerobic exercise (150+ minutes/week)",
                    "🍷 Limit alcohol to <1 drink/day (women), <2 drinks/day (men)",
                    "😴 Prioritize stress management and adequate sleep",
                    "📊 Consider supervised intermittent fasting protocols"
                ],
                'dietary': [
                    "🥗 Mediterranean diet with whole, unprocessed foods",
                    "🍬 Eliminate refined carbs, added sugars, high-fructose corn syrup",
                    "🌾 Increase soluble fiber intake to 25-35g daily",
                    "🐟 Replace saturated fats with omega-3 rich foods",
                    "🥤 Eliminate sugar-sweetened beverages",
                    "🍎 Focus on low-glycemic index foods"
                ],
                'medical': [
                    "⚠️ Hepatologist consultation within 2-4 weeks",
                    "🔬 Consider liver biopsy or advanced imaging for staging",
                    "❤️ Comprehensive cardiovascular disease screening",
                    "💊 Optimize diabetes management if present",
                    "📅 Intensive monitoring every 3 months"
                ]
            },
            'nonalcoholic_medium_risk': {
                'lifestyle': [
                    "📉 Target moderate weight loss of 5-7%",
                    "🚶 Regular physical activity (120+ minutes/week)",
                    "🍷 Limit alcohol consumption appropriately",
                    "💤 Maintain sleep schedule and stress management"
                ],
                'dietary': [
                    "🍽️ Portion control and balanced nutrition",
                    "🥫 Reduce processed foods and trans fats",
                    "🥬 Increase vegetable and fruit intake",
                    "🐟 Choose lean proteins"
                ],
                'medical': [
                    "📊 Annual liver function monitoring",
                    "💓 Cardiovascular risk management",
                    "👩⚕️ Dietitian consultation for meal planning"
                ]
            },
            'low_risk': {
                'lifestyle': [
                    "✅ Maintain healthy weight within normal BMI",
                    "💪 Continue regular exercise routine",
                    "🧘 Practice stress management",
                    "⚖️ Monitor weight trends"
                ],
                'dietary': [
                    "🥗 Continue healthy eating patterns",
                    "⚖️ Maintain balanced nutrition",
                    "💧 Stay adequately hydrated",
                    "🍎 Emphasize whole foods"
                ],
                'medical': [
                    "📅 Routine annual health check-ups",
                    "✅ Continue preventive care measures",
                    "👀 Monitor for new risk factors"
                ]
            }
        }

    def assess_alcohol_risk(self, patient_data):
        """
        Assess alcohol-related risk factors based on available data
        """
        alcohol_risk_score = 0
        alcohol_indicators = []

        # Check GGT levels (elevated in alcoholic liver disease)
        ggt = patient_data.get('GGT', 0)
        if ggt > 50:
            alcohol_risk_score += 2
            alcohol_indicators.append(f"Elevated GGT: {ggt} (normal: 9-48)")
        elif ggt > 35:
            alcohol_risk_score += 1
            alcohol_indicators.append(f"Moderately elevated GGT: {ggt}")

        # Check AST/ALT ratio (typically >2 in alcoholic liver disease)
        ast = patient_data.get('AST', 0)
        alt = patient_data.get('ALT', 0)
        if ast > 0 and alt > 0:
            ast_alt_ratio = ast / alt
            if ast_alt_ratio > 2:
                alcohol_risk_score += 2
                alcohol_indicators.append(f"AST/ALT ratio: {ast_alt_ratio:.2f} (suggests alcoholic etiology)")
            elif ast_alt_ratio > 1.5:
                alcohol_risk_score += 1
                alcohol_indicators.append(f"AST/ALT ratio: {ast_alt_ratio:.2f} (mildly suggestive)")

        # Check mean corpuscular volume (MCV) - often elevated in chronic alcohol use
        # This would require additional data not in current dataset

        # Smoking status as additional risk factor
        smoking_status = patient_data.get('Smoking Status (Never Smoked=1, Left Smoking=2, Smoking=3)', 1)
        if smoking_status == 3:  # Current smoker
            alcohol_risk_score += 1
            alcohol_indicators.append("Current smoking status increases alcohol-related risk")

        return alcohol_risk_score, alcohol_indicators

    def classify_liver_disease_type(self, patient_data):
        """
        Classify whether liver disease is more likely alcoholic or non-alcoholic
        """
        alcohol_risk_score, alcohol_indicators = self.assess_alcohol_risk(patient_data)

        # Classification based on risk score
        if alcohol_risk_score >= 4:
            return "alcoholic", "high", alcohol_indicators
        elif alcohol_risk_score >= 2:
            return "alcoholic", "medium", alcohol_indicators
        else:
            return "nonalcoholic", "medium", alcohol_indicators

    def load_and_train_model(self, csv_file_path='Liver_Disease_Data.csv'):
        """
        Load dataset, train Random Forest model, and prepare for predictions
        """
        print("=" * 80)
        print("🔬 ENHANCED LIVER DISEASE CLINICAL DECISION SUPPORT SYSTEM")
        print("=" * 80)

        # 1. Load dataset (with fallbacks)
        print("\n🗂️ LOADING CLINICAL DATASET")
        chosen_csv = None

        # Prefer the user-provided path first
        if csv_file_path and os.path.exists(csv_file_path):
            chosen_csv = csv_file_path

        # Common alternative used in this workspace
        alt_candidates = [
            'Liver Patient Dataset (LPD)_train.csv',
            'NAFLD.csv',
            'Liver_Disease_Data.csv',
        ]

        for c in alt_candidates:
            if chosen_csv:
                break
            if os.path.exists(c):
                chosen_csv = c

        # If no candidate found, scan for any CSV in the cwd
        if chosen_csv is None:
            csvs = glob.glob('*.csv')
            if csvs:
                # Prefer files with 'liver' or 'LPD' in name
                liver_like = [f for f in csvs if 'liver' in f.lower() or 'lpd' in f.lower()]
                chosen_csv = liver_like[0] if liver_like else csvs[0]

        # Try to load CSV if available
        df = None
        if chosen_csv:
            # Try several encodings to handle files saved in different formats
            encodings = ['utf-8-sig', 'utf-8', 'latin-1']
            last_exc = None
            for enc in encodings:
                try:
                    df = pd.read_csv(chosen_csv, encoding=enc)
                    print(f"✅ Dataset loaded from: {chosen_csv} ({len(df)} patients, {df.shape[1]} variables) [encoding={enc}]")
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
            if df is None and last_exc is not None:
                print(f"❌ Failed to load '{chosen_csv}': {last_exc}")

        # If still no dataframe, try to load pre-trained model and imputer
        if df is None:
            print("⚠️ No CSV dataset found or failed to load. Attempting to load saved model instead...")
            model_loaded = False
            # Try common pickle names saved in this workspace
            model_paths = [
                ('liver_disease_model.pkl', 'liver_disease_imputer.pkl'),
                ('nafld_clinical_model.pkl', 'nafld_imputer.pkl')
            ]
            for mpath, ipath in model_paths:
                if os.path.exists(mpath) and os.path.exists(ipath):
                    try:
                        self.model = joblib.load(mpath)
                        self.imputer = joblib.load(ipath)
                        print(f"✅ Loaded model from '{mpath}' and imputer from '{ipath}'")

                        # Try to recover trained feature count and importances
                        if hasattr(self.model, 'n_features_in_'):
                            nfi = int(self.model.n_features_in_)
                            # Best-effort: take the first nfi features from our feature list
                            self.trained_features = [f for f in self.feature_names][:nfi]
                        else:
                            # Fallback to using all known feature names
                            self.trained_features = [f for f in self.feature_names]

                        if hasattr(self.model, 'feature_importances_'):
                            fi = list(self.model.feature_importances_)
                            self.feature_importance = dict(zip(self.trained_features, fi[:len(self.trained_features)]))
                        else:
                            self.feature_importance = {f: 0.0 for f in self.trained_features}

                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"❌ Failed to load model pickles: {e}")

            if model_loaded:
                print("🔁 Model loaded — skipping training step.")
                return True

            print(f"❌ Error: No usable dataset or saved model found. Looked for: {csv_file_path} and common alternatives.")
            return False

        # 2. Prepare features and target
        print("\n🎯 PREPARING CLINICAL FEATURES")

        # Auto-detect target column from common names for this workspace/dataset
        possible_targets = [
            'Diagnosis according to SAF (NASH=1, NAFL=2)',
            'Result',
            'result',
            'Diagnosis',
            'diagnosis',
            'Outcome'
        ]
        target_column = None
        for t in possible_targets:
            if t in df.columns:
                target_column = t
                break

        if target_column is None:
            # Try to find a column with only 1/2 values or small integer labels
            for c in df.columns:
                if df[c].dropna().isin([1, 2]).all() and df[c].nunique() <= 3:
                    target_column = c
                    break

        if target_column is None:
            print("❌ Could not find a target/label column in the dataset. Please provide a dataset with a binary/ordinal target (e.g. 'Result').")
            return False

        print(f"🧭 Using target column: '{target_column}'")

        # Convert to binary: treat label '1' as positive (NASH/severe) similar to original convention
        try:
            y = (pd.to_numeric(df[target_column], errors='coerce') == 1).astype(int)
        except Exception:
            y = (df[target_column] == 1).astype(int)

        # Encode gender if present in dataset
        gender_col_candidates = ['Gender (Female=1, Male=2)', 'Gender of the patient', 'Gender']
        gender_col = None
        for g in gender_col_candidates:
            if g in df.columns:
                gender_col = g
                break

        if gender_col is not None:
            # Create standardized gender column expected by rest of the code
            def encode_gender(v):
                if pd.isna(v):
                    return np.nan
                s = str(v).strip().lower()
                if s.startswith('f'):
                    return 1
                if s.startswith('m'):
                    return 2
                try:
                    return int(v)
                except Exception:
                    return np.nan

            df['Gender (Female=1, Male=2)'] = df[gender_col].apply(encode_gender)

        # Select numeric features available in this dataset (exclude target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_column]

        # Also include encoded gender column if present (may be numeric but ensure included)
        if 'Gender (Female=1, Male=2)' in df.columns and 'Gender (Female=1, Male=2)' not in numeric_cols:
            numeric_cols.append('Gender (Female=1, Male=2)')

        if not numeric_cols:
            print("❌ No numeric features found to train on. Dataset appears incompatible.")
            return False

        available_features = numeric_cols
        X = df[available_features].copy()

        print(f"📊 Patient distribution:")
        print(f" • NASH (severe): {sum(y)} patients ({sum(y)/len(y)*100:.1f}%)")
        print(f" • NAFL (mild): {len(y)-sum(y)} patients ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        print(f" • Available features: {len(available_features)}")

        # 3. Handle missing values
        print("\n🔧 PREPROCESSING CLINICAL DATA")
        missing_before = X.isnull().sum().sum()
        print(f"📝 Missing values: {missing_before}")

        self.imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        print("✅ Missing values imputed")

        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, stratify=y, random_state=42
        )

        print(f"📚 Training: {len(X_train)} patients")
        print(f"🧪 Validation: {len(X_test)} patients")

        # 5. Train Random Forest model
        print("\n🤖 TRAINING MACHINE LEARNING MODEL")
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)
        print("✅ Model training completed")

        # 6. Evaluate model performance
        print("\n📈 MODEL PERFORMANCE EVALUATION")
        y_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Validation Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_imputed, y, cv=5, scoring='accuracy')
        print(f"🔄 5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std() * 2:.3f}")

        print("\n📊 CLASSIFICATION METRICS:")
        print(classification_report(y_test, y_pred, target_names=['NAFL', 'NASH']))

        # 7. Feature importance
        self.feature_importance = dict(zip(available_features, self.model.feature_importances_))
        # Save the exact features used during training so we can use the same order at prediction time
        self.trained_features = available_features
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)

        print("\n🔝 TOP PREDICTIVE FEATURES:")
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<35} {importance:.3f}")

        # 8. Save model
        print("\n💾 SAVING MODEL")
        try:
            joblib.dump(self.model, 'liver_disease_model.pkl')
            joblib.dump(self.imputer, 'liver_disease_imputer.pkl')
            print("✅ Model saved successfully")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

        return True

    def comprehensive_patient_assessment(self, patient_data):
        """
        Comprehensive patient assessment including alcohol vs non-alcoholic classification
        """
        if self.model is None or self.imputer is None:
            return {"error": "Model not trained. Run load_and_train_model() first."}

        try:
            # 1. Prepare data for ML prediction
            input_df = pd.DataFrame([patient_data])

            # Use the exact features the model was trained on. If not available, fill with NaN.
            trained_features = getattr(self, 'trained_features', list(self.feature_importance.keys()))

            for feature in trained_features:
                if feature not in input_df.columns:
                    input_df[feature] = np.nan

            # Reorder columns to match training
            input_df = input_df[trained_features]

            # Impute missing values using the same imputer fitted during training
            input_imputed = pd.DataFrame(
                self.imputer.transform(input_df),
                columns=trained_features
            )

            # 2. Make NASH/NAFL prediction
            prediction = self.model.predict(input_imputed)[0]
            probabilities = self.model.predict_proba(input_imputed)[0]

            # 3. Classify alcohol vs non-alcoholic
            disease_type, alcohol_risk_level, alcohol_indicators = self.classify_liver_disease_type(patient_data)

            # 4. Analyze clinical risk factors
            risk_factors = self._analyze_risk_factors(patient_data)

            # 5. Determine overall risk level
            nash_prob = probabilities[1] if len(probabilities) > 1 else 0
            overall_risk = self._determine_overall_risk(nash_prob, risk_factors, disease_type, alcohol_risk_level)

            # 6. Generate recommendations
            rec_key = f"{disease_type}_{overall_risk}"
            if rec_key not in self.recommendations:
                rec_key = "low_risk"

            recommendations = self._generate_personalized_recommendations(
                rec_key, risk_factors, patient_data, alcohol_indicators
            )

            return {
                'nash_nafl_prediction': 'NASH (Severe)' if prediction == 1 else 'NAFL (Mild)',
                'disease_type': 'Alcoholic Liver Disease' if disease_type == 'alcoholic' else 'Non-Alcoholic Fatty Liver Disease',
                'nash_probability': float(nash_prob),
                'nafl_probability': float(probabilities[0]) if len(probabilities) > 0 else 1-nash_prob,
                'confidence': float(np.max(probabilities)) if len(probabilities) > 1 else max(nash_prob, 1-nash_prob),
                'alcohol_risk_level': alcohol_risk_level,
                'overall_risk_level': overall_risk,
                'alcohol_indicators': alcohol_indicators,
                'clinical_risk_factors': risk_factors,
                'recommendations': recommendations,
                'feature_contributions': self._get_feature_contributions(patient_data)
            }

        except Exception as e:
            return {'error': f"Assessment failed: {str(e)}"}

    def _analyze_risk_factors(self, patient_data):
        """Analyze patient's clinical risk factors"""
        risk_factors = []
        gender = 'Female' if patient_data.get('Gender (Female=1, Male=2)', 1) == 1 else 'Male'

        for param, value in patient_data.items():
            if param in self.normal_ranges and value is not None:
                normal_range = self.normal_ranges[param]

                if isinstance(normal_range, dict):
                    normal_range = normal_range.get(gender, list(normal_range.values())[0])

                min_val, max_val = normal_range

                if value > max_val:
                    severity = 'severe' if value > max_val * 1.5 else 'moderate' if value > max_val * 1.2 else 'mild'
                    risk_factors.append({
                        'parameter': param,
                        'value': value,
                        'status': 'Elevated',
                        'normal_range': f"{min_val}-{max_val}",
                        'severity': severity
                    })
                elif value < min_val:
                    risk_factors.append({
                        'parameter': param,
                        'value': value,
                        'status': 'Low',
                        'normal_range': f"{min_val}-{max_val}",
                        'severity': 'moderate'
                    })

        return risk_factors

    def _determine_overall_risk(self, nash_prob, risk_factors, disease_type, alcohol_risk_level):
        """Determine overall risk level"""
        severe_count = sum(1 for rf in risk_factors if rf['severity'] == 'severe')

        if disease_type == 'alcoholic':
            if alcohol_risk_level == 'high' or nash_prob > 0.7:
                return 'high'
            elif alcohol_risk_level == 'medium' or nash_prob > 0.4:
                return 'medium'
        else:
            if nash_prob > 0.7 or severe_count >= 3:
                return 'high'
            elif nash_prob > 0.4 or severe_count >= 1:
                return 'medium'

        return 'low'

    def _generate_personalized_recommendations(self, rec_key, risk_factors, patient_data, alcohol_indicators):
        """Generate personalized recommendations"""
        base_recs = self.recommendations.get(rec_key, self.recommendations['low_risk'])

        specific_recs = []

        # Add specific recommendations based on risk factors
        for risk in risk_factors:
            param = risk['parameter']
            if param in ['AST', 'ALT']:
                specific_recs.append(f"🔬 Monitor {param} closely - currently {risk['status'].lower()}")
            elif param == 'GGT' and risk['status'] == 'Elevated':
                specific_recs.append("🍷 GGT elevation suggests alcohol involvement - consider complete cessation")
            elif param == 'Body Mass Index' and risk['status'] == 'Elevated':
                specific_recs.append("⚖️ Urgent weight management program needed")
            elif param == 'Glucose' and risk['status'] == 'Elevated':
                specific_recs.append("🍬 Immediate diabetes evaluation required")

        # Add alcohol-specific recommendations
        for indicator in alcohol_indicators:
            if "AST/ALT ratio" in indicator:
                specific_recs.append("🔬 Liver enzyme pattern suggests alcohol-related damage")
            elif "GGT" in indicator:
                specific_recs.append("🧪 Elevated GGT warrants alcohol use assessment")

        return {
            'lifestyle': base_recs['lifestyle'],
            'dietary': base_recs['dietary'],
            'medical': base_recs['medical'],
            'specific': list(set(specific_recs))
        }

    def _get_feature_contributions(self, patient_data):
        """Get feature contributions"""
        contributions = []
        for feature in self.feature_names:
            if feature in patient_data and feature in self.feature_importance:
                contributions.append({
                    'feature': feature,
                    'value': patient_data[feature],
                    'importance': self.feature_importance.get(feature, 0)
                })

        return sorted(contributions, key=lambda x: x['importance'], reverse=True)

    def generate_comprehensive_report(self, patient_data, output_file='liver_disease_report.txt'):
        """Generate comprehensive patient report"""
        assessment = self.comprehensive_patient_assessment(patient_data)

        if 'error' in assessment:
            return assessment

        gender = 'Female' if patient_data.get('Gender (Female=1, Male=2)', 1) == 1 else 'Male'

        report = f"""
{'='*90}
COMPREHENSIVE LIVER DISEASE ASSESSMENT REPORT
{'='*90}

PATIENT INFORMATION:
• Age: {patient_data.get('Age', 'N/A')} years
• Gender: {gender}
• BMI: {patient_data.get('Body Mass Index', 'N/A')} kg/m²
• Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

CLINICAL PREDICTIONS:
• Primary Classification: {assessment['disease_type']}
• Liver Condition: {assessment['nash_nafl_prediction']}
• NASH Probability: {assessment['nash_probability']:.1%}
• Overall Risk Level: {assessment['overall_risk_level'].upper()}
• Alcohol Risk Level: {assessment['alcohol_risk_level'].upper()}
• Confidence: {assessment['confidence']:.1%}

ALCOHOL-RELATED INDICATORS:
"""

        if assessment['alcohol_indicators']:
            for indicator in assessment['alcohol_indicators']:
                report += f"• {indicator}\n"
        else:
            report += "• No significant alcohol-related indicators detected\n"

        report += "\nCLINICAL RISK FACTORS:\n"
        if assessment['clinical_risk_factors']:
            for rf in assessment['clinical_risk_factors']:
                report += f"• {rf['parameter']}: {rf['value']} ({rf['status']}, Severity: {rf['severity']})\n"
        else:
            report += "• No significant clinical risk factors identified\n"

        report += f"""

COMPREHENSIVE TREATMENT RECOMMENDATIONS:

LIFESTYLE MODIFICATIONS:
"""
        for rec in assessment['recommendations']['lifestyle']:
            report += f"• {rec}\n"

        report += "\nDIETARY INTERVENTIONS:\n"
        for rec in assessment['recommendations']['dietary']:
            report += f"• {rec}\n"

        report += "\nMEDICAL MANAGEMENT:\n"
        for rec in assessment['recommendations']['medical']:
            report += f"• {rec}\n"

        if assessment['recommendations']['specific']:
            report += "\nSPECIFIC INTERVENTIONS:\n"
            for rec in assessment['recommendations']['specific']:
                report += f"• {rec}\n"

        report += f"""

TOP CONTRIBUTING FACTORS:
"""
        for contrib in assessment['feature_contributions'][:7]:
            report += f"• {contrib['feature']}: {contrib['value']} (Importance: {contrib['importance']:.3f})\n"

        report += f"""

{'='*90}
CLINICAL DISCLAIMER:
This assessment combines machine learning analysis with clinical decision support.
Results should be interpreted by qualified healthcare professionals.
Always consult with hepatologists and addiction specialists for comprehensive care.
{'='*90}
"""

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Comprehensive report saved to: {output_file}")
        return assessment

    def interactive_assessment(self):
        """Interactive patient assessment"""
        # If running in a non-interactive environment, skip interactive prompts
        if not sys.stdin.isatty():
            print("⚠️ stdin is not interactive — skipping interactive assessment.")
            return

        while True:
            print("\n" + "="*80)
            print("🩺 INTERACTIVE LIVER DISEASE ASSESSMENT")
            print("Enter patient clinical data:")
            print("="*80)

            patient_data = {}

            # Essential features for assessment
            essential_features = [
                'Age',
                'Gender (Female=1, Male=2)', 
                'Body Mass Index',
                'AST',
                'ALT',
                'GGT',
                'Glucose'
            ]

            for feature in essential_features:
                while True:
                    try:
                        prompt = f"➡️ {feature}: "
                        if feature == 'Gender (Female=1, Male=2)':
                            prompt = "➡️ Gender (1=Female, 2=Male): "

                        value_str = input(prompt)
                        if value_str == "":
                            patient_data[feature] = np.nan
                            break
                        patient_data[feature] = float(value_str)
                        break
                    except ValueError:
                        print("❌ Invalid input. Enter number or press Enter to skip.")

            # Optional features
            optional_features = [f for f in self.feature_names if f not in essential_features]

            print("\n🔧 Optional parameters (press Enter to skip):")
            for feature in optional_features[:5]:  # Limit to avoid too many questions
                try:
                    value_str = input(f"➡️ {feature}: ")
                    if value_str != "":
                        patient_data[feature] = float(value_str)
                except ValueError:
                    pass

            print("\n🔬 Analyzing patient data...")
            assessment = self.comprehensive_patient_assessment(patient_data)

            if 'error' in assessment:
                print(f"❌ Error: {assessment['error']}")
            else:
                print("\n" + "-"*50)
                print("📊 ASSESSMENT RESULTS")
                print("-"*50)
                print(f"🏥 Disease Type: {assessment['disease_type']}")
                print(f"🎯 Condition: {assessment['nash_nafl_prediction']}")
                print(f"⚠️ Overall Risk: {assessment['overall_risk_level'].upper()}")
                print(f"🍷 Alcohol Risk: {assessment['alcohol_risk_level'].upper()}")
                print(f"📊 NASH Probability: {assessment['nash_probability']:.1%}")

                if assessment['alcohol_indicators']:
                    print("\n🚨 Alcohol-Related Indicators:")
                    for indicator in assessment['alcohol_indicators'][:3]:
                        print(f"  • {indicator}")

                if assessment['clinical_risk_factors']:
                    print("\n⚠️ Clinical Risk Factors:")
                    for rf in assessment['clinical_risk_factors'][:3]:
                        print(f"  • {rf['parameter']}: {rf['value']} ({rf['severity']})")

                # Offer to generate full report
                generate_report = input("\n📄 Generate detailed report? (y/n): ").lower()
                if generate_report == 'y':
                    self.generate_comprehensive_report(patient_data)

            another = input("\n🔄 Assess another patient? (y/n): ").lower()
            if another != 'y':
                break

def main():
    """Main function demonstrating the system"""
    liver_system = LiverDiseaseClinicalDecisionSupport()

    # Allow disabling interactive mode with a CLI flag
    no_interactive = '--no-interactive' in sys.argv
    # Single-patient prediction mode: --predict <json_file>
    predict_idx = None
    if '--predict' in sys.argv:
        try:
            predict_idx = sys.argv.index('--predict')
            predict_file = sys.argv[predict_idx + 1]
        except Exception:
            predict_file = None
    else:
        predict_file = None

    # Train or load model. Prefer the workspace dataset if available.
    # If a saved model exists and training is expensive, the loader will load pickles.
    if not liver_system.load_and_train_model('Liver Patient Dataset (LPD)_train.csv'):
        return

    # If prediction mode requested, load patient JSON and run assessment
    if predict_file:
        import json
        if not os.path.exists(predict_file):
            print(f"❌ Prediction file not found: {predict_file}")
            return
        try:
            with open(predict_file, 'r', encoding='utf-8') as pf:
                patient = json.load(pf)
        except Exception as e:
            print(f"❌ Failed to read prediction file: {e}")
            return

        print("\n📥 Running single-patient prediction from JSON file")
        assessment = liver_system.generate_comprehensive_report(patient, 'prediction_report.txt')
        if isinstance(assessment, dict) and 'error' in assessment:
            print(f"❌ Prediction failed: {assessment.get('error')}")
        else:
            print("📊 Prediction completed. Report saved to prediction_report.txt")
        return

    print("\n" + "="*80)
    print("🩺 DEMONSTRATION: LIVER DISEASE ASSESSMENT")
    print("="*80)

    # Example 1: High-risk patient with alcohol indicators
    alcoholic_patient = {
        'Age': 48,
        'Gender (Female=1, Male=2)': 2,
        'Body Mass Index': 31.5,
        'Waist Circumference': 108,
        'Systolic Blood Pressure': 145,
        'Diastolic Blood Pressure': 92,
        'AST': 78,  # Elevated
        'ALT': 45,  # AST/ALT ratio > 1.7 suggests alcohol
        'GGT': 85,  # Highly elevated (suggests alcohol)
        'Total Cholesterol': 195,
        'Triglycerides': 245,
        'HDL': 35,
        'Glucose': 108,
        'Smoking Status (Never Smoked=1, Left Smoking=2, Smoking=3)': 3
    }

    print("\n📋 CASE 1: SUSPECTED ALCOHOLIC LIVER DISEASE")
    assessment1 = liver_system.generate_comprehensive_report(
        alcoholic_patient, 
        'alcoholic_liver_disease_report.txt'
    )
    if isinstance(assessment1, dict) and 'error' in assessment1:
        print(f"❌ Assessment failed: {assessment1.get('error')}")
    else:
        print(f"🎯 Result: {assessment1.get('disease_type')}")
        print(f"⚠️ Risk: {assessment1.get('overall_risk_level','N/A').upper()}")

    # Example 2: NAFLD patient
    nafld_patient = {
        'Age': 42,
        'Gender (Female=1, Male=2)': 1,
        'Body Mass Index': 33.2,
        'Waist Circumference': 98,
        'Systolic Blood Pressure': 128,
        'Diastolic Blood Pressure': 82,
        'AST': 35,
        'ALT': 52,  # ALT > AST suggests NAFLD
        'GGT': 28,  # Normal
        'Total Cholesterol': 215,
        'Triglycerides': 185,
        'HDL': 42,
        'Glucose': 95,
        'Smoking Status (Never Smoked=1, Left Smoking=2, Smoking=3)': 1
    }

    print("\n📋 CASE 2: SUSPECTED NON-ALCOHOLIC FATTY LIVER DISEASE")
    assessment2 = liver_system.generate_comprehensive_report(
        nafld_patient,
        'nafld_patient_report.txt'
    )
    if isinstance(assessment2, dict) and 'error' in assessment2:
        print(f"❌ Assessment failed: {assessment2.get('error')}")
    else:
        print(f"🎯 Result: {assessment2.get('disease_type')}")
        print(f"⚠️ Risk: {assessment2.get('overall_risk_level','N/A').upper()}")

    # Interactive mode (optional)
    if not no_interactive:
        print("\n" + "="*80)
        print("🔄 STARTING INTERACTIVE MODE")
        print("="*80)
        liver_system.interactive_assessment()
    else:
        print("⚠️ Interactive mode disabled via --no-interactive flag; exiting demo.")

    print("\n" + "="*80)
    print("🎉 LIVER DISEASE ASSESSMENT SYSTEM SESSION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
