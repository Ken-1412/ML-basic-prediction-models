from flask import Flask, request, jsonify
from flask_cors import CORS
from model import HealthDataAnalyzer
import numpy as np

app = Flask(__name__)
CORS(app)

# Initialize the analyzer
analyzer = HealthDataAnalyzer()
try:
    analyzer.load_data()
    
    # Process and train models for both datasets
    X_diabetes, y_diabetes, _ = analyzer.preprocess_diabetes_data()
    analyzer.train_models(X_diabetes, y_diabetes, 'diabetes')
    
    X_heart, y_heart, _ = analyzer.preprocess_heart_data()
    analyzer.train_models(X_heart, y_heart, 'heart')
    
except Exception as e:
    print(f"Error initializing models: {str(e)}")

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        input_data = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]
        
        predictions = analyzer.predict(input_data, 'diabetes')
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        data = request.json
        input_data = [
            float(data['Age']),
            float(data['Sex']),
            float(data['ChestPainType']),
            float(data['RestingBP']),
            float(data['Cholesterol']),
            float(data['FastingBS']),
            float(data['RestingECG']),
            float(data['MaxHR']),
            float(data['ExerciseAngina']),
            float(data['Oldpeak']),
            float(data['ST_Slope'])
        ]
        
        predictions = analyzer.predict(input_data, 'heart')
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/metrics/<dataset>', methods=['GET'])
def metrics(dataset):
    """Return evaluation metrics (AUC + ROC points) for the requested dataset.
    dataset should be 'diabetes' or 'heart'."""
    try:
        if dataset not in analyzer.test_sets:
            return jsonify({'success': False, 'error': 'No test data available for dataset'}), 400

        X_test, y_test = analyzer.test_sets[dataset]
        results = analyzer.evaluate_models(X_test, y_test, dataset)

        # Convert numpy arrays to lists and simplify payload
        simplified = {}
        for model_name, r in results.items():
            simplified[model_name] = {
                'auc_score': float(r['auc_score']),
                'fpr': [float(x) for x in r['fpr'].tolist()],
                'tpr': [float(x) for x in r['tpr'].tolist()]
            }

        return jsonify({'success': True, 'metrics': simplified})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)