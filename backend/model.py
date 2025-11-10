import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class HealthDataAnalyzer:
    def __init__(self):
        self.diabetes_data = None
        self.heart_data = None
        self.models = {}
        self.scalers = {}
        self.pca_components = {}
        self.test_sets = {}
        
    def load_data(self):
        # Load PIMA Indians Diabetes Dataset
        self.diabetes_data = pd.read_csv('../data/diabetes.csv')
        # Load Heart Disease Dataset
        self.heart_data = pd.read_csv('../data/heart.csv')
        
    def preprocess_diabetes_data(self):
        # Handle missing and zero values
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in zero_columns:
            self.diabetes_data[column] = self.diabetes_data[column].replace(0, np.nan)
            self.diabetes_data[column].fillna(self.diabetes_data[column].mean(), inplace=True)
            
        # Scale features
        X = self.diabetes_data.drop('Outcome', axis=1)
        y = self.diabetes_data['Outcome']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['diabetes'] = scaler
        
        # PCA transformation
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        self.pca_components['diabetes'] = pca
        
        return X_scaled, y, X_pca
    
    def preprocess_heart_data(self):
        # Handle categorical variables
        le = LabelEncoder()
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for col in categorical_cols:
            self.heart_data[col] = le.fit_transform(self.heart_data[col])
        
        # Scale features
        X = self.heart_data.drop('HeartDisease', axis=1)
        y = self.heart_data['HeartDisease']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['heart'] = scaler
        
        # PCA transformation
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        self.pca_components['heart'] = pca
        
        return X_scaled, y, X_pca
    
    def train_models(self, X, y, dataset_name):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # store test set for later evaluation / API access
        self.test_sets[dataset_name] = (X_test, y_test)
        
        # Initialize models
        models = {
            'logistic': LogisticRegression(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'naive_bayes': GaussianNB()
        }
        
        # Train and store models
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[f"{dataset_name}_{name}"] = model
            
        return X_test, y_test
    
    def evaluate_models(self, X_test, y_test, dataset_name):
        results = {}
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            if name.startswith(dataset_name):
                # Get predictions and probabilities
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                conf_matrix = confusion_matrix(y_test, y_pred)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = auc(fpr, tpr)
                
                # Store results
                results[name] = {
                    'confusion_matrix': conf_matrix,
                    'auc_score': auc_score,
                    'fpr': fpr,
                    'tpr': tpr
                }
                
                # Plot ROC curve
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name.capitalize()} Dataset')
        plt.legend()
        plt.savefig(f'../data/{dataset_name}_roc_curve.png')
        plt.close()
        
        return results
    
    def plot_feature_importance(self, dataset_name):
        # Get feature names
        if dataset_name == 'diabetes':
            features = self.diabetes_data.drop('Outcome', axis=1).columns
        else:
            features = self.heart_data.drop('HeartDisease', axis=1).columns
        
        # Get Decision Tree model
        dt_model = self.models[f'{dataset_name}_decision_tree']
        importances = dt_model.feature_importances_
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        plt.title(f'Feature Importance - {dataset_name.capitalize()} Dataset')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'../data/{dataset_name}_feature_importance.png')
        plt.close()
        
    def predict(self, data, dataset_type):
        # Scale the input data
        scaled_data = self.scalers[dataset_type].transform([data])
        
        results = {}
        for name, model in self.models.items():
            if name.startswith(dataset_type):
                prob = model.predict_proba(scaled_data)[0][1]
                results[name.split('_')[1]] = float(prob)
        
        return results

# Usage example:
if __name__ == "__main__":
    analyzer = HealthDataAnalyzer()
    analyzer.load_data()
    
    # Process and analyze diabetes data
    X_diabetes, y_diabetes, X_diabetes_pca = analyzer.preprocess_diabetes_data()
    X_test_diabetes, y_test_diabetes = analyzer.train_models(X_diabetes, y_diabetes, 'diabetes')
    diabetes_results = analyzer.evaluate_models(X_test_diabetes, y_test_diabetes, 'diabetes')
    analyzer.plot_feature_importance('diabetes')
    
    # Process and analyze heart disease data
    X_heart, y_heart, X_heart_pca = analyzer.preprocess_heart_data()
    X_test_heart, y_test_heart = analyzer.train_models(X_heart, y_heart, 'heart')
    heart_results = analyzer.evaluate_models(X_test_heart, y_test_heart, 'heart')
    analyzer.plot_feature_importance('heart')