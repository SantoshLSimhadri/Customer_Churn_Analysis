import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class CustomerChurnAnalyzer:
    """
    A comprehensive class for customer churn analysis and prediction.
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def load_data(self, file_path):
        """
        Load customer data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing customer data
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """
        Perform exploratory data analysis on the customer dataset.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        print("=" * 50)
        print("CUSTOMER CHURN DATA EXPLORATION")
        print("=" * 50)
        
        # Basic information
        print("\nDataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Missing values
        print("\nMissing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Data types
        print("\nData Types:")
        print(self.data.dtypes)
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        # Churn distribution
        if 'churn' in self.data.columns:
            print("\nChurn Distribution:")
            churn_counts = self.data['churn'].value_counts()
            print(churn_counts)
            print(f"Churn Rate: {churn_counts[1] / len(self.data) * 100:.2f}%")
    
    def preprocess_data(self):
        """
        Clean and preprocess the data for machine learning.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        print("Preprocessing data...")
        
        # Make a copy for preprocessing
        df = self.data.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill missing values
        if len(numeric_columns) > 0:
            imputer_num = SimpleImputer(strategy='median')
            df[numeric_columns] = imputer_num.fit_transform(df[numeric_columns])
        
        if len(categorical_columns) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_columns:
            if col != 'churn':  # Don't encode target variable yet
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        
        # Encode target variable
        if 'churn' in df.columns:
            le_target = LabelEncoder()
            df['churn'] = le_target.fit_transform(df['churn'])
        
        self.data_processed = df
        self.label_encoders = label_encoders
        print("Data preprocessing completed.")
        
    def create_features(self):
        """
        Create additional features for better prediction.
        """
        if self.data_processed is None:
            print("Data not preprocessed. Please preprocess data first.")
            return
            
        df = self.data_processed.copy()
        
        # Create feature combinations (example features)
        if 'monthly_charges' in df.columns and 'tenure' in df.columns:
            df['total_charges_estimated'] = df['monthly_charges'] * df['tenure']
            df['charges_per_tenure'] = df['monthly_charges'] / (df['tenure'] + 1)
        
        if 'total_charges' in df.columns and 'monthly_charges' in df.columns:
            df['charges_ratio'] = df['total_charges'] / (df['monthly_charges'] + 1)
        
        # Create tenure categories
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], 
                                      bins=[0, 12, 24, 48, 72, float('inf')], 
                                      labels=['0-1 year', '1-2 years', '2-4 years', 
                                             '4-6 years', '6+ years'])
            df['tenure_group'] = LabelEncoder().fit_transform(df['tenure_group'])
        
        self.data_processed = df
        print("Feature engineering completed.")
    
    def prepare_model_data(self, target_column='churn'):
        """
        Prepare data for machine learning models.
        
        Args:
            target_column (str): Name of the target column
        """
        if self.data_processed is None:
            print("Data not processed. Please preprocess data first.")
            return
            
        # Separate features and target
        if target_column in self.data_processed.columns:
            X = self.data_processed.drop(target_column, axis=1)
            y = self.data_processed[target_column]
        else:
            print(f"Target column '{target_column}' not found.")
            return
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data prepared for modeling:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
    
    def train_models(self):
        """
        Train multiple machine learning models for churn prediction.
        """
        if self.X_train is None:
            print("Model data not prepared. Please prepare data first.")
            return
            
        print("Training models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Logistic Regression, original for tree-based models
            if name == 'Logistic Regression':
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_use, self.y_train, 
                                      cv=5, scoring='roc_auc')
            
            # Test predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            model_scores[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test AUC: {auc_score:.4f}")
        
        self.models = model_scores
        
        # Select best model
        best_model_name = max(model_scores.keys(), 
                            key=lambda x: model_scores[x]['test_auc'])
        self.best_model = model_scores[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best AUC: {self.best_model['test_auc']:.4f}")
    
    def evaluate_model(self):
        """
        Evaluate the best model with detailed metrics and visualizations.
        """
        if self.best_model is None:
            print("No trained models available.")
            return
            
        print("=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        
        y_pred = self.best_model['predictions']
        y_pred_proba = self.best_model['probabilities']
        
        # Classification report
        print(f"\nClassification Report for {self.best_model_name}:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Feature importance (for tree-based models)
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            feature_importance = self.best_model['model'].feature_importances_
            feature_names = self.X_train.columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))
    
    def generate_insights(self):
        
        # Generate business insights from the churn analysis
        print("=" * 50)
        print("BUSINESS INSIGHTS")
        print("=" * 50)
        
        if self.data is None:
            print("No data available for insights.")
            return
            
        # Calculate churn rate by different segments
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        print("\nChurn Rate by Segments:")
        for col in categorical_cols:
            if col != 'churn' and col in self.data.columns:
                churn_by_segment = self.data.groupby(col)['churn'].agg(['count', 'sum', 'mean'])
                churn_by_segment['churn_rate'] = churn_by_segment['mean'] * 100
                print(f"\n{col.upper()}:")
                print(churn_by_segment[['count', 'sum', 'churn_rate']].round(2))
        
        # Risk segments
        if self.best_model is not None:
            risk_threshold_high = 0.7
            risk_threshold_medium = 0.3
            
            high_risk = (self.best_model['probabilities'] >= risk_threshold_high).sum()
            medium_risk = ((self.best_model['probabilities'] >= risk_threshold_medium) & 
                          (self.best_model['probabilities'] < risk_threshold_high)).sum()
            low_risk = (self.best_model['probabilities'] < risk_threshold_medium).sum()
            
            print(f"\nRisk Segmentation (Test Set):")
            print(f"High Risk (>={risk_threshold_high}): {high_risk} customers")
            print(f"Medium Risk ({risk_threshold_medium}-{risk_threshold_high}): {medium_risk} customers")
            print(f"Low Risk (<{risk_threshold_medium}): {low_risk} customers")
    
    def save_results(self, output_path='churn_analysis_results.csv'):
        """
        Save the analysis results to a CSV file.
        
        Args:
            output_path (str): Path where to save the results
        """
        if self.best_model is None:
            print("No model results to save.")
            return
            
        # Create results dataframe
        results_df = pd.DataFrame({
            'actual_churn': self.y_test,
            'predicted_churn': self.best_model['predictions'],
            'churn_probability': self.best_model['probabilities']
        })
        
        # Add risk categories
        results_df['risk_category'] = pd.cut(
            results_df['churn_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

def main():
    
    Main function to run the complete churn analysis pipeline
    # Initialize analyzer
    analyzer = CustomerChurnAnalyzer()
    
    print("Customer Churn Analysis Framework Ready!")
    print("Please load your customer data using analyzer.load_data('your_file.csv')")

if __name__ == "__main__":
    main()
