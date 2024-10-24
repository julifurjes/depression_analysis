import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataPreparation

class RandomForestLongitudinalModel:
    def __init__(self, data):
        self.data = data
        self.subject_col = 'ID'
        self.data_long = DataPreparation().handle_data(self.data)
        if not os.path.exists('long_rf/output'):
            os.makedirs('long_rf/output')  # Ensure output directory exists

    def preprocess_data(self):
        df = self.data_long.copy()

        # Exclude unnecessary columns
        exclude_cols = ['ID', 'DepressionScore', 'Pontuacao_Depressao', 'Score_Depressao']
        X = df.drop(columns=exclude_cols + ['DepressionStatus'])

        y = df['DepressionStatus']

        # Handle missing values (if any) - impute or drop
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Process categorical variables (if not already encoded)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes

        # Standardize variables
        X = DataPreparation()._standardize_variables(X)

        # Variable screening to select top features
        selected_vars, X, y = DataPreparation().variable_screening(X.columns.tolist(), df)
        X = X[selected_vars]

        return X, y

    def run_random_forest(self):
        X, y = self.preprocess_data()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Weighted Random Forest Model
        class_weights = class_weight.compute_class_weight(class_weight={0: 1, 1: 30},
                                                          classes=np.unique(y_train),
                                                          y=y_train)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=class_weights_dict)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]

        # Adjust threshold to improve recall
        threshold = 0.3  # Lowering threshold
        y_pred = (y_proba >= threshold).astype(int)

        # Add Random Forest predictions to the data_long for GEE
        self.data_long['rf_predictions'] = rf.predict(self.data_long[X.columns])

        return self.data_long, X.columns  # Return updated data for use in GEE

class GeeModel:
    def __init__(self, data):
        self.data = data
        self.subject_col = 'ID'
        self.data_long = DataPreparation().handle_data(self.data)
        if not os.path.exists('gee/output'):
            os.makedirs('gee/output')  # Ensure output directory exists

    def calculate_vif(self, X):
        X = X.assign(constant=1)  # Add constant term for VIF calculation
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    
    def custom_gee_cross_validation(self, X, y, groups, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            groups_train = groups.iloc[train_index]

            # Add constant for intercept
            X_train = sm.add_constant(X_train)
            X_test = sm.add_constant(X_test)

            # Define and fit GEE model
            model = sm.GEE(y_train, X_train, groups=groups_train, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
            result = model.fit()

            # Predict probabilities for the test set
            y_pred_prob = result.predict(X_test)
            
            # Compute ROC AUC score
            auc = roc_auc_score(y_test, y_pred_prob)
            auc_scores.append(auc)
            
        return auc_scores

    def run_gee_model(self, rf_data_long, rf_features):
        # Preprocess the training data
        selected_vars, X, y = DataPreparation().variable_screening(rf_features, rf_data_long)

        # Split the data into training and testing sets
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        # Calculate VIF for the training data
        vif_data = self.calculate_vif(X_train_full)
        print(vif_data)

        # Remove features with VIF > 5
        high_vif = vif_data[vif_data['VIF'] > 5]['Feature']
        selected_vars = [var for var in selected_vars if var not in high_vif]
        X_train = X_train_full[selected_vars]

        # Add a constant to the training and test data
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test_full[selected_vars])

        # Ensure all data is in float64 format
        X_train = X_train.astype('float64')
        X_test = X_test.astype('float64')

        # Define the GEE model and correlation structure
        groups_train = rf_data_long.loc[X_train_full.index, self.subject_col].reset_index(drop=True)
        ind = sm.cov_struct.Independence()

        # Fit the GEE model
        model = sm.GEE(y_train, X_train, groups=groups_train, family=sm.families.Binomial(), cov_struct=ind)
        result = model.fit()

        # Print and evaluate the model results
        print(result.summary())
        self.evaluate_predictive_performance(result, X_test, y_test)

        # Ensure all data is in float64 format
        X = X.astype('float64')
        y = y.astype('float64')

        # Prepare groups
        groups = rf_data_long[self.subject_col]

        # Run custom GEE cross-validation
        auc_scores = self.custom_gee_cross_validation(X[selected_vars], y, groups, n_splits=5)
        print('AUC scores: ', auc_scores)

    def evaluate_predictive_performance(self, result, X, y_true):
        """
        Evaluate the predictive performance of the GEE model.
        """
        # Get predicted probabilities
        y_pred_prob = result.predict(X)

        # Classify using threshold (e.g., 0.5)
        threshold = 0.5
        y_pred_class = (y_pred_prob >= threshold).astype(int)

        # Calculate evaluation metrics
        roc_auc = roc_auc_score(y_true, y_pred_prob)
        accuracy = accuracy_score(y_true, y_pred_class)
        precision = precision_score(y_true, y_pred_class)
        recall = recall_score(y_true, y_pred_class)
        f1 = f1_score(y_true, y_pred_class)

        # Print evaluation metrics
        print("\nPredictive Performance Metrics:")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        precision2, recall2, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(recall2, precision2)
        print(f"Precision-Recall AUC: {pr_auc}")

        # Compute and print confusion matrix
        cm = confusion_matrix(y_true, y_pred_class)
        print("\nConfusion Matrix:")
        print(cm)

        # Save evaluation metrics to a text file
        if not os.path.exists('gee/output'):
            os.makedirs('gee/output')

        with open('gee/output/predictive_performance_metrics.txt', 'w') as f:
            f.write("Predictive Performance Metrics:\n")
            f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        print("Predictive performance metrics saved to 'gee/output/predictive_performance_metrics.txt'")

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'GEE Model (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for GEE Model')
        plt.legend(loc='lower right')
        plt.savefig('gee/output/roc_curve.png')
        plt.show()
        print("ROC curve saved to 'gee/output/roc_curve.png'")

def main():
    # Load the dataset
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")
    filtered_data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Step 1: Run Random Forest Model
    rf_model = RandomForestLongitudinalModel(filtered_data)
    rf_data_long, rf_features = rf_model.run_random_forest()

    # Step 2: Run GEE Model using Random Forest predictions
    gee_model = GeeModel(filtered_data)
    gee_model.run_gee_model(rf_data_long, rf_features)

if __name__ == "__main__":
    main()