from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTENC
from sklearn.compose import make_column_selector
from imblearn.under_sampling import TomekLinks
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataPreparation

class RandomForestLongitudinalModel:
    def __init__(self, data):
        self.data = data
        self.subject_col = 'ID'
        self.data_long = DataPreparation().handle_data(self.data)
        if not os.path.exists('gee_rf/output'):
            os.makedirs('gee_rf/output')  # Ensure output directory exists

    def calculate_vif(self, X):
        X = X.assign(constant=1)  # Add constant term for VIF calculation
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

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
            X[col] = X[col].astype('category')

        # Variable screening to select top features
        selected_vars, X, y = DataPreparation().variable_screening(X.columns.tolist(), df)
        X = X[selected_vars]

        return X, y

    def run_random_forest(self):
        X, y = self.preprocess_data()

        vif_data = self.calculate_vif(X)
        print('Random Forest VIF:', vif_data)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Identify categorical columns in your dataset
        categorical_indices = make_column_selector(dtype_include="category")(X_train)
        categorical_features = [X_train.columns.get_loc(col) for col in categorical_indices]

        # Apply SMOTETomek to balance the training set
        smote_tomek = SMOTETomek(random_state=42)
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

        class_weight_options = [{0: 1, 1: w} for w in [5, 10, 20, 30]]
        thresholds = [0.5, 0.4, 0.3, 0.2]
        performance_file = 'gee_rf/output/performance_metrics_all_combinations.txt'
        
        with open(performance_file, 'w') as f:
            f.write("Performance Metrics for All Combinations\n")
            f.write("========================================\n\n")

        for cw in class_weight_options:
            rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=cw)
            rf.fit(X_train_balanced, y_train_balanced)
            y_proba = rf.predict_proba(X_test)[:, 1]

            for threshold in thresholds:
                print(f"\nEvaluating for Class Weight {cw} and Threshold {threshold}")
                metrics = self.evaluate_performance(y_test, y_proba, threshold)

                with open(performance_file, 'a') as f:
                    f.write(f"Class Weight: {cw}, Threshold: {threshold}\n")
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")

        self.data_long['rf_predictions'] = rf.predict(self.data_long[X.columns])
        self.extract_rf_important_features(rf, X.columns)
        print(f"All performance metrics saved to '{performance_file}'")
        return self.data_long, X.columns  # Return updated data for use in GEE

    def extract_rf_important_features(self, model, feature_names):
        """
        Extract the most important features from the Random Forest model and save them.
        """
        feature_importances = pd.Series(model.feature_importances_, index=feature_names)
        important_features = feature_importances.nlargest(10)  # Top 10 important features

        print("\nTop 10 Important Features in Random Forest:")
        print(important_features)

        # Save important features to a CSV file
        important_features.to_csv('gee_rf/output/random_forest_important_features.csv')
        print("Important features saved to 'gee_rf/output/random_forest_important_features.csv'")

    def evaluate_performance(self, y_true, y_proba, dataset_label, threshold=0.3):
        """
        Evaluate and print model performance, including confusion matrix.
        """
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "ROC AUC Score": f"{roc_auc_score(y_true, y_proba):.4f}",
            "Accuracy": f"{accuracy_score(y_true, y_pred):.4f}",
            "Precision": f"{precision_score(y_true, y_pred):.4f}",
            "Recall": f"{recall_score(y_true, y_pred):.4f}",
            "F1 Score": f"{f1_score(y_true, y_pred):.4f}"
        }
        precision2, recall2, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall2, precision2)
        metrics["Precision-Recall AUC"] = f"{pr_auc:.4f}"
        cm = confusion_matrix(y_true, y_pred)
        metrics["Confusion Matrix"] = f"{cm}"
        return metrics

class GeeModel:
    def __init__(self, data):
        self.data = data
        self.subject_col = 'ID'
        self.data_long = DataPreparation().handle_data(self.data)
        if not os.path.exists('gee_rf/output'):
            os.makedirs('gee_rf/output')  # Ensure output directory exists

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
            model = sm.GEE(y_train, X_train, groups=groups_train, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence())
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
        print('GEE VIF: ', vif_data)

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

        # Extract significant features based on p-value
        self.extract_significant_gee_features(result)

        return result

    def extract_significant_gee_features(self, model):
        """
        Extract statistically significant features from the GEE model based on p-value.
        """
        pvalues = model.pvalues
        significant_features = pvalues[pvalues < 0.05]  # Consider p-value < 0.05 as significant

        print("\nSignificant Features in GEE Model:")
        print(significant_features)

        # Save significant features to a CSV file
        significant_features.to_csv('gee_rf/output/gee_significant_features.csv')
        print("Significant features saved to 'gee_rf/output/gee_significant_features.csv'")

    def evaluate_predictive_performance(self, result, X, y_true):
        """
        Evaluate the predictive performance of the GEE model.
        """
        # Get predicted probabilities
        y_pred_prob = result.predict(X)

        # Classify using threshold (e.g., 0.3)
        threshold = 0.3
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
        if not os.path.exists('gee_rf/output'):
            os.makedirs('gee_rf/output')

        with open('gee_rf/output/predictive_performance_metrics.txt', 'w') as f:
            f.write("Predictive Performance Metrics:\n")
            f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        print("Predictive performance metrics saved to 'gee_rf/output/predictive_performance_metrics.txt'")

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'GEE Model (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for GEE Model')
        plt.legend(loc='lower right')
        plt.savefig('gee_rf/output/roc_curve.png')
        plt.show()
        print("ROC curve saved to 'gee_rf/output/roc_curve.png'")

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