from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, accuracy_score, 
                             precision_score, recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import make_column_selector
from imblearn.pipeline import Pipeline
import seaborn as sns
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataPreparation

class RandomForestLongitudinalModel:
    def __init__(self, data, dataset_label):
        self.data = data
        self.subject_col = 'ID'
        self.dataset_label = dataset_label
        if not os.path.exists('random_forest_balanced/output'):
            os.makedirs('random_forest_balanced/output')  # Ensure output directory exists

    def calculate_vif(self, X):
        X = X.assign(constant=1)  # Add constant term for VIF calculation
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def preprocess_data(self):
        df = self.data.copy()

        # We will be predicting DepressivePathGroup, so first ensure that this column exists.
        # Exclude these columns from predictors:
        exclude_cols = ['ID', 'Pontuacao_Depressao', 'DepressivePathGroup',
                        'hads12', 'hads6', 'hads4', 'hads2', 'hads8', 'hads10', 'hads14', 'Time',
                        'DepressionStatus']
        X = df.loc[:, ~df.columns.str.startswith(tuple(exclude_cols))]
        y = df['DepressivePathGroup']

        # Ensure only numeric columns remain
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        # Replace NaN values with 0 for participant columns
        participante_columns = [col for col in df.columns if col.startswith('Participante')]
        df[participante_columns] = df[participante_columns].fillna(0)

        # Handle missing values (if any)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Process categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
            X[col] = X[col].astype('category')

        # Variable screening
        selected_vars, X, y = DataPreparation().variable_screening(X.columns.tolist(), df)

        # Remove Time or DepressivePathGroup if they appear in selected vars
        for col_to_remove in ['Time', 'DepressivePathGroup']:
            if col_to_remove in selected_vars:
                selected_vars.remove(col_to_remove)

        X = X[selected_vars]

        return X, y

    def run_random_forest(self):
        self.create_depressive_path_columns()

        X, y = self.preprocess_data()

        # Exclude 'DepressivePath' from predictors (it's for grouping, not prediction)
        if 'DepressivePath' in X.columns:
            X = X.drop(columns=['DepressivePath'])

        vif_data = self.calculate_vif(X)
        print('Random Forest VIF:', vif_data)

        # Split the data using the path groups (y) for stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Apply SMOTETomek on the training data
        smote_tomek = SMOTETomek(random_state=42)
        X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)

        # Reconstruct a balanced dataset (train + test) for evaluation and for downstream use
        data_bal = pd.concat([
            pd.DataFrame(X_train_bal, columns=X.columns),
            pd.DataFrame(X_test, columns=X.columns)
        ], ignore_index=True)

        # Create the DepressivePath and Group again in the balanced data
        data_bal['DepressivePath'] = (
            data_bal[['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']]
            .fillna(-1)
            .astype(int)
            .astype(str)
            .agg(''.join, axis=1)
        )

        path_to_group = {
            '000': 1,
            '001': 2, '010': 2, '100': 2, '101': 2,
            '011': 3, '110': 3, '111': 3
        }

        data_bal['DepressivePathGroup'] = data_bal['DepressivePath'].map(path_to_group).fillna(0).astype(int)

        # Update train/test splits with the balanced set (only train part is balanced)
        # Note: y_train_bal and X_train_bal already balanced
        # y_test unchanged; X_test unchanged

        # Define class weights for the multinomial scenario
        # We'll treat this as a multi-class classification with RF
        classes = np.unique(y_train)
        class_weights_dict = None  # TODO: Add class weights again

        pipeline = Pipeline([
            ('classification', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'))
        ])

        pipeline.fit(X_train_bal, y_train_bal)

        # Predictions on test
        y_proba_test = pipeline.predict_proba(X_test)
        # For evaluation per path group, we consider one-vs-all. We'll use predicted class probabilities.
        y_pred_original = pipeline.predict(X_test)

        # Feature columns and full prediction
        feature_columns = X.columns
        X_full = self.data[feature_columns].fillna(0)
        full_proba = pipeline.predict_proba(X_full)
        self.data['rf_predictions'] = pipeline.predict(X_full)

        # Extract important features (top 10)
        rf_model = pipeline.named_steps['classification']
        #self.extract_rf_important_features(rf_model, feature_columns, self.dataset_label)

        # Run separate models for each depressive path group
        self.run_rf_for_each_pathgroup(data_bal, feature_columns, self.dataset_label)

    def plot_confusion_matrix(self, y_true, y_pred, dataset_label, path_group):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Confusion Matrix - {dataset_label} at Path Group {path_group}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'random_forest_balanced/output/confusion_matrix_{dataset_label.replace(" ", "_")}_PG{path_group}.png', dpi=100, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, dataset_label, path_group):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_label} at Path Group {path_group}')
        plt.legend(loc="lower right")
        plt.savefig(f'random_forest_balanced/output/roc_curve_{dataset_label.replace(" ", "_")}_PG{path_group}.png', dpi=100, bbox_inches='tight')
        plt.close()

    def run_rf_for_each_pathgroup(self, data_bal, feature_columns, dataset_label):
        """
        Run separate Random Forest models for each depressive path group as a binary classification:
        That group vs. all others.
        """
        path_groups = [1, 2, 3]

        # Store metrics and data for plotting
        metrics = []
        plot_data = []

        for pg in path_groups:
            subset = data_bal.copy()
            subset['y_binary'] = (subset['DepressivePathGroup'] == pg).astype(int)

            if subset['y_binary'].nunique() < 2:
                print(f"Insufficient data or classes at Path Group {pg}. Cannot train model.")
                continue

            # Prepare data
            exclude_cols = ['DepressivePath', 'DepressivePathGroup', 'Score_Depressao', 'y_binary',
                            'ID', 'Pontuacao_Depressao', 'hads12', 'hads6',
                            'hads4', 'hads2', 'hads8', 'hads10', 'hads14']
            X = subset.loc[:, ~subset.columns.str.startswith(tuple(exclude_cols))]
            y = subset['y_binary']

            if X.empty:
                print(f"No features at Path Group {pg}.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            if y_test.nunique() > 1:
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = np.nan

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"\nPerformance Metrics for Path Group {pg} (Group vs Rest) - RF:")
            if not np.isnan(roc_auc):
                print(f"ROC AUC Score: {roc_auc:.4f}")
            else:
                print("ROC AUC: Undefined")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # Top 10 Important Features
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            important_features = feature_importances.nlargest(10)

            print(f"\nTop 10 Important Features at Path Group {pg} vs Rest - RF:")
            print(important_features)
            important_features.to_csv(f'random_forest_balanced/output/random_forest_important_features_{dataset_label.replace(" ", "_")}_PG{pg}.csv')

            # Collect data for plotting
            metrics.append({'pg': pg, 'roc_auc': roc_auc, 'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'f1': f1})
            plot_data.append({'pg': pg, 'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba})

        # Plot confusion matrix and ROC curve for all path groups after the loop
        for data in plot_data:
            pg = data['pg']
            y_test = data['y_test']
            y_pred = data['y_pred']
            y_proba = data['y_proba']

            self.plot_confusion_matrix(y_test, y_pred, dataset_label, pg)

            if y_test.nunique() > 1:
                self.plot_roc_curve(y_test, y_proba, dataset_label, pg)

    def create_depressive_path_columns(self):
        # Create DepressivePath and DepressivePathGroup columns in self.data if not exist
        if not {'Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'}.issubset(self.data.columns):
            raise ValueError("Missing necessary score columns in the dataset.")

        score_columns = [col for col in self.data.columns if col.startswith('Score')]
        print('Score Columns:', score_columns)

        self.data['DepressivePath'] = (
            self.data[['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']]
            .fillna(-1)
            .astype(int)
            .astype(str)
            .agg(''.join, axis=1)
        )

        path_to_group = {
            '000': 1,
            '001': 2, '010': 2, '100': 2, '101': 2,
            '011': 3, '110': 3, '111': 3
        }
        self.data['DepressivePathGroup'] = self.data['DepressivePath'].map(path_to_group).fillna(0).astype(int)

def main():
    # Load the dataset
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")
    filtered_data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    dataset_label = 'SMOTETomek Balanced Data'

    # Run Random Forest Model with Y as DepressivePathGroup
    rf_model = RandomForestLongitudinalModel(filtered_data, dataset_label)
    rf_model.run_random_forest()

if __name__ == "__main__":
    main()