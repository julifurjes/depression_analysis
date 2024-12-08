from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, accuracy_score, 
                             precision_score, recall_score, f1_score, precision_recall_curve, auc, confusion_matrix)
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
from imblearn.pipeline import Pipeline
import seaborn as sns
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataPreparation  # Ensure this file exists with the DataPreparation class

class RandomForestLongitudinalModel:
    def __init__(self, data, dataset_label):
        self.data = data
        self.subject_col = 'ID'
        self.data_long = DataPreparation().handle_data(self.data)
        self.dataset_label = dataset_label
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

        # Exclude specific columns and 'Time' from predictors
        exclude_cols = ['ID', 'DepressionScore', 'Pontuacao_Depressao', 'Score_Depressao',
                        'hads12', 'hads6', 'hads4', 'hads2', 'hads8', 'hads10', 'hads14', 'Time']
        X = df.drop(columns=exclude_cols + ['DepressionStatus'], errors='ignore')
        y = df['DepressionStatus']

        # Handle missing values (if any)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Process categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
            X[col] = X[col].astype('category')

        # Variable screening (Time is already removed)
        selected_vars, X, y = DataPreparation().variable_screening(X.columns.tolist(), df)
        # Ensure Time is not re-introduced
        if 'Time' in selected_vars:
            selected_vars.remove('Time')
        X = X[selected_vars]

        return X, y

    def run_random_forest(self):
        X, y = self.preprocess_data()

        vif_data = self.calculate_vif(X)
        print('Random Forest VIF:', vif_data)

        # Split the data into training and testing sets
        # Note: Time is excluded from X, but we still have it in data_long for grouping
        df_temp = self.data_long.copy()
        # Extract Time for later evaluation
        # Align indices
        df_temp = df_temp.loc[X.index]
        time = df_temp['Time']

        X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
            X, y, time, stratify=y, test_size=0.2, random_state=42
        )

        # Apply SMOTETomek to balance the training set
        smote_tomek = SMOTETomek(random_state=42)
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

        # Define class weights
        class_weights = class_weight.compute_class_weight(
            class_weight={0: 1, 1: 10}, classes=np.unique(y_train), y=y_train
        )
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Define pipeline
        pipeline = Pipeline([
            ('sampling', SMOTETomek(random_state=42)),
            ('classification', RandomForestClassifier(n_estimators=300, random_state=42, class_weight=class_weights_dict))
        ])

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Predictions on test
        y_pred_original = pipeline.predict(X_test)
        y_proba_original = pipeline.predict_proba(X_test)[:, 1]

        # Evaluate per timestamp with visualizations
        self.evaluate_performance_per_timestamp(y_test, y_proba_original, time_test, self.dataset_label, threshold=0.2)

        # Feature columns and full prediction
        feature_columns = X_train.columns
        X_full = self.data_long[feature_columns].fillna(0)  # Ensure no missing for predict
        self.data_long['rf_predictions'] = pipeline.predict(X_full)

        # Extract important features
        rf_model = pipeline.named_steps['classification']
        self.extract_rf_important_features(rf_model, feature_columns, self.dataset_label)

        # Run separate models for each timestamp
        self.run_rf_for_each_timestamp()

        return self.data_long, feature_columns

    def plot_confusion_matrix(self, y_true, y_pred, dataset_label, time_point):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Confusion Matrix - {dataset_label} at Time T{time_point}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'gee_rf/output/confusion_matrix_{dataset_label.replace(" ", "_")}_T{time_point}.png', dpi=100, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, dataset_label, time_point):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_label} at Time T{time_point}')
        plt.legend(loc="lower right")
        plt.savefig(f'gee_rf/output/roc_curve_{dataset_label.replace(" ", "_")}_T{time_point}.png', dpi=100, bbox_inches='tight')
        plt.close()

    def evaluate_performance_per_timestamp(self, y_true, y_proba, time_test, dataset_label, threshold=0.2):
        y_pred = (y_proba >= threshold).astype(int)
        test_results = pd.DataFrame({
            'Time': time_test.reset_index(drop=True),
            'y_true': y_true.reset_index(drop=True),
            'y_pred': y_pred,
            'y_proba': y_proba
        })

        for time_point in test_results['Time'].unique():
            time_data = test_results[test_results['Time'] == time_point]
            y_true_time = time_data['y_true']
            y_pred_time = time_data['y_pred']
            y_proba_time = time_data['y_proba']

            roc_auc = roc_auc_score(y_true_time, y_proba_time)
            accuracy = accuracy_score(y_true_time, y_pred_time)
            precision = precision_score(y_true_time, y_pred_time, zero_division=0)
            recall = recall_score(y_true_time, y_pred_time, zero_division=0)
            f1 = f1_score(y_true_time, y_pred_time, zero_division=0)

            print(f"\nPerformance Metrics for {dataset_label} at Time T{time_point}:")
            print(f"ROC AUC Score: {roc_auc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # Plot confusion matrix and ROC curve for this time point
            self.plot_confusion_matrix(y_true_time, y_pred_time, dataset_label, time_point)
            self.plot_roc_curve(y_true_time, y_proba_time, dataset_label, time_point)

    def run_rf_for_each_timestamp(self):
        """
        Run separate Random Forest models for each timestamp (T0, T1, T3)
        and print the top features for each.
        """
        time_points = [0, 1, 3]
        for t in time_points:
            subset = self.data_long[self.data_long['Time'] == t].copy()

            exclude_cols = ['ID', 'DepressionScore', 'Pontuacao_Depressao', 'Score_Depressao',
                            'hads12', 'hads6', 'hads4', 'hads2', 'hads8', 'hads10', 'hads14', 'Time', 'DepressionStatus']
            X = subset.drop(columns=[col for col in exclude_cols if col in subset.columns], errors='ignore')
            y = subset['DepressionStatus']

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                X[col] = X[col].astype('category').cat.codes
                X[col] = X[col].astype('category')

            # Variable screening without Time
            selected_vars, X_selected, y_selected = DataPreparation().variable_screening(X.columns.tolist(), subset)
            # Ensure Time is removed if present
            if 'Time' in selected_vars:
                selected_vars.remove('Time')

            X = X_selected
            y = y_selected

            if X.empty or len(y.unique()) < 2:
                print(f"Insufficient data or classes at Time T{t}. Cannot train model.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

            smote_tomek = SMOTETomek(random_state=42)
            X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)

            class_weights = class_weight.compute_class_weight(
                class_weight={0: 1, 1: 10}, classes=np.unique(y_train), y=y_train
            )
            class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

            model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=class_weights_dict)
            model.fit(X_train_bal, y_train_bal)

            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            important_features = feature_importances.nlargest(10)

            print(f"\nTop 10 Important Features at Time T{t}:")
            print(important_features)

    def extract_rf_important_features(self, model, feature_names, dataset_label):
        feature_importances = pd.Series(model.feature_importances_, index=feature_names)
        important_features = feature_importances.nlargest(10)

        print("\nTop 10 Important Features in Random Forest (All Times Combined):")
        print(important_features)

        # Save
        important_features.to_csv(f'gee_rf/output/random_forest_important_features_{dataset_label.replace(" ", "_")}.csv')
        print("Important features saved.")


class GeeModel:
    def __init__(self, data, dataset_label):
        self.data = data
        self.subject_col = 'ID'
        self.data_long = DataPreparation().handle_data(self.data)
        self.dataset_label = dataset_label
        if not os.path.exists('gee_rf/output'):
            os.makedirs('gee_rf/output')  # Ensure output directory exists

    def preprocess_data(self):
        df = self.data_long.copy()

        # Exclude specific columns and Time from predictors
        exclude_cols = ['ID', 'DepressionScore', 'Pontuacao_Depressao', 'Score_Depressao',
                        'hads12', 'hads6', 'hads4', 'hads2', 'hads8', 'hads10', 'hads14', 'Time']
        X = df.drop(columns=exclude_cols + ['DepressionStatus'], errors='ignore')
        y = df['DepressionStatus']

        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Process categorical vars
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
            X[col] = X[col].astype('category')

        # Variable screening without Time
        selected_vars, X, y = DataPreparation().variable_screening(X.columns.tolist(), df)
        if 'Time' in selected_vars:
            selected_vars.remove('Time')

        X = X[selected_vars]

        return X, y, selected_vars

    def calculate_vif(self, X):
        X = X.assign(constant=1)
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def run_gee_model(self, rf_data_long, rf_features):
        X, y, selected_vars = self.preprocess_data()

        # Extract Time separately for grouping but do not use it as predictor
        df_temp = rf_data_long.loc[X.index]
        time = df_temp['Time']

        X_train_full, X_test_full, y_train, y_test, time_train, time_test = train_test_split(
            X, y, time, test_size=0.3, random_state=42, stratify=y
        )

        # No Time in predictors
        vif_data = self.calculate_vif(X_train_full)
        print('GEE VIF: ', vif_data)

        high_vif = vif_data[vif_data['VIF'] > 5]['Feature']
        selected_vars = [var for var in selected_vars if var not in high_vif and var in X_train_full.columns]

        X_train = X_train_full[selected_vars]
        X_test = X_test_full[selected_vars]

        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        X_train = X_train.astype('float64')
        X_test = X_test.astype('float64')

        groups_train = rf_data_long.loc[X_train_full.index, self.subject_col].reset_index(drop=True)
        ind = sm.cov_struct.Independence()

        model = sm.GEE(y_train, X_train, groups=groups_train, family=sm.families.Binomial(), cov_struct=ind)
        result = model.fit()

        print(result.summary())
        self.evaluate_predictive_performance(result, X_test, y_test, time_test, self.dataset_label)

        # Also run GEE model for each timestamp
        self.run_gee_for_each_timestamp(rf_data_long, rf_features)

        return result

    def run_gee_for_each_timestamp(self, rf_data_long, rf_features):
        time_points = [0, 1, 3]

        for t in time_points:
            subset = rf_data_long[rf_data_long['Time'] == t].copy()

            if subset['DepressionStatus'].nunique() < 2:
                print(f"Not enough variation in DepressionStatus at Time T{t} for GEE.")
                continue

            exclude_cols = ['ID', 'DepressionScore', 'Pontuacao_Depressao', 'Score_Depressao',
                            'hads12', 'hads6', 'hads4', 'hads2', 'hads8', 'hads10', 'hads14', 'DepressionStatus', 'Time']
            X = subset.drop(columns=[col for col in exclude_cols if col in subset.columns], errors='ignore')
            y = subset['DepressionStatus']

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                X[col] = X[col].astype('category').cat.codes
                X[col] = X[col].astype('category')

            selected_vars, X_selected, y_selected = DataPreparation().variable_screening(X.columns.tolist(), subset)
            if 'Time' in selected_vars:
                selected_vars.remove('Time')

            X = X_selected
            y = y_selected

            if X.empty or len(y.unique()) < 2:
                print(f"No features or not enough classes at Time T{t} for GEE.")
                continue

            groups = subset.loc[X.index, self.subject_col].reset_index(drop=True)

            X = sm.add_constant(X)
            X = X.astype('float64')

            ind = sm.cov_struct.Independence()
            model = sm.GEE(y, X, groups=groups, family=sm.families.Binomial(), cov_struct=ind)
            result = model.fit()

            print(f"\nGEE Results at Time T{t}:")
            print(result.summary())

            pvalues = result.pvalues
            significant_features = pvalues[pvalues < 0.05]
            print(f"\nSignificant Features in GEE at Time T{t}:")
            print(significant_features)

            significant_features.to_csv(f'gee_rf/output/gee_important_features_{self.dataset_label.replace(" ", "_")}_T{t}.csv')
            print("Significant features saved.")

            self.extract_significant_gee_features(result, self.dataset_label)

    def extract_significant_gee_features(self, model, dataset_label):
        pvalues = model.pvalues
        significant_features = pvalues[pvalues < 0.05]
        print("\nSignificant Features in GEE Model:")
        print(significant_features)
        significant_features.to_csv(f'gee_rf/output/gee_significant_features_{dataset_label.replace(" ", "_")}.csv')
        print("Significant features saved.")

    def evaluate_predictive_performance(self, result, X, y_true, time_test, dataset_label):
        X = X.reset_index(drop=True)
        y_true = y_true.reset_index(drop=True)
        time_test = time_test.reset_index(drop=True)

        y_pred_prob = result.predict(X)
        threshold = 0.2
        y_pred_class = (y_pred_prob >= threshold).astype(int)

        test_results = pd.DataFrame({
            'Time': time_test,
            'y_true': y_true,
            'y_pred': y_pred_class,
            'y_proba': y_pred_prob
        })

        test_results = test_results[~test_results['Time'].isna()]
        test_results = test_results[~test_results['Time'].isin([np.inf, -np.inf])]
        time_test = time_test.astype('int64')
        test_results['Time'] = test_results['Time'].astype('int64')

        # Evaluate per timestamp
        for time_point in sorted(test_results['Time'].unique()):
            time_data = test_results[test_results['Time'] == time_point]
            y_true_time = time_data['y_true']
            y_pred_time = time_data['y_pred']
            y_proba_time = time_data['y_proba']

            if y_true_time.isna().any() or y_proba_time.isna().any():
                print(f"Skipping Time {time_point} due to NaN values.")
                continue

            if len(y_true_time.unique()) > 1:
                roc_auc = roc_auc_score(y_true_time, y_proba_time)
            else:
                roc_auc = 'Undefined (only one class present)'

            accuracy = accuracy_score(y_true_time, y_pred_time)
            precision = precision_score(y_true_time, y_pred_time, zero_division=0)
            recall = recall_score(y_true_time, y_pred_time, zero_division=0)
            f1 = f1_score(y_true_time, y_pred_time, zero_division=0)

            print(f"\nPredictive Performance Metrics for {dataset_label} at Time T{time_point}:")
            print(f"ROC AUC Score: {roc_auc}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

def main():
    # Load the dataset
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")
    filtered_data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    dataset_label = 'SMOTETomek Balanced Data'

    # Step 1: Run Random Forest Model
    rf_model = RandomForestLongitudinalModel(filtered_data, dataset_label)
    rf_data_long, rf_features = rf_model.run_random_forest()

    # Step 2: Run GEE Model using Random Forest predictions
    gee_model = GeeModel(filtered_data, dataset_label)
    gee_model.run_gee_model(rf_data_long, rf_features)

if __name__ == "__main__":
    main()