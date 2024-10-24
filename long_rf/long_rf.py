import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

    def run_random_forest(self, method='balanced_rf'):
        X, y = self.preprocess_data()

        # Create a directory for the current method
        method_folder = f'long_rf/output/{method}'
        if not os.path.exists(method_folder):
            os.makedirs(method_folder)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        if method == 'smote':
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_resampled, y_train_resampled)

            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]

        elif method == 'balanced_rf':
            # Use Balanced Random Forest
            brf = BalancedRandomForestClassifier(
                n_estimators=100, random_state=42, replacement=False
            )
            brf.fit(X_train, y_train)

            y_pred = brf.predict(X_test)
            y_proba = brf.predict_proba(X_test)[:, 1]

        elif method == 'weighted':
            # Default: Original Random Forest with class weights
            class_weights = class_weight.compute_class_weight(class_weight={0: 1, 1: 30},
                                                              classes=np.unique(y_train),
                                                              y=y_train)
            class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
            rf = RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight=class_weights_dict
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]

            # Adjust threshold to improve recall
            threshold = 0.3  # Lowering threshold
            y_pred = (y_proba >= threshold).astype(int)

        # Evaluate the model
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")

        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='f1')
        print('F1 cross-validation scores:', scores)

        # Plot and save ROC Curve
        self.plot_roc_curve(y_test, y_proba, roc_auc, method_folder)

        # Feature Importance
        if method == 'balanced_rf':
            feature_importances = pd.Series(brf.feature_importances_, index=X.columns)
        else:
            feature_importances = pd.Series(rf.feature_importances_, index=X.columns)

        self.plot_feature_importance(feature_importances, method_folder)

        # Save results
        self.save_results(y_test, y_pred, roc_auc, feature_importances, method_folder)

    def plot_roc_curve(self, y_test, y_proba, roc_auc, method_folder):
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Random Forest')
        plt.legend(loc='lower right')
        plt.savefig(f'{method_folder}/roc_curve_{roc_auc:.4f}.png')  # Save ROC curve
        plt.close()

    def plot_feature_importance(self, feature_importances, method_folder):
        sorted_idx = feature_importances.sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        sorted_idx.head(20).plot(kind='barh')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{method_folder}/feature_importance.png')  # Save Feature Importance plot
        plt.close()

    def save_results(self, y_test, y_pred, roc_auc, feature_importances, method_folder):
        # Save classification report
        with open(f'{method_folder}/classification_report.txt', 'w') as f:
            f.write(classification_report(y_test, y_pred))
        
        # Save ROC AUC score
        with open(f'{method_folder}/roc_auc_score.txt', 'w') as f:
            f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
        
        # Save feature importances
        feature_importances.to_csv(f'{method_folder}/feature_importances.csv')

def main():
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")
    data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    rf_model = RandomForestLongitudinalModel(data)
    rf_model.run_random_forest(method='weighted')

if __name__ == "__main__":
    main()