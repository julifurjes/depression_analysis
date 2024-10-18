import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

class RandomForestLongitudinalModel:
    def __init__(self, data):
        self.data = data
        self.subject_col = 'ID'
        self.data_long = self.prepare_long_data()

    def process_datetime_columns(self, X):
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X[col + '_year'] = X[col].dt.year
            X[col + '_month'] = X[col].dt.month
            X[col + '_day'] = X[col].dt.day
        X = X.drop(columns=datetime_cols)
        return X

    def prepare_long_data(self):
        data = self.data.copy()
        score_cols = ['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']
        data_long = pd.melt(data, id_vars=['ID'], value_vars=score_cols,
                            var_name='Time', value_name='DepressionScore')

        data_long['Time'] = data_long['Time'].str.extract(r'T(\d+)').astype(int)

        repeated_vars = [col[:-3] for col in data.columns if col.endswith(('_T0', '_T1', '_T3'))]
        repeated_vars = list(set(repeated_vars))

        for var in repeated_vars:
            time_cols = [f"{var}_T0", f"{var}_T1", f"{var}_T3"]
            available_time_cols = [col for col in time_cols if col in data.columns]
            if not available_time_cols:
                continue
            melted = pd.melt(data[['ID'] + available_time_cols], id_vars=['ID'], value_vars=available_time_cols,
                             var_name='TimeVar', value_name=var)
            melted['Time'] = melted['TimeVar'].str.extract(r'T(\d+)').astype(int)
            melted = melted.drop('TimeVar', axis=1)
            data_long = pd.merge(data_long, melted, on=['ID', 'Time'], how='left')

        time_invariant_vars = [col for col in data.columns if not col.endswith(('_T0', '_T1', '_T3')) and col != 'ID']
        data_long = pd.merge(data_long, data[['ID'] + time_invariant_vars], on='ID', how='left')

        threshold = 0
        data_long['DepressionStatus'] = (data_long['DepressionScore'] > threshold).astype(int)

        return data_long

    def preprocess_data(self):
        df = self.data_long.copy()

        # Exclude unnecessary columns, including 'Pontuacao_Depressao' and 'Score_Depressao'
        exclude_cols = ['ID', 'DepressionScore', 'Pontuacao_Depressao', 'Score_Depressao']
        X = df.drop(columns=exclude_cols + ['DepressionStatus'])

        y = df['DepressionStatus']

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        X = self.process_datetime_columns(X)

        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        return X, y

    def handle_class_imbalance(self, y):
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(y),
                                                          y=y)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        return class_weights_dict

    def run_random_forest(self, method='balanced_rf'):
        X, y = self.preprocess_data()

        # Create a directory for the current method
        method_folder = f'long_rf/output/{method}'
        if not os.path.exists(method_folder):
            os.makedirs(method_folder)

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
            class_weights = self.handle_class_imbalance(y_train)
            rf = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight=class_weights
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]

        # Evaluate the model
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")

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
        plt.show()

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
        plt.show()

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
    rf_model.run_random_forest(method='weighted')  # You can choose 'smote', 'balanced_rf', or 'weighted'

if __name__ == "__main__":
    main()