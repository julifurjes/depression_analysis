import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from pymer4.models import Lmer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataPreparation

class MixedEffectsDepressionModel:
    def __init__(self, data):
        self.data = data
        self.subject_col = 'ID'
        self.data_long = DataPreparation().handle_data(self.data)
        if not os.path.exists('mixed_effects/output'):
            os.makedirs('mixed_effects/output')

    def run_model(self):
        print(f"Running Improved Mixed-Effects Model")

        # Prepare the data
        exclude_cols = ['ID', 'Time', 'DepressionStatus', 'DepressionScore', 'Score_Depressao']
        exclude_cols += [col for col in self.data_long.columns if 'Depress' in col or 'Score' in col]

        fixed_effects = [col for col in self.data_long.columns if col not in exclude_cols]

        # Handle missing values
        self.data_long = self.data_long.dropna(subset=fixed_effects + ['DepressionStatus'])

        # Perform variable screening to select top features
        selected_features, X, y = DataPreparation().variable_screening(fixed_effects, self.data_long)

        print(f"Selected features: {selected_features}")

        # If no features are selected, stop
        if len(selected_features) == 0:
            print("No predictors selected.")
            return

        # Check for multicollinearity among selected features
        if len(selected_features) > 1:
            X_selected = X[selected_features]
            vif_data = pd.DataFrame()
            vif_data['feature'] = X_selected.columns
            vif_data['VIF'] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]
            print("VIF for selected features:")
            print(vif_data)

            # Remove features with high VIF
            high_vif = vif_data[vif_data['VIF'] > 5]['feature'].tolist()
            X_selected = X_selected.drop(columns=high_vif)
            selected_features = X_selected.columns.tolist()
            print(f"Features after removing high VIF: {selected_features}")
        else:
            X_selected = X[selected_features]

        # Prepare the final data for modeling
        self.data_long = self.data_long.reset_index(drop=True)
        # Ensure 'ID' is a string before converting to category
        self.data_long['ID'] = self.data_long['ID'].astype(str).astype('category')

        final_data = pd.concat([self.data_long[['ID', 'DepressionStatus']], X_selected], axis=1)

        # Build the formula
        formula = 'DepressionStatus ~ ' + ' + '.join(selected_features) + ' + (1|ID)'

        # Split data into training and testing sets
        train_data, test_data = train_test_split(final_data, test_size=0.2, random_state=42)

        # Fit the model on the training data
        model = Lmer(formula, data=train_data, family='binomial')
        result = model.fit(factors=None)

        # Print the summary
        print(result)

        # Save results
        self.save_results(result)

        # Evaluate on the test data
        self.evaluate_predictive_performance(model, test_data)

    def save_results(self, result):
        output_file = f'mixed_effects/output/mixed_effects_results.txt'
        with open(output_file, 'w') as f:
            f.write(result.to_string())
        print(f"Results saved to {output_file}")

        coef_df = result
        coef_df.to_csv(f'mixed_effects/output/mixed_effects_coefficients.csv', index=False)
        print(f"Coefficients saved to 'mixed_effects/output/mixed_effects_coefficients.csv'")

    def evaluate_predictive_performance(self, model, data):
        """
        Evaluate the predictive performance of the mixed-effects model.
        """
        # Get predicted probabilities without random effects
        y_pred_prob = model.predict(data, verify_predictions=False, use_rfx=False)

        # Convert y_pred_prob to NumPy array
        y_pred_prob = np.array(y_pred_prob).astype(float)

        # True labels
        y_true = data['DepressionStatus']

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

        # Save evaluation metrics to a text file
        with open('mixed_effects/output/predictive_performance_metrics.txt', 'w') as f:
            f.write("Predictive Performance Metrics:\n")
            f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        print("Predictive performance metrics saved to 'mixed_effects/output/predictive_performance_metrics.txt'")

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Mixed Effects Model (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Mixed Effects Model')
        plt.legend(loc='lower right')
        plt.savefig('mixed_effects/output/roc_curve.png')
        plt.show()
        print("ROC curve saved to 'mixed_effects/output/roc_curve.png'")

def main():
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")
    filtered_data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])
    mixed_effects_model = MixedEffectsDepressionModel(filtered_data)
    mixed_effects_model.run_model()

if __name__ == "__main__":
    main()