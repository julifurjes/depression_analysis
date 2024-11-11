import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataPreparation

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
        vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                           for i in range(X.shape[1])]
        return vif_data

    def run_gee_model(self):
        """
        Run the GEE model and evaluate predictive performance.
        """
        print(f"Running GEE Model")

        # Exclude outcome-related and problematic variables
        exclude_cols = ['ID', 'Time', 'DepressionStatus', 'DepressionScore', 'Pontuacao_Depressao'] + \
                    [col for col in self.data_long.columns if 'Depress' in col or 'Score' in col]

        # Prepare the full dataset X and y
        fixed_effects = [col for col in self.data_long.columns if col not in exclude_cols]

        # Preprocess the training data
        selected_vars, X, y = DataPreparation().variable_screening(fixed_effects, self.data_long)

        # Include the 'ID' column for grouping
        X_with_ID = X.copy()
        X_with_ID['ID'] = self.data_long['ID'].reset_index(drop=True)

        # Use GroupShuffleSplit to split the data based on 'ID'
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_inds, test_inds = next(gss.split(X_with_ID, y, groups=X_with_ID['ID']))

        # Split the data
        X_train_full = X_with_ID.iloc[train_inds].reset_index(drop=True)
        X_test_full = X_with_ID.iloc[test_inds].reset_index(drop=True)
        y_train = y.iloc[train_inds].reset_index(drop=True)
        y_test = y.iloc[test_inds].reset_index(drop=True)

        # Remove 'ID' from predictors after splitting (if necessary)
        X_train_full = X_train_full.drop(columns=['ID'])
        X_test_full = X_test_full.drop(columns=['ID'])

        # Remove multicollinear variables from training data
        vif_data = self.calculate_vif(X_train_full[selected_vars])
        high_vif = vif_data[vif_data['VIF'] > 5]['Feature']
        selected_vars = [var for var in selected_vars if var not in high_vif]
        X_train = X_train_full[selected_vars]

        # Prepare the testing data using the same selected variables
        X_test = X_test_full[selected_vars]

        # Standardize the testing data using training data parameters
        X_test = (X_test_full - X_train_full.mean()) / X_train_full.std()

        # Add the intercept term
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        # Specify the groups (subjects) in training and testing data
        groups_train = self.data_long.loc[X_train_full.index, self.subject_col].reset_index(drop=True)
        groups_test = self.data_long.loc[X_test_full.index, self.subject_col].reset_index(drop=True)

        # Define the correlation structure
        ind = sm.cov_struct.Independence()

        # Define the GEE model using training data
        model = sm.GEE(y_train, X_train, groups=groups_train, family=sm.families.Binomial(), cov_struct=ind)

        # Fit the model
        result = model.fit()

        # Print the summary
        print(result.summary())

        # Call the method to print and plot significant coefficients
        self.print_and_plot_significant_coefficients(result)

        # Save results
        self.save_gee_results(result)

        # Evaluate predictive performance on test data
        self.evaluate_predictive_performance(result, X_test, y_test)

    def print_and_plot_significant_coefficients(self, result):
        """
        Print and plot statistically significant coefficients.
        """
        # Get the model's coefficients, p-values, and standard errors
        params = result.params
        pvalues = result.pvalues
        bse = result.bse

        # Define a significance threshold (e.g., 0.05)
        significance_threshold = 0.05

        # Filter statistically significant coefficients (p-value < 0.05)
        significant_coefs = params[pvalues < significance_threshold]
        significant_pvalues = pvalues[pvalues < significance_threshold]
        significant_bse = bse[pvalues < significance_threshold]

        # Print the significant coefficients
        if not significant_coefs.empty:
            print("\nStatistically Significant Coefficients (p-value < 0.05):")
            for feature, coef, pval in zip(significant_coefs.index, significant_coefs.values, significant_pvalues.values):
                print(f"{feature}: Coefficient = {coef}, p-value = {pval}")
        else:
            print("\nNo statistically significant coefficients found.")

        # Plot the significant coefficients
        if not significant_coefs.empty:
            plt.figure(figsize=(8, 6))
            plt.barh(significant_coefs.index, significant_coefs.values, xerr=significant_bse.values, color='skyblue')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Features')
            plt.title('Statistically Significant Coefficients (p-value < 0.05)')
            plt.tight_layout()
            plt.savefig('gee/output/significant_coefficients.png')
            plt.show()
            print("Significant coefficients plot saved to 'gee/output/significant_coefficients.png'")

    def save_gee_results(self, result):
        """
        Save the GEE model summary and coefficients to files.
        """
        # Save the model summary to a text file
        output_file = f'gee/output/gee_model_results.txt'
        with open(output_file, 'w') as f:
            f.write(str(result.summary()))
        print(f"GEE Results saved to {output_file}")

        # Save the coefficients and standard errors to a CSV file
        params = result.params
        bse = result.bse
        pvalues = result.pvalues
        coef_df = pd.DataFrame({
            'Coefficient': params,
            'Std_Err': bse,
            'P_value': pvalues
        })
        coef_df.to_csv(f'gee/output/gee_model_coefficients.csv', index=True)
        print(f"GEE Coefficients saved to 'gee/output/gee_model_coefficients.csv'")

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

        # Save evaluation metrics to a text file
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
    # Read data from the "R" sheet of the Excel file "BD_Rute.xlsx"
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    filtered_data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Instantiate the GeeModel class
    gee_model = GeeModel(filtered_data)

    # Run the GEE model
    gee_model.run_gee_model()

if __name__ == "__main__":
    main()