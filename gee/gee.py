import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectKBest, chi2
import os
import numpy as np
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
        Run the GEE model for the specified variables.
        """
        print(f"Running GEE Model")

        # Exclude outcome-related and problematic variables
        exclude_cols = ['ID', 'Time', 'DepressionStatus', 'DepressionScore', 'Pontuacao_Depressao'] + \
                    [col for col in self.data_long.columns if 'Depress' in col or 'Score' in col]

        fixed_effects = [col for col in self.data_long.columns if col not in exclude_cols]

        selected_vars, X, y = DataPreparation().variable_screening(fixed_effects, self.data_long)

        # Remove multicollinear variables
        vif_data =self.calculate_vif(X[selected_vars])
        high_vif = vif_data[vif_data['VIF'] > 5]['Feature']
        selected_vars = [var for var in selected_vars if var not in high_vif]

        # Prepare the design matrix
        X_gee = X[selected_vars]
        y_gee = y

        # Add the intercept term
        X_gee = sm.add_constant(X_gee)

        # Specify the groups (subjects)
        groups = self.data_long[self.subject_col]

        # Define the correlation structure
        ind = sm.cov_struct.Independence()

        # Define the GEE model
        model = sm.GEE(y_gee, X_gee, groups=groups, family=sm.families.Binomial(), cov_struct=ind)

        # Fit the model
        result = model.fit()

        # Print the summary
        print(result.summary())

        # Call the new method to print and plot significant coefficients
        self.print_and_plot_significant_coefficients(result)

        # Save results
        self.save_gee_results(result)

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
            plt.show()

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

def main():
    # Read data from the "R" sheet of the Excel file "BD_Rute.xlsx"
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    filtered_data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Instantiate the DepressionModel class
    gee_model = GeeModel(filtered_data)

    # Run the GEE model
    gee_model.run_gee_model()

if __name__ == "__main__":
    main()