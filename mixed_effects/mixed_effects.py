import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectKBest, chi2
from pymer4.models import Lmer
import os
import numpy as np
import sys
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import DataPreparation

class MixedEffectsDepressionModel:
    def __init__(self, data):
        """
        Initialize the mixed-effects model with the dataset and subject identifier.
        :param data: pandas DataFrame containing the dataset.
        :param subject_col: string representing the column name for the subject identifier (random effect).
        """
        self.data = data
        self.subject_col = 'ID'
        self.data_long = DataPreparation().handle_data(self.data)
        if not os.path.exists('mixed_effects/output'):
            os.makedirs('mixed_effects/output')  # Ensure output directory exists

    def run_model(self):
        """
        Run the mixed-effects model for a specific score and time column.
        """
        print(f"Running Mixed-Effects Model")

        # Prepare the formula
        exclude_cols = ['ID', 'Time', 'DepressionStatus', 'DepressionScore']
        fixed_effects = [col for col in self.data_long.columns if col not in exclude_cols]

        selected_features = DataPreparation().variable_screening(fixed_effects, self.data_long)[0]

        self.data_long['ID'] = self.data_long['ID'].astype('category')
        print(self.data_long['ID'].dtype)

        # Prepare formula with selected features
        formula = 'DepressionStatus ~ ' + ' + '.join(selected_features) + ' + (1|ID)'

        # Fit the model using pymer4
        model = Lmer(formula, data=self.data_long, family='binomial')
        result = model.fit()
        
        # Print the result
        print(result)  # Since result is a DataFrame, print it directly

        # Optionally, save results
        self.save_results(result)

    def save_results(self, result):
        """
        Save the mixed-effects model summary and coefficients to files.

        :param result: The result DataFrame from the fitted model.
        """
        # Save the model result to a text file
        output_file = f'mixed_effects/output/mixed_effects_results.txt'
        with open(output_file, 'w') as f:
            f.write(result.to_string())  # Use to_string() to save the DataFrame content as text
        print(f"Results saved to {output_file}")

        # Save the coefficients and standard errors to a CSV file
        coef_df = result  # The result from pymer4 is a DataFrame, so save it directly
        coef_df.to_csv(f'mixed_effects/output/mixed_effects_coefficients.csv', index=False)
        print(f"Coefficients saved to 'mixed_effects/output/mixed_effects_coefficients.csv'")

def main():
    # Read data from the "R" sheet of the Excel file "BD_Rute.xlsx"
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    filtered_data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Instantiate the DepressionModel class
    mixed_effects_model = MixedEffectsDepressionModel(filtered_data)

    # Run model with class weighting
    mixed_effects_model.run_model()

if __name__ == "__main__":
    main()