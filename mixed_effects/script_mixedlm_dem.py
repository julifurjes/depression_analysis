import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import os

class MixedEffectsDepressionModel:
    def __init__(self, data, demographic_cols, subject_col):
        """
        Initialize the mixed-effects model with the dataset and demographic columns.

        :param data: pandas DataFrame containing the dataset.
        :param demographic_cols: list of strings representing demographic column names.
        :param subject_col: string representing the column name for the subject identifier (random effect).
        """
        self.data = data
        self.demographic_cols = demographic_cols
        self.subject_col = subject_col
        if not os.path.exists('mixed_effects/output'):
            os.makedirs('mixed_effects/output')  # Ensure output directory exists

    def run_model(self, score_column, time_column, model_type='score'):
        """
        Run the mixed-effects model for a specific score and time column.

        :param score_column: string representing the column name of the target score.
        :param time_column: string representing the column name of the time (age) feature.
        :param model_type: string indicating whether it's a 'score' model.
        """
        print(f"Running Mixed-Effects Model for {score_column} with {time_column}")

        # Prepare the feature set (demographics + specific time/age column)
        X = self.data[self.demographic_cols + [time_column]].copy()

        # Check for and remove rows with missing or infinite values in X
        X.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
        X.dropna(inplace=True)

        # Ensure target variable (y) and random effects are aligned with X after dropping rows
        y = self.data.loc[X.index, score_column]
        random_effects = self.data.loc[X.index, self.subject_col]

        # Ensure indices of X, y, and random_effects are aligned
        assert len(X) == len(y) == len(random_effects), "Mismatched indices in X, y, and random effects"

        # Add a constant for the intercept
        X = sm.add_constant(X)

        # Fit the mixed-effects model
        model = MixedLM(y, X, groups=random_effects)
        result = model.fit()

        # Output the results
        self.save_results(result, score_column, model_type)
        self.save_predictions(result, X, y, score_column)
        print(result.summary())

    def save_results(self, result, score_column, model_type):
        """
        Save the mixed-effects model summary and coefficients to files.

        :param result: The result object from the fitted model.
        :param score_column: The column name of the target score.
        :param model_type: The type of model (score or difference).
        """
        # Save the model summary to a text file
        output_file = f'mixed_effects/output/mixed_effects_results_{model_type}_{score_column}.txt'
        with open(output_file, 'w') as f:
            f.write(result.summary().as_text())
        print(f"Results saved to {output_file}")

        # Save the coefficients and standard errors to a CSV file
        coef_df = pd.DataFrame({
            'Coefficient': result.params,
            'Std_Err': result.bse,
            'Z_value': result.tvalues,
            'P_value': result.pvalues
        })
        coef_df.to_csv(f'mixed_effects/output/mixed_effects_coefficients_{model_type}_{score_column}.csv', index=True)
        print(f"Coefficients saved to 'mixed_effects/output/mixed_effects_coefficients_{model_type}_{score_column}.csv'")

    def save_predictions(self, result, X, y, score_column):
        """
        Save the actual vs predicted results to a CSV file.

        :param result: The result object from the fitted model.
        :param X: The feature matrix used for predictions.
        :param y: The actual target values.
        :param score_column: The column name of the target score.
        """
        # Predict the outcomes
        y_pred = result.predict(X)

        # Save actual vs predicted results to CSV
        predictions_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        })
        predictions_df.to_csv(f'mixed_effects/output/mixed_effects_predictions_{score_column}.csv', index=False)
        print(f"Predictions saved to 'mixed_effects/output/mixed_effects_predictions_{score_column}.csv'")

def main():
    # Read data from the "R" sheet of the Excel file "BD_Rute.xlsx"
    dataDepr1 = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    dataDepr = dataDepr1.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Select demographic columns (including subject ID to track within-subject variability)
    demographic_cols = ['SEXO', 'Estado_Civil', 'Escolaridade', 'NUTII']  # Add demographic variables
    subject_col = 'ID'  # Specify the subject identifier

    # Define the score and age columns for each time point
    score_age_pairs = [
        ('Score_Depressao_T0', 'Idade_T0'),
        ('Score_Depressao_T1', 'Idade_T1'),
        ('Score_Depressao_T3', 'Idade_T3')
    ]

    # Instantiate the MixedEffectsDepressionModel class
    mixed_effects_model = MixedEffectsDepressionModel(dataDepr, demographic_cols, subject_col)

    # Loop over each score and age pair and run the mixed-effects model
    for score_column, age_column in score_age_pairs:
        mixed_effects_model.run_model(score_column, age_column)

# Ensure the script runs only when executed directly (not when imported)
if __name__ == "__main__":
    main()