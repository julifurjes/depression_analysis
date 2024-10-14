import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
import os

class LongitudinalLogisticRegression:
    def __init__(self, data, demographic_cols, subject_col, time_col):
        """
        Initialize the longitudinal logistic regression model with the dataset and demographic columns.

        :param data: pandas DataFrame containing the dataset.
        :param demographic_cols: list of strings representing demographic column names.
        :param subject_col: string representing the column name for the subject identifier (random effect).
        :param time_col: string representing the column name for the time variable (e.g., age).
        """
        self.data = data
        self.demographic_cols = demographic_cols
        self.subject_col = subject_col
        self.time_col = time_col
        if not os.path.exists('log_reg/output'):
            os.makedirs('log_reg/output')  # Ensure output directory exists

    def prepare_long_format(self):
        """
        Reshape the data into long format for longitudinal analysis.
        The long format will have one row per subject per time point.
        """
        # Filter columns for the ones I'll actually use
        self.data = self.data[['ID', 'SEXO', 'Estado_Civil', 'Escolaridade', 'NUTII', 'Score_Depressao_T0', 'Idade_T0', 'Score_Depressao_T1', 'Idade_T1', 'Score_Depressao_T3', 'Idade_T3']]
        # Reshape the data into long format, stacking the score columns (T0, T1, T3)
        long_data = pd.wide_to_long(
            self.data,
            stubnames=['Score_Depressao', 'Idade'],
            i=self.subject_col,
            j='Time',
            sep='_',
            suffix='\\w+'
        ).reset_index()

        return long_data
    
    def handle_missing_data(self, df):
        """
        Handle missing and infinite data by removing rows containing NaN or infinite values.
        """
        print(df)
        # Replace infinite values with NaN
        df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
        
        # Drop rows with any missing values
        df.dropna(inplace=True)
        
        return df

    def fit_gee(self, long_data, outcome_col, cov_structure=Exchangeable()):
        """
        Fit a GEE (Generalized Estimating Equations) model with logistic regression for longitudinal data.

        :param long_data: pandas DataFrame in long format.
        :param outcome_col: string representing the column name of the binary outcome variable (e.g., depression score).
        :param cov_structure: covariance structure to account for within-subject correlation (default: Exchangeable).
        """
        
        long_data = self.handle_missing_data(long_data)

        # Prepare the design matrix (X) with demographic and time variables
        X = long_data[self.demographic_cols + [self.time_col]]
        X = sm.add_constant(X)  # Add intercept

        # Prepare the outcome variable (y)
        y = long_data[outcome_col]

        # Define the GEE model with Binomial family (for logistic regression)
        model = GEE(y, X, groups=long_data[self.subject_col], family=Binomial(), cov_struct=cov_structure)
        result = model.fit()

        return result
    
    def save_results(self, result, long_data, outcome_col):
        """
        Save the GEE model summary, coefficients, and predicted results to CSV files.

        :param result: The result object from the fitted model.
        :param long_data: The long-format dataset used in the model.
        :param outcome_col: The column name of the target outcome.
        """
        # Save the model summary to a text file
        output_file = f'log_reg/output/gee_results_{outcome_col}.txt'
        with open(output_file, 'w') as f:
            f.write(result.summary().as_text())
        print(f"Results saved to {output_file}")
        
        # Save the model coefficients and standard errors to CSV
        coef_df = pd.DataFrame({
            'Coefficient': result.params,
            'Std_Err': result.bse,
            'Z_value': result.tvalues,
            'P_value': result.pvalues
        })
        coef_df.to_csv(f'log_reg/output/gee_coefficients_{outcome_col}.csv', index=True)
        print(f"Coefficients saved to 'log_reg/output/gee_coefficients_{outcome_col}.csv'")

        # Save actual vs predicted results to CSV
        y_pred = result.predict()  # Predicted values
        predictions_df = pd.DataFrame({
            'ID': long_data[self.subject_col],
            'Actual': long_data[outcome_col],
            'Predicted': y_pred
        })
        predictions_df.to_csv(f'log_reg/output/gee_predictions_{outcome_col}.csv', index=False)
        print(f"Predictions saved to 'log_reg/output/gee_predictions_{outcome_col}.csv'")

def main():
    # Read data from the "R" sheet of the Excel file "BD_Rute.xlsx"
    dataDepr1 = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    dataDepr = dataDepr1.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Select demographic columns (e.g., gender, marital status, education level)
    demographic_cols = ['SEXO', 'Estado_Civil', 'Escolaridade', 'NUTII']
    subject_col = 'ID'  # Subject identifier
    time_col = 'Idade'  # Age variable representing time

    # Instantiate the LongitudinalLogisticRegression class
    longitudinal_model = LongitudinalLogisticRegression(dataDepr, demographic_cols, subject_col, time_col)

    # Reshape the data into long format
    long_data = longitudinal_model.prepare_long_format()

    # Fit the GEE model (logistic regression for longitudinal data)
    outcome_col = 'Score_Depressao'  # Binary depression score
    result = longitudinal_model.fit_gee(long_data, outcome_col)

    # Save the model results, coefficients, and predictions
    longitudinal_model.save_results(result, long_data, outcome_col)

    # Print the summary of the model
    print(result.summary())

if __name__ == "__main__":
    main()