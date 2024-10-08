# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

class XGBoostDepressionModel:
    def __init__(self, data, demographic_cols):
        """
        Initialize the model with the dataset and demographic columns.

        :param data: pandas DataFrame containing the dataset.
        :param demographic_cols: list of strings representing demographic column names.
        """
        self.data = data
        self.demographic_cols = demographic_cols
        if not os.path.exists('xgboost/output'):
            os.makedirs('xgboost/output')  # Ensure output directory exists

    def run_model(self, target_column, age_column, model_type='difference'):
        """
        Run the XGBoost model for a specific target (either a score or score difference) and age column.

        :param target_column: string representing the column name of the target (score or score difference).
        :param age_column: string representing the column name of the age.
        :param model_type: string indicating whether it's a 'score' or 'difference' model.
        """
        print(f"Running XGBoost for {target_column} with {age_column}")

        # Prepare the feature set (demographics + specific age column)
        X = self.data[self.demographic_cols + [age_column]]

        # Preprocess categorical variables (one-hot encoding)
        X = pd.get_dummies(X, drop_first=True)

        # Target variable is the score or score difference (binary: 0 or 1)
        y = self.data[target_column]

        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define and train the XGBoost classifier (since it's a binary classification problem)
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)

        # Make predictions
        y_pred = xgb_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {target_column}: {accuracy}")
        print(f"Classification Report for {target_column}:\n", classification_report(y_test, y_pred))

        # Plot feature importance
        self.plot_feature_importance(xgb_model, target_column, model_type)

    def plot_feature_importance(self, model, target_column, model_type):
        """
        Plot and save the feature importance graph for the given model.

        :param model: XGBoost model object
        :param target_column: string representing the column name of the target score or difference.
        :param model_type: string indicating whether it's a 'score' or 'difference' model.
        """
        xgb.plot_importance(model, importance_type='weight', max_num_features=10)
        plt.title(f'Feature Importance for {target_column} ({model_type} model)')
        plt.savefig(f'xgboost/output/feature_importance_{model_type}_{target_column}.png')
        plt.show()


def create_score_differences(data):
    """
    Create columns representing changes in the depression scores between time points.
    This will be used as target variables to explore within-subject changes.
    
    :param data: pandas DataFrame
    :return: pandas DataFrame with new columns representing differences in scores.
    """
    # Calculate differences in scores between consecutive time points
    data['Score_Diff_T0_T1'] = data['Score_Depressao_T1'] - data['Score_Depressao_T0']
    data['Score_Diff_T1_T3'] = data['Score_Depressao_T3'] - data['Score_Depressao_T1']

    # Ensure binary values (convert -1 to 0 where depression improves)
    data['Score_Diff_T0_T1'] = data['Score_Diff_T0_T1'].apply(lambda x: 1 if x == 1 else 0)
    data['Score_Diff_T1_T3'] = data['Score_Diff_T1_T3'].apply(lambda x: 1 if x == 1 else 0)

    return data

def main():
    # Read data from the "R" sheet of the Excel file "BD_Rute.xlsx"
    dataDepr1 = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    dataDepr = dataDepr1.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Create columns representing changes in scores
    dataDepr = create_score_differences(dataDepr)

    # Select demographic columns (including subject ID to track within-subject variability)
    demographic_cols = ['SEXO', 'Estado_Civil', 'Escolaridade', 'NUTII', 'ID']  # Add subject ID as well

    # Define the score and age columns for each time point (for individual scores and differences)
    score_age_pairs = [
        # Individual Scores
        ('Score_Depressao_T0', 'Idade_T0', 'score'),
        ('Score_Depressao_T1', 'Idade_T1', 'score'),
        ('Score_Depressao_T3', 'Idade_T3', 'score'),

        # Score Differences
        ('Score_Diff_T0_T1', 'Idade_T1', 'difference'),
        ('Score_Diff_T1_T3', 'Idade_T3', 'difference')
    ]

    # Instantiate the XGBoostDepressionModel class
    xgboost_model = XGBoostDepressionModel(dataDepr, demographic_cols)

    # Loop over each score/difference and age pair and run the XGBoost model
    for target_column, age_column, model_type in score_age_pairs:
        xgboost_model.run_model(target_column, age_column, model_type)

# Ensure the script runs only when executed directly (not when imported)
if __name__ == "__main__":
    main()