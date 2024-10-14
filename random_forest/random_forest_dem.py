import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

class RandomForestDepressionModel:
    def __init__(self, data, demographic_cols, target_col):
        """
        Initialize the RandomForestDepressionModel class.

        :param data: pandas DataFrame containing the dataset.
        :param demographic_cols: list of strings representing demographic column names.
        :param target_col: string representing the column name of the target variable (binary depression score).
        """
        self.data = data
        self.demographic_cols = demographic_cols
        self.target_col = target_col
        if not os.path.exists('random_forest/output'):
            os.makedirs('random_forest/output')  # Ensure output directory exists

    def preprocess_data(self):
        """
        Preprocess the data by encoding categorical variables, converting datetime columns, and handling missing values.
        """
        # Drop rows with missing values in the target column
        self.data = self.data.dropna(subset=[self.target_col])

        # Convert any datetime columns to a numerical format (e.g., ordinal or year)
        for col in self.data.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                # Fill NaT with the median datetime (or use any other appropriate value)
                median_date = self.data[col].dropna().median()
                self.data[col] = self.data[col].fillna(median_date)
                self.data[col] = self.data[col].apply(lambda x: x.toordinal() if pd.notnull(x) else x)

        # One-hot encode categorical variables (e.g., SEXO, Estado_Civil)
        self.data = pd.get_dummies(self.data, columns=self.demographic_cols, drop_first=True)

        # Remove any non-numeric columns (those that contain text such as 'toalha para lavar as costas')
        self.data = self.data.select_dtypes(include=['float64', 'int64', 'uint8'])

        return self.data

    def fit_random_forest(self):
        """
        Fit a Random Forest model to predict depression scores using demographic variables.
        """
        # Preprocess the data
        self.data = self.preprocess_data()

        # Define the feature matrix (X) and target variable (y)
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and fit the Random Forest Classifier with class weights to handle imbalance
        rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

        # Save the evaluation results
        self.save_results(accuracy, report)

        return rf_model

    def save_results(self, accuracy, report):
        """
        Save the Random Forest evaluation results to a text file.

        :param accuracy: The accuracy of the Random Forest model.
        :param report: The classification report of the Random Forest model.
        """
        output_file = 'random_forest/output/random_forest_results.txt'
        with open(output_file, 'w') as f:
            f.write(f"Accuracy: {accuracy}\n\n")
            f.write(f"Classification Report:\n{report}\n")
        print(f"Results saved to {output_file}")

def main():
    # Read data from the "R" sheet of the Excel file "BD_Rute.xlsx"
    dataDepr1 = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    dataDepr = dataDepr1.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Select demographic columns (e.g., gender, marital status, education level)
    demographic_cols = ['SEXO', 'Estado_Civil', 'Escolaridade', 'NUTII']  # Replace with appropriate demographic columns
    target_col = 'Score_Depressao_T0'  # Target variable (depression score at baseline, T0)

    # Instantiate the RandomForestDepressionModel class
    random_forest_model = RandomForestDepressionModel(dataDepr, demographic_cols, target_col)

    # Fit the Random Forest model
    rf_model = random_forest_model.fit_random_forest()

if __name__ == "__main__":
    main()