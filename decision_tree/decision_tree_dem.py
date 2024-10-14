import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn import tree
import matplotlib.pyplot as plt
import os

class DecisionTreeDepressionModel:
    def __init__(self, data, demographic_cols, target_col):
        """
        Initialize the Decision Tree model with the dataset and demographic columns.

        :param data: pandas DataFrame containing the dataset.
        :param demographic_cols: list of strings representing demographic column names.
        :param target_col: string representing the column name of the target variable (binary depression score).
        """
        self.data = data
        self.demographic_cols = demographic_cols
        self.target_col = target_col
        if not os.path.exists('decision_tree/output'):
            os.makedirs('decision_tree/output')  # Ensure output directory exists

    def preprocess_data(self):
        """
        Preprocess the data by encoding categorical variables, handling missing values, and converting datetime columns.
        """
        # Filter columns for the ones I'll actually use
        self.data = self.data[['ID', 'SEXO', 'Estado_Civil', 'Escolaridade', 'NUTII', 'Score_Depressao_T0', 'Idade_T0', 'Score_Depressao_T1', 'Idade_T1', 'Score_Depressao_T3', 'Idade_T3']]
        # Drop rows with missing values in the selected columns
        self.data = self.data.dropna(subset=self.demographic_cols + [self.target_col])

        # Convert any datetime columns to numeric (e.g., ordinal or year)
        for col in self.demographic_cols:
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                # Convert datetime to ordinal (days since a reference date)
                self.data[col] = self.data[col].apply(lambda x: x.toordinal() if pd.notnull(x) else x)

        # One-hot encode categorical variables (e.g., SEXO, Estado_Civil)
        self.data = pd.get_dummies(self.data, columns=self.demographic_cols, drop_first=True)

        return self.data

    def fit_decision_tree(self):
        """
        Fit a Decision Tree model to predict depression scores using demographic variables.
        """
        # Preprocess the data
        self.data = self.preprocess_data()

        # Define the feature matrix (X) and target variable (y)
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]

        # Debug: Check for datetime columns in X
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                raise ValueError(f"Column {col} is still a datetime object!")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and fit the Decision Tree Classifier
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)  # Get report as a dictionary

        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Save the evaluation results
        self.save_results(accuracy, report, y_test, y_pred)

        return clf

    def save_results(self, accuracy, report, y_test, y_pred):
        """
        Save the Decision Tree evaluation results to a CSV file and save the predictions.

        :param accuracy: The accuracy of the Decision Tree model.
        :param report: The classification report as a dictionary.
        :param y_test: The actual labels.
        :param y_pred: The predicted labels.
        """
        # Save metrics (accuracy, precision, recall, f1-score) to a CSV
        metrics_df = pd.DataFrame(report).transpose()  # Convert the classification report to DataFrame
        metrics_df['accuracy'] = accuracy  # Add accuracy to all rows
        metrics_df.to_csv('decision_tree/output/decision_tree_metrics.csv', index=True)
        print(f"Metrics saved to 'decision_tree/output/decision_tree_metrics.csv'")

        # Save actual vs predicted results to CSV
        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        predictions_df.to_csv('decision_tree/output/decision_tree_predictions.csv', index=False)
        print(f"Predictions saved to 'decision_tree/output/decision_tree_predictions.csv'")

    def plot_tree(self, clf, feature_names):
        """
        Plot the Decision Tree and save it as an image file.

        :param clf: Trained Decision Tree Classifier.
        :param feature_names: List of feature names (columns used in the model).
        """
        plt.figure(figsize=(20, 10))
        tree.plot_tree(clf, feature_names=feature_names, filled=True, class_names=['No Depression', 'Depression'], rounded=True)

        # Save the tree plot
        plt.savefig('decision_tree/output/decision_tree_plot.png')
        print("Decision Tree plot saved to 'decision_tree/output/decision_tree_plot.png'")
        plt.show()

def main():
    # Read data from the "R" sheet of the Excel file "BD_Rute.xlsx"
    dataDepr1 = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    dataDepr = dataDepr1.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Select demographic columns
    demographic_cols = ['SEXO', 'Estado_Civil', 'Escolaridade', 'NUTII']  # Demographic predictors
    target_col = 'Score_Depressao_T0'  # Target variable (depression score at baseline, T0)

    # Instantiate the DecisionTreeDepressionModel class
    decision_tree_model = DecisionTreeDepressionModel(dataDepr, demographic_cols, target_col)

    # Fit the Decision Tree model
    clf = decision_tree_model.fit_decision_tree()

    # Plot the Decision Tree
    decision_tree_model.plot_tree(clf, feature_names=decision_tree_model.data.columns.drop(target_col))

if __name__ == "__main__":
    main()