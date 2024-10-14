import pandas as pd

class DataDescription:
    def __init__(self, data, target_cols):
        """
        Initialize the DataDescription class.

        :param data: pandas DataFrame containing the dataset.
        :param target_cols: list of strings representing the column names of the depression scores at different time points.
        """
        self.data = data
        self.target_cols = target_cols

    def count_depression_ratio(self, target_col):
        """
        Calculate the ratio of depressed vs non-depressed individuals for a given target column.

        :param target_col: string representing the column name of the target depression score.
        :return: A dictionary with counts and ratio of depressed and non-depressed individuals for the specified column.
        """
        # Ensure the target column only contains valid binary values (0 for non-depressed, 1 for depressed)
        valid_data = self.data[target_col].dropna()

        # Count the occurrences of each class (0 and 1)
        depression_counts = valid_data.value_counts()

        non_depressed_count = depression_counts.get(0, 0)
        depressed_count = depression_counts.get(1, 0)

        total = non_depressed_count + depressed_count

        # Calculate the ratio
        depression_ratio = {
            'Non-Depressed Count': non_depressed_count,
            'Depressed Count': depressed_count,
            'Total': total,
            'Depressed Ratio': depressed_count / total if total > 0 else 0,
            'Non-Depressed Ratio': non_depressed_count / total if total > 0 else 0
        }

        return depression_ratio

    def print_depression_ratios(self):
        """
        Print the ratio of depressed vs non-depressed individuals for each depression score column.
        """
        for col in self.target_cols:
            print(f"\nDepression Data for {col}:")
            ratio = self.count_depression_ratio(col)
            print(f"Total Individuals: {ratio['Total']}")
            print(f"Non-Depressed Count: {ratio['Non-Depressed Count']} ({ratio['Non-Depressed Ratio']:.2%})")
            print(f"Depressed Count: {ratio['Depressed Count']} ({ratio['Depressed Ratio']:.2%})")

def main():
    # Read the data from the Excel file
    dataDepr1 = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where the depression scores are not missing
    dataDepr = dataDepr1.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Specify the target columns representing the binary depression scores at different time points
    target_cols = ['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']

    # Instantiate the DataDescription class
    data_description = DataDescription(dataDepr, target_cols)

    # Print the ratio of depressed vs non-depressed individuals for all depression columns
    data_description.print_depression_ratios()

if __name__ == "__main__":
    main()