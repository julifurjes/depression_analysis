import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataDescription:
    def __init__(self, data, target_cols, timestamp_cols, subject_col):
        """
        Initialize the DataDescription class.

        :param data: pandas DataFrame containing the dataset.
        :param target_cols: list of strings representing the column names of the depression scores at different time points.
        :param timestamp_col: string representing the column name of the timestamp.
        :param subject_col: string representing the column name of the subject identifier.
        """
        self.data = data
        self.target_cols = target_cols
        self.timestamp_cols = timestamp_cols
        self.subject_col = subject_col

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

    def plot_class_overlap(self, feature1, feature2, target_col):
        """
        Plot the overlap between depressed and non-depressed individuals based on two features.

        :param feature1: string representing the first feature to plot on the x-axis.
        :param feature2: string representing the second feature to plot on the y-axis.
        :param target_col: string representing the target depression score column.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x=feature1, y=feature2, hue=target_col, palette={0: "blue", 1: "red"})
        plt.title(f"Class Overlap: {feature1} vs {feature2} ({target_col})")
        plt.show()

    def plot_intervals(self):
        """
        Plot the interval between T0 and T3 for all subjects as a distribution of time intervals.
        """
        # Convert the timestamp columns to datetime
        for col in self.timestamp_cols:
            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')

        # Drop rows with missing T0 or T3 timestamps
        valid_data = self.data.dropna(subset=[self.timestamp_cols[0], self.timestamp_cols[2]])

        # Calculate the time intervals between T0 and T3 for each subject
        valid_data['interval_days'] = (valid_data[self.timestamp_cols[2]] - valid_data[self.timestamp_cols[0]]).dt.days

        # Plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_data['interval_days'], bins=50, kde=True, color='blue', alpha=0.6)
        plt.axvline(valid_data['interval_days'].mean(), color='red', linestyle='--', label=f'Mean: {valid_data["interval_days"].mean():.1f} days')
        plt.title('Distribution of Time Intervals Between T0 and T3')
        plt.xlabel('Days Between T0 and T3')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def plot_timestamp_distribution(self):
        """
        Plot the distribution of the timestamps (T0, T1, T3) over time.
        """
        # Convert the timestamp columns to datetime
        for col in self.timestamp_cols:
            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')

        plt.figure(figsize=(12, 6))

        # Plot each timestamp column as a rug plot (or scatter if preferred)
        for i, col in enumerate(self.timestamp_cols):
            sns.histplot(self.data[col], kde=False, bins=50, label=f'Distribution of {col}', alpha=0.5)

        plt.title('Distribution of T0, T1, and T3 Timestamps Over Time')
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def count_specific_condition(self):
        """
        Count how many individuals have 0 at T0 and T1, but 1 at T3.
        """
        # Filter based on the condition: 0 at T0, 0 at T1, and 1 at T3
        condition = (self.data[self.target_cols[0]] == 0) & \
                    (self.data[self.target_cols[1]] == 0) & \
                    (self.data[self.target_cols[2]] == 1)

        # Count the number of subjects meeting this condition
        count = self.data[condition][self.subject_col].nunique()

        print(f"Number of individuals with 0 at {self.target_cols[0]} and {self.target_cols[1]}, but 1 at {self.target_cols[2]}: {count}")
        return count

def main():
    # Read the data from the Excel file
    dataDepr1 = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where the depression scores are not missing
    dataDepr = dataDepr1.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Specify the target columns representing the binary depression scores at different time points
    target_cols = ['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']

    # Specify the timestamp and subject columns
    timestamp_cols = ['Data_Entrevista_T0', 'Data_Entrevista_T1', 'Data_Entrevista_T3']
    subject_col = 'ID'

    # Instantiate the DataDescription class
    data_description = DataDescription(dataDepr, target_cols, timestamp_cols, subject_col)

    # Print the ratio of depressed vs non-depressed individuals for all depression columns
    data_description.print_depression_ratios()

    # Plot class overlap for two demographic features
    data_description.plot_class_overlap('Idade_T0', 'Escolaridade', 'Score_Depressao_T0')

    # Plot intervals between T0 and T3
    data_description.plot_intervals()

    # Plot the distribution of timestamps
    data_description.plot_timestamp_distribution()

    # Count individuals with 0 at T0 and T1, but 1 at T3
    data_description.count_specific_condition()

if __name__ == "__main__":
    main()