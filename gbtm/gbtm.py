import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

class GBTMModel:
    def __init__(self, data):
        self.data = data
        self.subject_col = 'ID'
        self.score_cols = ['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']
        self.data_wide = self.prepare_wide_data()
        if not os.path.exists('gbtm/output'):
            os.makedirs('gbtm/output')  # Ensure output directory exists

    def prepare_wide_data(self):
        # Ensure the data has the necessary columns
        data = self.data.copy()
        required_cols = [self.subject_col] + self.score_cols
        data = data[required_cols].dropna()

        # Reshape the data so each row represents an individual with their scores over time
        data_wide = data.set_index(self.subject_col)
        return data_wide

    def determine_optimal_groups(self, max_groups=5):
        X = self.data_wide[self.score_cols].values
        bic_scores = []
        n_components_range = range(1, max_groups + 1)

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))

        # Plot BIC scores
        plt.figure(figsize=(8, 6))
        plt.plot(n_components_range, bic_scores, marker='o')
        plt.xlabel('Number of Trajectory Groups')
        plt.ylabel('BIC Score')
        plt.title('BIC Scores for Different Numbers of Trajectory Groups')
        plt.xticks(n_components_range)
        plt.tight_layout()
        plt.savefig('gbtm/output/bic_scores.png')
        plt.show()

        # Select the number of components with the lowest BIC
        optimal_groups = n_components_range[np.argmin(bic_scores)]
        print(f'Optimal number of trajectory groups: {optimal_groups}')
        return optimal_groups

    def fit_gbtm(self, n_groups):
        X = self.data_wide[self.score_cols].values
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit the GMM
        gmm = GaussianMixture(n_components=n_groups, covariance_type='full', random_state=42)
        gmm.fit(X_scaled)

        # Predict the trajectory group for each individual
        group_assignments = gmm.predict(X_scaled)
        self.data_wide['TrajectoryGroup'] = group_assignments

        # Save the group assignments
        self.data_wide['TrajectoryGroup'].to_csv('gbtm/output/trajectory_groups.csv')
        print('Trajectory group assignments saved to gbtm/output/trajectory_groups.csv')

        return gmm

    def plot_trajectories(self):
        # Melt the data for plotting
        data_melted = self.data_wide.reset_index().melt(
            id_vars=[self.subject_col, 'TrajectoryGroup'],
            value_vars=self.score_cols,
            var_name='Time',
            value_name='DepressionScore'
        )
        # Map time variable to numerical values
        time_mapping = {'Score_Depressao_T0': 0, 'Score_Depressao_T1': 1, 'Score_Depressao_T3': 3}
        data_melted['TimePoint'] = data_melted['Time'].map(time_mapping)

        # Plot the trajectories
        plt.figure(figsize=(10, 8))
        sns.lineplot(
            data=data_melted,
            x='TimePoint',
            y='DepressionScore',
            hue='TrajectoryGroup',
            estimator='mean',
            ci=None,
            palette='Set1'
        )
        plt.xlabel('Time Point')
        plt.ylabel('Mean Depression Score')
        plt.title('Trajectories of Depression Scores by Group')
        plt.legend(title='Trajectory Group', loc='best')
        plt.tight_layout()
        plt.savefig('gbtm/output/trajectory_plot.png')
        plt.show()

    def analyze_predictors(self, top_n=20):
        # Merge the group assignments back to the original data
        data = self.data.merge(
            self.data_wide[['TrajectoryGroup']],
            left_on='ID',
            right_index=True
        )

        # Choose predictors (exclude the depression scores and ID)
        predictors = data.columns.difference(
            [self.subject_col] + self.score_cols + ['TrajectoryGroup']
        )

        # Prepare data for modeling
        X = data[predictors]
        y = data['TrajectoryGroup']

        # Handle datetime columns (convert them to numerical or drop them)
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X[col + '_year'] = X[col].dt.year
            X[col + '_month'] = X[col].dt.month
            X[col + '_day'] = X[col].dt.day
        X = X.drop(columns=datetime_cols)

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Handle missing values
        X = X.fillna(0)

        # Standardize the predictors
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit a multinomial logistic regression with L1 regularization
        clf = LogisticRegression(
            penalty='l1',
            multi_class='multinomial',
            solver='saga',
            max_iter=1000,
            C=0.1  # Adjust C to control regularization strength
        )
        clf.fit(X_scaled, y)

        # Extract the coefficients
        coef_df = pd.DataFrame(clf.coef_, columns=X.columns)
        coef_df['TrajectoryGroup'] = clf.classes_
        coef_df = coef_df.set_index('TrajectoryGroup')

        # For each trajectory group, get the top N predictors
        top_predictors = {}
        for group in clf.classes_:
            coefs = coef_df.loc[group]
            # Select top N predictors based on absolute coefficient values
            top_features = coefs.abs().sort_values(ascending=False).head(top_n)
            top_predictors[group] = top_features.index.tolist()

        # Display and save the top predictors
        with open('gbtm/output/top_predictors.txt', 'w') as f:
            for group, predictors in top_predictors.items():
                print(f"\nTop {top_n} predictors for Trajectory Group {group}:")
                print(predictors)
                f.write(f"Top {top_n} predictors for Trajectory Group {group}:\n")
                for predictor in predictors:
                    coef_value = coef_df.loc[group, predictor]
                    print(f"{predictor}: Coefficient = {coef_value}")
                    f.write(f"{predictor}: Coefficient = {coef_value}\n")
                f.write("\n")
        print('Top predictors saved to gbtm/output/top_predictors.txt')

    def plot_top_coefficients(self, coef_df, top_n=20):
        for group in coef_df.index.unique():
            coefs = coef_df.loc[group]
            top_coefs = coefs.abs().sort_values(ascending=False).head(top_n)
            top_features = top_coefs.index
            coef_values = coefs[top_features]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=coef_values.values, y=top_features)
            plt.title(f'Top {top_n} Coefficients for Trajectory Group {group}')
            plt.xlabel('Coefficient Value')
            plt.tight_layout()
            plt.savefig(f'gbtm/output/top_coefficients_group_{group}.png')
            plt.show()

def main():
    # Load your data
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")

    # Filter data to include only rows where specified scores are not missing
    data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    # Instantiate the GBTM model
    gbtm_model = GBTMModel(data)

    # Determine the optimal number of trajectory groups
    optimal_groups = gbtm_model.determine_optimal_groups(max_groups=5)

    # Fit the GBTM
    gmm = gbtm_model.fit_gbtm(n_groups=optimal_groups)

    # Plot the trajectories
    gbtm_model.plot_trajectories()

    # Analyze associations with predictors
    gbtm_model.analyze_predictors(top_n=20)  # Adjust top_n as needed

    # Optionally, plot the top coefficients
    coef_df = pd.read_csv('gbtm/output/predictor_coefficients.csv', index_col='TrajectoryGroup')
    gbtm_model.plot_top_coefficients(coef_df, top_n=20)

if __name__ == "__main__":
    main()