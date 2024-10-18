from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTENC
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreparation():

    def __init__(self) -> None:
        pass

    def _check_distribution(self, X_original, X_resampled):
        """
        Check the distribution of features before and after SMOTETomek resampling.
        """

        num_features = len(X_original.columns)

        # Determine group size based on the number of features and number of groups
        num_groups = math.ceil(num_features / 6)
        group_size = (num_features + num_groups - 1) // num_groups  # Split evenly across groups

        # Split the features into `num_groups`
        feature_groups = [X_original.columns[i:i + group_size] for i in range(0, num_features, group_size)]

        for group_idx, feature_group in enumerate(feature_groups):
            num_group_features = len(feature_group)
            num_rows = (num_group_features + 1) // 2

            fig, axes = plt.subplots(num_rows, 2, figsize=(10, num_rows * 3))
            axes = axes.ravel()  # Flatten axes array to easily iterate over

            for idx, feature in enumerate(feature_group):
                sns.histplot(X_original[feature], color="blue", kde=True, label="Original", stat="density", linewidth=0, ax=axes[idx])
                sns.histplot(X_resampled[feature], color="orange", kde=True, label="SMOTETomek", stat="density", linewidth=0, ax=axes[idx])
                axes[idx].set_title(f"Distribution of {feature} Before and After SMOTETomek")
                axes[idx].legend()
                axes[idx].set_xlabel('')

            # Hide any unused subplots if there are fewer features than subplots
            for i in range(idx + 1, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()
            plt.show()

    def _check_pca_boundaries(self, X_resampled, y_resampled):
        """
        Perform PCA and plot boundaries before and after resampling to check decision boundaries.
        """
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_resampled)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_resampled, cmap='coolwarm', edgecolor='k', s=50)
        plt.title('PCA of Resampled Data')
        plt.show()

    def generate_smote_tomek(self):
        categorical_features = ['SEXO', 'Estado_Civil', 'Escolaridade', 'NUTII']
        print(self.data.columns)
        categorical_indices = [self.data.columns.get_loc(col) for col in categorical_features]  # Get column indices

        # Define the feature matrix (X) and target variable (y)
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')  # Impute missing values using the mean
        X = imputer.fit_transform(X)

        # Convert X back to a DataFrame with the original column names after imputation
        X = pd.DataFrame(X, columns=self.data.drop(columns=[self.target_col]).columns)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Save original data for performance checks
        X_train_original, y_train_original = X_train.copy(), y_train.copy()
        X_test_original, y_test_original = X_test.copy(), y_test.copy()

        # Initialize SMOTE-NC and Tomek Links
        smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
        tomek = TomekLinks()

        # **Resample Training Data**

        # Step 1: Apply SMOTE-NC (oversampling) to training data
        X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)

        # Step 2: Apply Tomek Links (undersampling) to training data
        X_train_resampled, y_train_resampled = tomek.fit_resample(X_train_resampled, y_train_resampled)

        # **Resample Testing Data**

        # Step 1: Apply SMOTE-NC (oversampling) to testing data
        X_test_resampled, y_test_resampled = smote_nc.fit_resample(X_test, y_test)

        # Step 2: Apply Tomek Links (undersampling) to testing data
        X_test_resampled, y_test_resampled = tomek.fit_resample(X_test_resampled, y_test_resampled)

        print(f"Original training class distribution: {Counter(y_train)}")
        print(f"Resampled training class distribution: {Counter(y_train_resampled)}")
        print(f"Original test class distribution: {Counter(y_test)}")
        print(f"Resampled test class distribution: {Counter(y_test_resampled)}")

        # Check feature distribution before and after SMOTETomek on training data
        self._check_distribution(X_train_original, X_train_resampled)

        # Check feature distribution before and after SMOTETomek on testing data
        self._check_distribution(X_test_original, X_test_resampled)

        # Check decision boundary via PCA on training data
        self._check_pca_boundaries(X_train_resampled, y_train_resampled)

        # Check decision boundary via PCA on testing data
        self._check_pca_boundaries(X_test_resampled, y_test_resampled)

        return X_train_resampled, y_train_resampled, X_test_resampled, y_test_resampled
    
    def _normalize_mixed_columns(self, df):
        # Normalize mixed-type columns (fill NaN and convert to string)
        mixed_columns = ['Outras_Pub_desc', 'Outra_Doenca_Cardiaca', 'haq_desc_outros', 'Outra_Doenca_Aparelho_Digestivo', 
                         'Outras_Priv_desc', 'Desc_Outro_Prest_Assist_Dom', 'Outros_Meds', 'Descricao_Outro_Local', 
                         'Outra_Doenca_Pulmonar', 'Desc', 'Outra_Doenca_Mental', 'Outra_Doenca_Neurologica', 'Outro_desc', 
                         'Outro_Exerc_desc', 'Outra_desc']
        for col in mixed_columns:
            if col in df.columns:
                # Convert to string and replace NaN with empty strings
                df.loc[:, col] = df[col].fillna('').astype(str)
        return df

    def handle_data(self, data):
        # Normalize mixed-type columns
        data = self._normalize_mixed_columns(data)

        # Fill missing values with a placeholder (if necessary)
        data['ID'] = data['ID'].fillna('Missing')

        # If you want to ensure ID is a string, convert it to string first
        data['ID'] = data['ID'].astype(str)

        # Now, convert ID to categorical
        data['ID'] = data['ID'].astype('category')

        score_cols = ['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']

        # Add an extra column where all 3 scores are present
        data['aux'] = (
            data['Score_Depressao_T0'].astype(str) +
            data['Score_Depressao_T1'].astype(str) +
            data['Score_Depressao_T3'].astype(str)
        )

        # Melt the depression scores to long format
        data_long = pd.melt(data, id_vars=['ID'], value_vars=score_cols,
                            var_name='Time', value_name='DepressionScore')

        # Extract time point (e.g., 'T0', 'T1', 'T3') from the variable names
        data_long['Time'] = data_long['Time'].str.extract(r'T(\d+)').astype(int)

        # Merge the repeated measures for other variables
        repeated_vars = [col[:-3] for col in data.columns if col.endswith(('_T0', '_T1', '_T3'))]
        repeated_vars = list(set(repeated_vars))

        # Reshape repeated variables
        for var in repeated_vars:
            time_cols = [f"{var}_T0", f"{var}_T1", f"{var}_T3"]
            available_time_cols = [col for col in time_cols if col in data.columns]
            if not available_time_cols:
                continue
            melted = pd.melt(data[['ID'] + available_time_cols], id_vars=['ID'], value_vars=available_time_cols,
                             var_name='TimeVar', value_name=var)
            melted['Time'] = melted['TimeVar'].str.extract(r'T(\d+)').astype(int)
            melted = melted.drop('TimeVar', axis=1)
            data_long = pd.merge(data_long, melted, on=['ID', 'Time'], how='left')

        # Convert the depression score to binary (if not already binary)
        threshold = 0  # Where depression score > 0 indicates depression
        data_long['DepressionStatus'] = (data_long['DepressionScore'] > threshold).astype(int)

        return data_long
    
    
    def _standardize_variables(self, X):
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        return X

    def variable_screening(self, fixed_effects, df):
        X = df[fixed_effects]
        y = df['DepressionStatus']

        # Convert categorical variables to numeric codes
        for col in X.select_dtypes(['category', 'object']).columns:
            X.loc[:, col] = X[col].astype('category').cat.codes

        # Convert datetime or timestamp columns to numeric by extracting components
        for col in X.select_dtypes(include=[np.datetime64]).columns:
            X.loc[:, col + '_year'] = X[col].dt.year
            X.loc[:, col + '_month'] = X[col].dt.month
            X.loc[:, col + '_day'] = X[col].dt.day
            X = X.drop(col, axis=1)  # Drop original timestamp column after extraction

        # Fill missing values and handle future warning for object dtype arrays
        X = X.fillna(0).infer_objects(copy=False)

        # Remove description columns, as they are not useful for quantitative analysis
        X = X.loc[:, ~X.columns.str.contains('desc', case=False)]

        # Turn 'other' columns into binary (if there is any or not) for the purpose of the analysis
        outra_columns = X.columns[X.columns.str.contains('outra', case=False)]
        for col in outra_columns:
            X.loc[:, col] = X[col].notna().astype(int)

        # Turn 'medicine descriptions' columns into binary (if there is any or not) for the purpose of the analysis
        outros_columns = X.columns[X.columns.str.contains('outros', case=False)]
        for col in outros_columns:
            X.loc[:, col] = X[col].notna().astype(int)

        # Transform score_eq5d and score_eq5d_T2 to non-negative by adding 1
        if 'score_eq5d' in X.columns:
            X.loc[:, 'score_eq5d'] = X['score_eq5d'] + 1
        if 'score_eq5d_T2' in X.columns:
            X.loc[:, 'score_eq5d_T2'] = X['score_eq5d_T2'] + 1

        # Standardize variables
        X = self._standardize_variables(X)

        # Use mutual information for feature selection
        selector = SelectKBest(mutual_info_classif, k=20)
        selector.fit(X, y)
        selected_vars = X.columns[selector.get_support()]
        
        return selected_vars, X, y