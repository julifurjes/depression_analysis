from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, 
                             precision_score, recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.pipeline import Pipeline
import seaborn as sns

class RandomForestLongitudinalModel:
    def __init__(self, data, dataset_label):
        self.data = data
        self.subject_col = 'ID'
        self.dataset_label = dataset_label
        if not os.path.exists('random_forest_balanced/output'):
            os.makedirs('random_forest_balanced/output')  # Ensure output directory exists
        if not os.path.exists('random_forest_balanced/variable_plots'):
            os.makedirs('random_forest_balanced/variable_plots')  # Ensure variable_plots directory exists
        if not os.path.exists('random_forest_balanced/variable_tables'):
            os.makedirs('random_forest_balanced/variable_tables')  # Ensure variable_tables directory exists
        if not os.path.exists('random_forest_balanced/frequency_tables'):
            os.makedirs('random_forest_balanced/frequency_tables')  # Ensure frequency_tables directory exists

    def generate_frequency_tables(self, df, group_col='DepressivePathGroup'):
        # Socio-demographic columns
        socio_demo_cols = ["Idade_T0", "Idade_T1", "Idade_T2", "Idade_T3", "Escolaridade", "Sit_Prof_T0", "Sit_Prof_T1", "Sit_Prof_T2"]
        frequency_tables = {}

        for col in socio_demo_cols:
            table = df.groupby(group_col)[col].value_counts(normalize=True).unstack(fill_value=0)
            frequency_tables[col] = table

        return frequency_tables

    def calculate_vif(self, X):
        X = X.assign(constant=1)  # Add constant term for VIF calculation
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    
    def define_chosen_variables(self, df):
        # Ageing
        ageing = ["Idade_T0", "Idade_T1", "Idade_T2", "Idade_T3"]

        # Multimorbidity
        chronic_conditions = {
            "T0": ["Hipertensao_T0", "Diabetes_T0", "Colesterol_T0", "Doenca_Pulmonar_T0", "Doenca_Cardiaca_T0", "Digestiva_T0", "Neurologica_T0", "Alergia_T0", "Acido_urico_T0"],
            "T1": ["Hipertensao_T1", "Diabetes_T1", "Colesterol_T1", "Neurologica_T1", "Alergias_T1"],
            "T2": ["Hipertensao_T2", "Diabetes_T2", "Colesterol_T2", "Doenca_Cardiaca_T2", "Doenca_Digestiva_T2", "Doenca_Neurologica_T2", "Alergias_T2"],
            "T3": ["Hipertensao_T3", "Diabetes_T3", "Colesterol_T3", "Doenca_Pulmonar_T3", "Doenca_Cardiaca_T3", "Doenca_Aparelho_Digestivo_T3", "Doenca_Neurologica_T2", "Alergias_T3", "Acido_Urico_Elevado_T3"]
        }

        # Calculate Multimorbidity for each wave
        for wave, variables in chronic_conditions.items():
            df[f"Multimorbidity_{wave}"] = df[variables].sum(axis=1, skipna=True)

        multimorbidity = ['Multimorbidity_T0', 'Multimorbidity_T1', 'Multimorbidity_T2', 'Multimorbidity_T3']

        # Chronic Diseases
        rheumatic_diseases = ["AR_T0", "EA_T0", "AP_T0", "OA_T0", "OP_T0", "Gota_T0", "Polimialgia_T0", 
            "Les_T0", "Fibromialgia_T0", "AR_T1", "EA_T1", "AP_T1", 
            "Gota_T1", "Polimialgia_T1", "LES_T1", "Fibromialgia_T1", 
            "Lombalgia_T1", "AR_T2", "EA_T2", "AP_T2", "OP_T2", "Gota_T2", 
            "Polimialgia_T2", "LES_T2", "Fibromialgia_T2", "Lombalgia_T2", "AR_T3", 
            "EA_T3", "AP_T3", "OP_T3", "Gota_T3", "Polimialgia_T3", "LES_T3", "Fibromialgia_T3"]

        # Lifestyles
        smoking_habits = ['Cod_Fumador_T0', 'Cod_Fumador_T1', 'Cod_Fumador_T2', 'Cod_Fumador_T3']
        alcohol_habits = ['Cod_Alcool_T0', 'Cod_Alcool_T1', 'Cod_Alcool_T2', 'Cod_Alcool_T3']
        exercise_freq = ['Exercicio_Regular_T0', 'Exercicio_Regular_T1', 'Exercicio_Regular_T2', 'Exercicio_Regular_T3']
        sleep_hours = ['Num_Horas_Dorme_Por_Dia_T1', 'Num_Horas_Dorme_Por_Dia_T2', 'Num_Horas_Dorme_Por_Dia_T3']
        dietary_behaviour = ['DM3_T2', 'DM4_T2', 'DM5_T2', 'DM10_T2', 'DM6_T2']

        # SES
        education = ['Escolaridade']
        employment = ['Sit_Prof_T0', 'Sit_Prof_T1', 'Sit_Prof_T2']

        # Social support
        household_members = ['Num_Pessoas_Agregado_T0', 'Num_Pessoas_Agregado_T2', 'Num_Pessoas_Agregado_T3']
        assistance = ['Assist_Domiciliaria_T0', 'Familiar_Prest_Assist_Dom_T0', 'Pessoa_Inst_Prest_Assist_Dom_T0',
                      'Assist_Domiciliaria_T1', 'Familiar_Prest_Assist_Dom_T1', 'Pessoa_Inst_Prest_Assist_Dom_T1', 'Outro_Prest_Assist_Dom_T1',
                      'Assist_Domiciliaria_T3', 'Familiar_Prest_Assist_Dom_T3', 'Pessoa_Inst_Prest_Assist_Dom_T3', 'Outro_Prest_Assist_Dom_T3']
        
        df[assistance] = df[assistance].fillna(0)

        # Merge all categories into selected_vars
        selected_vars = (ageing + multimorbidity + rheumatic_diseases + 
                        smoking_habits + alcohol_habits + exercise_freq + sleep_hours + dietary_behaviour +
                        education + employment + household_members + assistance)
        
        # List of variables to ensure are included
        required_vars = ['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']

        # Manually add required variables to selected_vars if they are in X.columns but not in selected_vars
        for var in required_vars:
            if var in df.columns and var not in selected_vars:
                selected_vars.append(var)
    
        return selected_vars, df

    def generate_variable_tables(data, variable_wave_map, group_col='DepressivePathGroup', output_dir='output/variable_tables'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        wave_label_map = {0: "T0", 1: "T1", 2: "T2", 3: "T3"}  # Mapping waves to labels

        for var_name, wave_dict in variable_wave_map.items():
            tables = []
            for wave, col_names in wave_dict.items():
                if not isinstance(col_names, list):
                    col_names = [col_names]

                for col_name in col_names:
                    if col_name not in data.columns:
                        continue  # Skip missing columns
                    
                    # Calculate mean by group
                    mean_table = data.groupby(group_col)[col_name].mean().reset_index()
                    mean_table['Wave'] = wave_label_map[wave]
                    mean_table['Variable'] = var_name
                    tables.append(mean_table)

            if tables:
                final_table = pd.concat(tables, ignore_index=True)
                final_table.to_csv(os.path.join(output_dir, f'{var_name}_table.csv'), index=False)

    def preprocess_data(self):
        df = self.data.copy()
        # Exclude these columns from predictors
        exclude_cols = ['ID', 'Pontuacao_Depressao', 'DepressivePathGroup',
                        'hads12', 'hads6', 'hads4', 'hads2', 'hads8', 'hads10', 'hads14', 'Time',
                        'DepressionStatus']
        
        selected_vars, df = self.define_chosen_variables(df)

        X = df.loc[:, ~df.columns.str.startswith(tuple(exclude_cols))]
        y = df['DepressivePathGroup']

        # Ensure only numeric columns remain
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        # Replace NaN values with 0 for participant columns
        participante_columns = [col for col in df.columns if col.startswith('Participante')]
        df[participante_columns] = df[participante_columns].fillna(0)

        # Handle missing values (if any)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Process categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
            X[col] = X[col].astype('category')

        # Remove Time or DepressivePathGroup if they appear in selected vars
        for col_to_remove in ['Time', 'DepressivePathGroup']:
            if col_to_remove in selected_vars:
                selected_vars.remove(col_to_remove)

        X = X[selected_vars]

        self.data = df

        return X, y

    def run_random_forest(self):
        self.create_depressive_path_columns()

        X, y = self.preprocess_data()

        # Exclude 'DepressivePath' from predictors (it's for grouping, not prediction)
        if 'DepressivePath' in X.columns:
            X = X.drop(columns=['DepressivePath'])

        # Split the data using the path groups (y) for stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Apply SMOTETomek on the training data
        smote_tomek = SMOTETomek(random_state=42)
        X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)

        # Generate frequency tables for socio-demographic variables
        frequency_tables = self.generate_frequency_tables(self.data)
        
        # Save frequency tables to CSV
        for col, table in frequency_tables.items():
            print("Frequency Table for ", col)
            print(table)
            table.to_csv(f'random_forest_balanced/output/frequency_tables/frequency_table_{col}.csv')

        # Reconstruct a balanced dataset (train + test) for evaluation and for downstream use
        data_bal = pd.concat([
            pd.DataFrame(X_train_bal, columns=X.columns),
            pd.DataFrame(X_test, columns=X.columns)
        ], ignore_index=True)

        # Create the DepressivePath and Group again in the balanced data
        data_bal['DepressivePath'] = (
            data_bal[['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']]
            .fillna(-1)
            .astype(int)
            .astype(str)
            .agg(''.join, axis=1)
        )

        path_to_group = {
            '000': 1,
            '001': 2, '010': 2, '100': 2, '101': 2,
            '011': 3, '110': 3, '111': 3
        }

        data_bal['DepressivePathGroup'] = data_bal['DepressivePath'].map(path_to_group).fillna(0).astype(int)

        # Define class weights for the multinomial scenario
        unique_classes = np.unique(np.concatenate((y_train, y_test)))
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=unique_classes, y=y_train
        )
        class_weights_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

        # Define pipeline
        pipeline = Pipeline([
            ('classification', RandomForestClassifier(n_estimators=300, random_state=42, class_weight=class_weights_dict))
        ])

        pipeline.fit(X_train_bal, y_train_bal)

        # Feature columns and full prediction
        feature_columns = X.columns
        X_full = self.data[feature_columns].fillna(0)
        full_proba = pipeline.predict_proba(X_full)
        self.data['rf_predictions'] = pipeline.predict(X_full)

        # Extract important features (top 10)
        #rf_model = pipeline.named_steps['classification']
        #self.extract_rf_important_features(rf_model, feature_columns, self.dataset_label)

        # Run separate models for each depressive path group
        self.run_rf_for_each_pathgroup(data_bal, feature_columns, self.dataset_label)

    def plot_confusion_matrix(self, y_true, y_pred, dataset_label, path_group):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Confusion Matrix - {dataset_label} at Path Group {path_group}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'random_forest_balanced/output/confusion_matrix_{dataset_label.replace(" ", "_")}_PG{path_group}.png', dpi=100, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, dataset_label, path_group):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_label} at Path Group {path_group}')
        plt.legend(loc="lower right")
        plt.savefig(f'random_forest_balanced/output/roc_curve_{dataset_label.replace(" ", "_")}_PG{path_group}.png', dpi=100, bbox_inches='tight')
        plt.close()

    def run_rf_for_each_pathgroup(self, data_bal, feature_columns, dataset_label):
        """
        Run separate Random Forest models for each depressive path group as a binary classification:
        That group vs. all others.
        """
        path_groups = [1, 2, 3]

        # Store metrics and data for plotting
        metrics = []
        plot_data = []

        for pg in path_groups:
            subset = data_bal.copy()
            subset['y_binary'] = (subset['DepressivePathGroup'] == pg).astype(int)

            if subset['y_binary'].nunique() < 2:
                print(f"Insufficient data or classes at Path Group {pg}. Cannot train model.")
                continue

            # Prepare data
            exclude_cols = ['DepressivePath', 'DepressivePathGroup', 'Score_Depressao', 'y_binary',
                            'ID', 'Pontuacao_Depressao', 'hads12', 'hads6',
                            'hads4', 'hads2', 'hads8', 'hads10', 'hads14']
            X = subset.loc[:, ~subset.columns.str.startswith(tuple(exclude_cols))]
            y = subset['y_binary']

            if X.empty:
                print(f"No features at Path Group {pg}.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            if y_test.nunique() > 1:
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = np.nan

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"\nPerformance Metrics for Path Group {pg} (Group vs Rest):")
            if not np.isnan(roc_auc):
                print(f"ROC AUC Score: {roc_auc:.4f}")
            else:
                print("ROC AUC: Undefined")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # Top 10 Important Features
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            important_features = feature_importances.nlargest(10)

            print(f"\nTop 10 Important Features at Path Group {pg} vs Rest:")
            print(important_features)
            important_features.to_csv(f'random_forest_balanced/output/random_forest_important_features_{dataset_label.replace(" ", "_")}_PG{pg}.csv')

            # Collect data for plotting
            metrics.append({'pg': pg, 'roc_auc': roc_auc, 'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'f1': f1})
            plot_data.append({'pg': pg, 'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba})

        # Plot confusion matrix and ROC curve for all path groups after the loop
        for data in plot_data:
            pg = data['pg']
            y_test = data['y_test']
            y_pred = data['y_pred']
            y_proba = data['y_proba']

            self.plot_confusion_matrix(y_test, y_pred, dataset_label, pg)

            if y_test.nunique() > 1:
                self.plot_roc_curve(y_test, y_proba, dataset_label, pg)

    def create_depressive_path_columns(self):
        # Create DepressivePath and DepressivePathGroup columns in self.data if not exist
        if not {'Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'}.issubset(self.data.columns):
            raise ValueError("Missing necessary score columns in the dataset.")

        self.data['DepressivePath'] = (
            self.data[['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3']]
            .fillna(-1)
            .astype(int)
            .astype(str)
            .agg(''.join, axis=1)
        )

        path_to_group = {
            '000': 1,
            '001': 2, '010': 2, '100': 2, '101': 2,
            '011': 3, '110': 3, '111': 3
        }
        self.data['DepressivePathGroup'] = self.data['DepressivePath'].map(path_to_group).fillna(0).astype(int)

    def create_wave_map(self):
        variable_wave_map = {
            "Ageing": {
                0: ["Idade_T0"],
                1: ["Idade_T1"],
                2: ["Idade_T2"],
                3: ["Idade_T3"]
            },
            "Multimorbidity": {
                0: "Multimorbidity_T0",
                1: "Multimorbidity_T1",
                2: "Multimorbidity_T2",
                3: "Multimorbidity_T3"
            },
            "Rheumatic Diseases": {
                0: ["AR_T0", "EA_T0", "AP_T0", "OA_T0", "OP_T0", "Gota_T0", "Polimialgia_T0", "Les_T0", "Fibromialgia_T0"],
                1: ["AR_T1", "EA_T1", "AP_T1", "Gota_T1", "Polimialgia_T1", "LES_T1", "Fibromialgia_T1", "Lombalgia_T1"],
                2: ["AR_T2", "EA_T2", "AP_T2", "OP_T2", "Gota_T2", "Polimialgia_T2", "LES_T2", "Fibromialgia_T2", "Lombalgia_T2"],
                3: ["AR_T3", "EA_T3", "AP_T3", "OP_T3", "Gota_T3", "Polimialgia_T3", "LES_T3", "Fibromialgia_T3"]
            },
            "Smoking": {
                0: ["Cod_Fumador_T0"],
                1: ["Cod_Fumador_T1"],
                2: ["Cod_Fumador_T2"],
                3: ["Cod_Fumador_T3"]
            },
            "Alcohol Consumption": {
                0: ["Cod_Alcool_T0"],
                1: ["Cod_Alcool_T1"],
                2: ["Cod_Alcool_T2"],
                3: ["Cod_Alcool_T3"]
            },
            "Regular Exercise": {
                0: ["Exercicio_Regular_T0"],
                1: ["Exercicio_Regular_T1"],
                2: ["Exercicio_Regular_T2"],
                3: ["Exercicio_Regular_T3"]
            },
            "Hours of Sleep per Day": {
                1: ["Num_Horas_Dorme_Por_Dia_T1"],
                2: ["Num_Horas_Dorme_Por_Dia_T2"],
                3: ["Num_Horas_Dorme_Por_Dia_T3"]
            },
            "Diet": {
                2: ["DM3_T2", "DM4_T2", "DM5_T2", "DM10_T2", "DM6_T2"]
            },
            "Education": {
                0: ["Sit_Prof_T0"]
            },
            "Profession": {
                0: ["Escolaridade"],
                1: ["Sit_Prof_T1"],
                2: ["Sit_Prof_T2"],
                3: ["Sit_Prof_T2"]
            },
            "Social Support": {
                0: ["Num_Pessoas_Agregado_T0", "Assist_Domiciliaria_T0", "Familiar_Prest_Assist_Dom_T0",
                    "Pessoa_Inst_Prest_Assist_Dom_T0"],
                1: ["Assist_Domiciliaria_T1", "Familiar_Prest_Assist_Dom_T1", "Pessoa_Inst_Prest_Assist_Dom_T1",
                    "Outro_Prest_Assist_Dom_T1"],
                2: ["Num_Pessoas_Agregado_T2"],
                3: ["Num_Pessoas_Agregado_T3", "Assist_Domiciliaria_T3", "Familiar_Prest_Assist_Dom_T3",
                    "Pessoa_Inst_Prest_Assist_Dom_T3", "Outro_Prest_Assist_Dom_T3"]
            }
        }

        return variable_wave_map
    
    def visualise(self, df):
        variable_wave_map = self.create_wave_map()
        group_col = 'DepressivePathGroup'
        out_dir = 'random_forest_balanced/output/variable_plots'
        table_out_dir = 'random_forest_balanced/output/variable_tables'
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(table_out_dir):
            os.makedirs(table_out_dir)
        
        # Map for categorical wave labels
        wave_label_map = {0: "T0", 1: "T1", 2: "T2", 3: "T3"}

        for var_name, wave_dict in variable_wave_map.items():
            print(f"Processing category: {var_name}")
            plot_df_list = []
            table_df_list = []  # To collect data for saving tables

            for w, col_names in wave_dict.items():
                # Ensure col_names is always a list
                if not isinstance(col_names, list):
                    col_names = [col_names]

                for col_name in col_names:
                    if col_name not in df.columns:
                        print(f"Missing column: {col_name} in category: {var_name}")
                        continue
                    
                    # Subset only rows that are not NaN
                    subset = df[[group_col, col_name]].dropna(subset=[col_name])
                    if subset.empty:
                        print(f"No data for column: {col_name} in category: {var_name}")
                        continue
                    
                    # Compute the mean by group
                    mean_by_group = subset.groupby(group_col)[col_name].mean().reset_index()
                    mean_by_group['wave'] = wave_label_map[w]  # Map wave to categorical label
                    mean_by_group['variable'] = var_name
                    mean_by_group.rename(columns={col_name: "value"}, inplace=True)
                    plot_df_list.append(mean_by_group)
                    table_df_list.append(mean_by_group)  # Add to table data

            if not plot_df_list:
                print(f"No data to plot for category: {var_name}")
                continue

            # Combine data for plotting
            long_df = pd.concat(plot_df_list, ignore_index=True)
            
            # Plot
            plt.figure(figsize=(6, 4))
            sns.lineplot(
                data=long_df,
                x='wave',
                y='value',
                hue=group_col,
                marker='o',
                palette='Dark2'
            )
            plt.title(f'{var_name} by Wave and Group')
            plt.xlabel('Wave')
            plt.ylabel(f'{var_name} (mean or proportion)')
            plt.legend(title=f'{group_col} (1,2,3)')
            
            plt.tight_layout()
            out_path = os.path.join(out_dir, f'longitudinal_{var_name}.png')
            plt.savefig(out_path, dpi=120)
            plt.close()

            # Save tables
            if table_df_list:
                final_table = pd.concat(table_df_list, ignore_index=True)
                table_path = os.path.join(table_out_dir, f'{var_name}_table.csv')
                final_table.to_csv(table_path, index=False)
                print(f"Saved table for {var_name} at {table_path}")


def main():
    # Load the dataset
    data = pd.read_excel("BD_Rute.xlsx", sheet_name="R")
    filtered_data = data.dropna(subset=['Score_Depressao_T0', 'Score_Depressao_T1', 'Score_Depressao_T3'])

    dataset_label = 'SMOTETomek Balanced Data'

    # Run Random Forest Model with Y as DepressivePathGroup
    rf_model = RandomForestLongitudinalModel(filtered_data, dataset_label)
    rf_model.run_random_forest()

    rf_model.visualise(rf_model.data)

if __name__ == "__main__":
    main()