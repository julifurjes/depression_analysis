import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

class ModelComparison:
    def __init__(self, metrics_files_dict, prediction_files_dict):
        """
        Initialize the ModelComparison class with the paths to the metrics files and prediction files.

        :param metrics_files_dict: A dictionary where keys are model names, and values are dictionaries of score file paths.
        :param prediction_files_dict: A dictionary where keys are model names, and values are dictionaries of prediction file paths.
        """
        self.metrics_files_dict = metrics_files_dict
        self.prediction_files_dict = prediction_files_dict
        self.combined_metrics = None

    def load_metrics(self, model_name, file_path):
        """
        Load the evaluation metrics for a given model from the CSV file.

        :param model_name: Name of the model (e.g., 'XGBoost_T0', 'MixedEffects', etc.).
        :param file_path: Path to the metrics CSV file.
        :return: A DataFrame with the metrics and model name.
        """
        if os.path.exists(file_path):
            metrics_df = pd.read_csv(file_path)
            metrics_df['Model'] = model_name
            return metrics_df
        else:
            print(f"File not found: {file_path}")
            return None

    def load_predictions(self, file_path):
        """
        Load the prediction file for a given model.

        :param file_path: Path to the predictions CSV file.
        :return: A DataFrame with the actual and predicted values.
        """
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"File not found: {file_path}")
            return None

    def calculate_metrics_from_predictions(self, y_true, y_pred, model_name):
        """
        Calculate accuracy, precision, recall, and F1-score from the actual and predicted values.

        :param y_true: Array of actual values.
        :param y_pred: Array of predicted values.
        :param model_name: Name of the model (for labeling).
        :return: A DataFrame with the calculated metrics.
        """
        # If y_pred contains continuous values (probabilities), convert to binary
        if y_pred.dtype != 'int' and y_pred.dtype != 'bool':
            y_pred = (y_pred >= 0.5).astype(int)  # Threshold at 0.5 to convert to binary

        # Calculate the metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        metrics = {
            'Model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
        }
        
        return pd.DataFrame([metrics])

    def process_predictions(self):
        """
        Process prediction files and calculate metrics for each time point.
        """
        all_metrics = []

        for model_name, files_dict in self.prediction_files_dict.items():
            for score_name, file_path in files_dict.items():
                prediction_data = self.load_predictions(file_path)
                if prediction_data is not None:
                    # Assuming columns are named 'Actual' and 'Predicted' in the prediction CSVs
                    y_true = prediction_data['Actual']
                    y_pred = prediction_data['Predicted']
                    metrics = self.calculate_metrics_from_predictions(y_true, y_pred, f"{model_name}_{score_name}")
                    all_metrics.append(metrics)
        
        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        else:
            print("No prediction metrics to process.")
            return None

    def load_all_metrics(self):
        """
        Load all model evaluation metrics from precomputed metrics files and combine them.
        """
        all_metrics = []

        # Load the metrics for each model and their associated CSVs
        for model_name, files_dict in self.metrics_files_dict.items():
            for score_name, file_path in files_dict.items():
                metrics = self.load_metrics(f"{model_name}_{score_name}", file_path)
                if metrics is not None:
                    all_metrics.append(metrics)

        # Combine all metrics into a single DataFrame
        if all_metrics:
            self.combined_metrics = pd.concat(all_metrics, ignore_index=True)
        else:
            print("No precomputed metrics to combine.")

    def save_combined_metrics(self, output_path):
        """
        Save the combined metrics DataFrame to a CSV file.

        :param output_path: Path to save the combined metrics CSV.
        """
        if self.combined_metrics is not None:
            self.combined_metrics.to_csv(output_path, index=False)
            print(f"Combined metrics saved to '{output_path}'")
        else:
            print("No combined metrics to save.")

    def compare_metrics(self):
        """
        Display and compare models based on accuracy and F1-score.
        """
        if self.combined_metrics is None:
            print("No metrics loaded. Please load metrics before comparison.")
            return

        print("Combined Metrics for All Models:")
        print(self.combined_metrics)

        # Perform comparison by metrics (e.g., accuracy, F1-score)
        print("\nComparing models based on accuracy and F1-score:")
        
        accuracy_df = self.combined_metrics[['Model', 'accuracy']].sort_values(by="accuracy", ascending=False)
        f1_score_df = self.combined_metrics[['Model', 'f1-score']].sort_values(by="f1-score", ascending=False)

        print("Accuracy Comparison:")
        print(accuracy_df)

        print("\nF1-Score Comparison:")
        print(f1_score_df)

def main():
    # Ensure the output directory exists
    if not os.path.exists("model_comparisons/output"):
        os.makedirs("model_comparisons/output")

    # Paths to the saved metrics files for each model and time point
    metrics_files_dict = {
        "XGBoost": {
            "T0": "xgboost/output/xgboost_metrics_score_Score_Depressao_T0.csv",
            "T1": "xgboost/output/xgboost_metrics_score_Score_Depressao_T1.csv",
            "T3": "xgboost/output/xgboost_metrics_score_Score_Depressao_T3.csv",
            "Difference_T1_T3": "xgboost/output/xgboost_metrics_difference_Score_Diff_T1_T3.csv"
        },
        "DecisionTree": {
            "Overall": "decision_tree/output/decision_tree_metrics.csv"
        }
    }

    # Paths to the prediction files for Mixed Effects and GEE models
    prediction_files_dict = {
        "MixedEffects": {
            "T0": "mixed_effects/output/mixed_effects_predictions_Score_Depressao_T0.csv",
            "T1": "mixed_effects/output/mixed_effects_predictions_Score_Depressao_T1.csv",
            "T3": "mixed_effects/output/mixed_effects_predictions_Score_Depressao_T3.csv"
        },
        "GEE": {
            "Score_Depressao": "log_reg/output/gee_predictions_Score_Depressao.csv"
        }
    }

    # Instantiate the ModelComparison class
    model_comparator = ModelComparison(metrics_files_dict, prediction_files_dict)

    # Load precomputed metrics from files (for DecisionTree and XGBoost)
    model_comparator.load_all_metrics()

    # Process Mixed Effects and GEE predictions and calculate metrics
    mixed_effects_gee_metrics = model_comparator.process_predictions()
    
    if mixed_effects_gee_metrics is not None:
        # Combine the Mixed Effects and GEE metrics with precomputed metrics
        if model_comparator.combined_metrics is not None:
            model_comparator.combined_metrics = pd.concat([model_comparator.combined_metrics, mixed_effects_gee_metrics], ignore_index=True)
        else:
            model_comparator.combined_metrics = mixed_effects_gee_metrics

    # Compare the models based on metrics
    model_comparator.compare_metrics()

    # Save the combined metrics to a CSV for reference
    output_path = "model_comparisons/output/combined_metrics.csv"
    model_comparator.save_combined_metrics(output_path)

if __name__ == "__main__":
    main()