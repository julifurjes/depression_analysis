# **Depression Analysis Using Different Models**

## **Project Overview**

In this project, I apply these models to a longitudinal dataset to investigate how different variables can possibly predict depression as a binary outcome.

> **Note**: Due to **GDPR** (General Data Protection Regulation), the dataset used in this project cannot be publicly uploaded. If you'd like to replicate the analysis, you must add the dataset locally.

## **Data Requirements**

To replicate this project:
- You need to add the dataset `BD_Rute.xlsx` to the **root folder** of the project.
- The dataset should follow the structure used in the analysis, with the required variables.

**Dataset Path**:  
```/depression_analysis/BD_Rute.xlsx```

## **Installation**

To set up the project environment and install dependencies:

1. Clone the repository:
```
git clone https://github.com/julifurjes/depression-analysis.git
cd depression-analysis
```

2. Install required packages (e.g., statsmodels, pandas, matplotlib, etc.):
```pip install -r requirements.txt```

> **Note**: Ensure you have Python 3.x installed.

## **Usage**

Once you have the dataset in place and the dependencies installed, you can run the analysis as follows:

1. Running the scripts:

Example:

```python gee_rf/gee_rf_smote_tomek_by_timestamps.py```

2. Interpreting the Results:
The results of the model is saved under ```gee_rf/output```. This folder also contains a text file called ```model_and_output_interpreted.md```, where you can read the output in an easily interpretable way. However, you can also find all the raw outputs and visualisations in that folder as well.

## **Acknowledgments**

This project is part of my internship work. Special thanks to my supervisor and team for this possibility.
