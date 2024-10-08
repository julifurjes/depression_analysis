# **Depression Analysis Using Different Models**

This project explores the analysis of depression data using different models, such as **Mixed-Effects Models** (Multilevel Models) and **XG Boost Model**. The focus is on understanding how demographic factors influence depression scores over time and across subjects, utilizing longitudinal data collected as part of my internship project.

## **Table of Contents**

- [Project Overview](#project-overview)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## **Project Overview**

In this project, I apply these models to a longitudinal dataset to investigate how variables such as gender, education level, and age affect depression scores across different time points. The analysis helps us capture both **within-subject** and **between-subject** variability in depression patterns.

The key steps include:
- Preparing and cleaning the dataset.
- Fitting the models for three time points of depression scores (`Score_Depressao_T0`, `Score_Depressao_T1`, and `Score_Depressao_T3`).
- Analyzing the outcome:
    - **Mixed Effects Model**: fixed effects (demographic variables) and random effects (subject variability).
    - **XG Boost**: feature importance of the demographic and subject ID columns.

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

```python mixed_effects/script_mixedlm_dem.py```

2. Interpreting the Results:
The results for each time point will be saved in the corresponding output folders as text files or images, containing the model summaries for each depression score.

Example:
```
mixed_effects/output/
    mixed_effects_results_score_Score_Depressao_T0.txt
    mixed_effects_results_score_Score_Depressao_T1.txt
    mixed_effects_results_score_Score_Depressao_T3.txt
```

## **Results**

The analysis provides insights into how demographic factors influence depression over time. The results include:

**Mixed Effects Model:**
- Fixed Effects: Effects of gender, education, and age on depression scores.
- Random Effects: Variability in depression scores across subjects.

**XG Boost:**
- Feature importance: The effect of each variable on the depression scores.

## **Acknowledgments**

This project is part of my internship work. Special thanks to my supervisor and team for this possibility.