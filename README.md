# **Depression Analysis Using Different Models**

This project explores the analysis of depression data using different models, such as **Mixed-Effects Models** (Multilevel Models) and **Random Forest Model**. The focus is on understanding how demographic factors influence depression scores over time and across subjects, utilizing longitudinal data collected as part of my internship project.

## **Table of Contents**

- [Project Overview](#project-overview)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

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

```python mixed_effects/mixed_effects.py```

2. Interpreting the Results:
The results for each time point will be saved in the corresponding output folders as text files or images, containing the model summaries for each depression score.

Example:
```
mixed_effects/output/
    mixed_effects_coefficients.csv
    mixed_effects_results.txt
    predictive_performance_metrics.txt
```

## **Results**

The analysis provides insights into how different factors influence depression over time.

## **Acknowledgments**

This project is part of my internship work. Special thanks to my supervisor and team for this possibility.