# Application-of-generative-models-for-anonymization-and-data-augmentation

This repository contains a collection of Jupyter notebooks dedicated to the training, generation, and evaluation of synthetic data using several generative models for tabular datasets.  
The project focuses on two benchmark datasets:

- German Credit Risk  
- Australian Credit Approval  

The objective is to compare both traditional and transformer-based generative models in terms of:

- Predictive utility (accuracy of classifiers trained on synthetic data)  
- Distinguishability (using a Random Forest discriminator)  
- Similarity between synthetic and real records (Distance to Closest Record, DCR)

---

## Notebook Overview

### 1. Training – German Credit Dataset
Trains the models CopulaGAN, TVAE, CTGAN, and Distil-GReaT on the German Credit dataset.  
Includes preprocessing, model fitting, and saving the trained models (`.pkl`).

---

### 2. GReaT Training – German Credit
Trains the full GPT-2 based GReaT model on the German Credit dataset.  
This notebook is separated due to the high training time required.

---

### 3. Synthetic Generation – German Credit
Loads all trained models and generates synthetic datasets with the same size as the original German dataset.  
Outputs are saved as `.csv` files for evaluation.

---

### 4. Training – Australian Credit Dataset
Same structure as the German training notebook, but applied to the Australian Credit dataset.  
Trains CopulaGAN, TVAE, CTGAN, and Distil-GReaT.

---

### 5. GReaT Training – Australian Credit
Trains the GReaT model on the Australian Credit dataset.  
Kept separate due to computational cost and longer runtime.

---

### 6. Synthetic Generation – Australian Credit
Generates synthetic samples for the Australian dataset using all trained models.  
Saves `.csv` datasets used later for evaluation.

---

### 7. Evaluation & Results
Performs a complete evaluation of all generated datasets, including:

- Predictive models: Logistic Regression, Decision Tree, and Random Forest  
- Discriminator: Random Forest distinguishing real vs. synthetic data  
- DCR analysis: Computes the Distance to Closest Record and visualizes its distribution  

Concludes with aggregated metrics, plots, and a comparative discussion of model performance.

---

## Dependencies

Main libraries used across notebooks:

```bash
pandas
numpy
matplotlib
scikit-learn
sdv
be_great
pickle
```
## Recommended Execution Order

- Train models on German Credit

- Train models on Australian Credit

- Generate synthetic datasets

- Run the Evaluation & Results notebook

## Outputs

The notebooks produce the following artifacts:

- Trained models (.pkl)

- Synthetic datasets (.csv)

- Evaluation metrics, plots, and summary tables
