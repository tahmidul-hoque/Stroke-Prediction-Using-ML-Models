# Predicting Stroke Occurrence Using Machine Learning

## 📌 Project Overview
Stroke is a leading cause of mortality and long-term disability globally. Early identification of high-risk patients allows medical professionals to implement preventative care, significantly reducing healthcare burdens and saving lives. 

This project applies a rigorous machine learning pipeline to predict the discrete binary target of "Stroke" (0 for no stroke, 1 for stroke) based on historical patient data, encompassing demographics, pre-existing conditions, and lifestyle metrics.

## 📊 Dataset Description
*   **Initial Size:** 5,110 patient records with 12 features.
*   **Cleaned Size:** 5,109 records (removed non-predictive `id` and one anomaly row).
*   **Class Imbalance:** The dataset is heavily imbalanced, with 95.1% (4,861) non-stroke cases and only 4.9% (249) stroke cases. This severe 95:5 disparity creates a "Majority Class Baseline" where naive models achieve 95.1% accuracy by simply predicting "No Stroke" every time.

## 🛠️ Data Pre-Processing & Feature Engineering
To ensure robust model training and avoid artificial bias, we implemented several pre-processing steps:
1.  **Categorical Encoding:** Applied binary mapping for two-state features (`gender`, `ever_married`, `Residence_type`) and One-Hot Encoding for nominal multi-state categories.
2.  **Correlation Analysis:** Utilized Spearman Rank (monotonic) and Pearson (linear) correlation matrices to detect relationships and multicollinearity.
3.  **Feature Selection:** Dropped `gender`, `work_type`, and `smoking_status_Unknown` due to low correlation with the target. Dropped `ever_married` and `bmi` due to high collinearity with `age`.
4.  **Feature Scaling:** Applied `StandardScaler` to continuous features (like `avg_glucose_level`) to prevent magnitude bias in distance-based algorithms.
5.  **Handling Imbalance:** Implemented **RandomUnderSampling** to force models to learn minority class decision boundaries, overcoming the Accuracy Paradox.

## 🤖 Machine Learning Models
We evaluated both supervised and unsupervised models:
*   **Logistic Regression:** A parametric model fitting a linear equation to scaled features.
*   **Neural Network:** A multi-layer feed-forward network utilizing ReLU activations for hidden layers and Sigmoid for the output, optimized with binary cross-entropy loss.
*   **K-Nearest Neighbors (KNN):** A distance-based algorithm predicting risk based on Euclidean distance.
*   **K-Means Clustering:** An unsupervised approach used to partition the feature space into distinct clusters.

## 📈 Evaluation & Results
Due to the dataset's severe imbalance, standard accuracy is deceptive. We utilized **Stratified 5-Fold Cross-Validation** and prioritized **Recall (Sensitivity)** to minimize False Negatives (critical misses) and **PR-AUC / AUC-ROC** for class separation capability.

### Aggregated 5-Fold Model Comparison
| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.6721 | 0.1175 | 0.8753 | 0.2070 | **0.8380** | **0.1955** |
| **Neural Network** | **0.6851** | 0.1182 | 0.8429 | **0.2072** | 0.8319 | 0.1855 |
| **KNN** | 0.6310 | 0.1071 | **0.8834** | 0.1906 | 0.8110 | 0.1493 |
| **K-Means (Unsup.)** | 0.8444 | 0.1342 | 0.4020 | 0.2012 | 0.6345 | 0.0862 |

## 🏆 Conclusion
**Logistic Regression** stands out as the overall best-performing and most reliable model for stroke prediction on this dataset. 

*   It achieved the highest discriminative power (AUC-ROC: 0.8380, PR-AUC: 0.1955) alongside a strong Recall (0.8753).
*   Its parametric nature prevented it from chasing noise in the smaller, undersampled dataset, leveraging the strong linear/monotonic relationships of key predictors like age and average glucose level.
*   It offers superior clinical interpretability compared to complex, "black-box" models.
*   Conversely, distance-based models (KNN, K-Means) struggled significantly due to the "Curse of Dimensionality" introduced by encoded nominal features.

## 👥 Authors
*   **S. M. Tahmidul Hoque** (ID: 24341017)
*   **Fairuz Imam** (ID: 24341041)
*   *Department of Computer Science and Engineering, BRAC University*
