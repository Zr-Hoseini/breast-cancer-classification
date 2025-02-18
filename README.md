# breast-cancer-classification
 "A machine learning project for breast cancer classification using UCI dataset and various ML models."
# 🏥 Breast Cancer Detection (UCI Dataset)

## 📌 Project Overview
This project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the **UCI Machine Learning Repository**, available in `sklearn.datasets`.  
The dataset contains **features extracted from digitized images of fine needle aspirates (FNA) of breast cancer biopsies**.  
The objective is to develop a **machine learning model** that can classify tumors as **benign** or **malignant** based on these features.

## 📝 Introduction
1. **Data Loading**: The dataset was loaded using `sklearn.datasets.load_breast_cancer()`.  
2. **Exploration**: Basic dataset characteristics were analyzed using `print(data.DESCR)`.  
3. **Preprocessing**:  
   - The dataset was split into **training** and **testing** sets.  
   - `MinMaxScaler` was applied to scale features within the **0-1 range**.  
## 🔬 Machine Learning Models Evaluated
Several classification algorithms were tested, including:  
- **Naïve Bayes (GaussianNB)**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree (DT)**  
- **Random Forest (RF)**  
- **Support Vector Machine (SVM)**  
- **Logistic Regression (LR)**  
- **Artificial Neural Network (ANN)**  

Each model was evaluated based on the following metrics:  
✅ **Training Accuracy**  
✅ **Testing Accuracy**  
✅ **Precision**  
✅ **Recall**  
✅ **F1-Score**  
✅ **Confusion Matrix**  

## 📊 Model Comparison & Results  

| Model                  | Test Accuracy | Precision | Recall | F1-Score | Overfitting |
|------------------------|--------------|-----------|--------|----------|-------------|
| **Naïve Bayes (GNB)**  | 0.93         | 0.926     | 0.974  | 0.950    | No          |
| **KNN**                | 0.95         | 0.939     | 0.987  | 0.9625   | No          |
| **Decision Tree (DT)** | 0.91         | 0.914     | 0.961  | 0.9375   | **Yes** (Train Acc = 100%) |
| **Random Forest (RF)** | 0.96         | 0.950     | 0.987  | 0.9685   | Low         |
| **SVM**                | 0.95         | 0.950     | 0.974  | 0.962    | No          |
| **Logistic Regression (LR)** | 0.94  | 0.938     | 0.974  | 0.9559   | No          |
| **Artificial Neural Network (ANN)** | **0.97** | **0.974** | **0.987** | **0.9808** | No |

📌 **Final Model Selection**:  
ANN outperformed all other models in terms of **accuracy, precision, recall, and F1-score**, making it the **best choice** for this task.  
Final ranking: **ANN > RF > SVM > KNN > LR > GNB > DT**  

## 📂 Installation & Usage  
To run this project on your local machine:  

###  Install Dependencies  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn




##📌 Future Improvements
Hyperparameter tuning for better ANN performance.
Ensemble Learning to combine multiple models.
Feature selection for better interpretability.


##📌 Author
Developed by [Zahra Hoseini] 🧑‍💻
If you found this project helpful, feel free to ⭐️ the repo!
