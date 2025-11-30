# 💳 Online Fraud Detection using Machine Learning

This project builds a machine learning classification model to detect fraudulent online transactions.  
Since fraudulent transactions are rare compared to legitimate ones, the dataset contains a **significant class imbalance**.  
To address this, sampling techniques and model evaluation strategies were applied to improve fraud identification accuracy.

---

## 📂 Dataset

The dataset contains online transaction records with fields such as:

- Transaction Type (cash-out, transfer, etc.)
- Transaction Amount
- Account balance before & after transaction
- Flags indicating suspicious activity
- **Target Variable: `isFraud`**
  - `1` → Fraudulent transaction  
  - `0` → Legitimate transaction  

---

## 🔍 Exploratory Data Analysis (EDA)

Key findings from data analysis:

- Fraud was mostly observed in **`TRANSFER`** and **`CASH_OUT`** transactions.
- The dataset is **highly imbalanced**, with very few fraudulent transactions.
- Several correlations and visualizations were plotted to understand patterns.

---

## 🛠️ Data Preprocessing

Steps performed:

- Handling missing values
- Feature scaling using normalization/standardization
- Encoding categorical transaction types
- Splitting dataset into training and testing sets
- Applying **undersampling** to balance classes

---

## 🤖 Machine Learning Models Tested

The following models were trained and evaluated:

| Model | Evaluated |
|-------|-----------|
| Logistic Regression | ✔ |
| Decision Tree | ✔ |
| Random Forest | ✔ |
| Gradient Boosting | ✔ |
| XGBoost | ✔ |
| Support Vector Machine (SVM) | ✔ |
| K-Nearest Neighbors (KNN) | ✔ |
| Naive Bayes | ✔ |

---

## 🧪 Best Model & Evaluation

After testing multiple algorithms, the best performing model was:

> 🏆 **Gradient Boosting Classifier**  
> 📈 **Accuracy:** `99.6%`

The model demonstrated strong classification performance on the balanced dataset.

---

## 🧰 Libraries & Tools Used

| Category | Frameworks |
|----------|------------|
| Programming | Python |
| Data Handling | `numpy`, `pandas` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Imbalance Handling | `imbalanced-learn` |

---

## ▶️ How to Run This Project Locally

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd online-fraud-detection
```
2️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run Jupyter Notebook
```bash
jupyter notebook online-fraud-detection-classification.ipynb
```

🧠 Predict Fraud Manually (If Model Exported)
```bash
import pickle
```

# Load model
```bash
model = pickle.load(open("model.pkl", "rb"))
```

# Example input (must match dataframe structure)
```bash
sample = [[2000, 0, 50000, "TRANSFER"]]  # Example transaction
prediction = model.predict(sample)
print("Prediction:", "Fraud" if prediction == 1 else "Not Fraud")
```

🚀 Future Improvements
1. Try SMOTE or ADASYN instead of undersampling
2. Train deep learning models (LSTM, Neural Networks)
3. Deploy model with a REST API or Streamlit UI

📄 License
This project is distributed under the MIT License.

❤️ Acknowledgments
Dataset Source: Kaggle

Open-source contributors of Python ML libraries
