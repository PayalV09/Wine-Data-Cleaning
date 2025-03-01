# Wine Data Cleaning & Machine Learning Project

## 🔗 Repository
[Wine Data Cleaning & Machine Learning](https://github.com/PayalV09/Wine-Data-Cleaning)

## 📜 Project Overview
This project focuses on cleaning and preprocessing a wine dataset to prepare it for machine learning models. The dataset undergoes various steps like handling missing values, outlier detection, feature engineering, and normalization. Additionally, a machine learning model is trained to predict wine quality based on the cleaned dataset.

## 🚀 Features
✔️ Data Preprocessing (Handling Missing Values, Duplicates, etc.)  
✔️ Exploratory Data Analysis (EDA) with Visualizations  
✔️ Outlier Detection and Removal  
✔️ Feature Engineering and Encoding  
✔️ Data Normalization & Scaling  
✔️ Machine Learning Model for Wine Quality Prediction  
✔️ Ready-to-use Cleaned Dataset  

## 📂 Dataset
The dataset contains information about different wine samples, including factors like acidity, alcohol content, and quality.

### 🔽 Download the dataset:
- **Source:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
- **Alternative:** If included in the repo, you can find it in the `/data` folder.

## 💻 Installation 🔧
### 1. Clone the Repository
```sh
git clone https://github.com/PayalV09/Wine-Data-Cleaning.git
cd Wine-Data-Cleaning
```

### 2. Install Dependencies
Make sure you have Python installed (>=3.8). Then, install the required libraries:
```sh
pip install -r requirements.txt
```
Or manually install:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Download the Dataset (if not included in repo)
```sh
mkdir data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv -P data/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv -P data/
```

## ▶️ How to Run the Project
### Using Jupyter Notebook
1. Open Jupyter Notebook:
2. Navigate to `wine_data_cleaning.ipynb` and run the cells step by step.

### Using Python Script
Run the preprocessing script directly:
```sh
python clean_wine_data.py
```

## 📊 Project Workflow
1️⃣ Load the dataset  
2️⃣ Handle missing values  
3️⃣ Remove duplicate entries  
4️⃣ Perform Exploratory Data Analysis (EDA)  
5️⃣ Detect & remove outliers  
6️⃣ Normalize & scale the data  
7️⃣ Train and evaluate a machine learning model  
8️⃣ Save the cleaned dataset for further use  

## 🤖 Machine Learning Model
### **Wine Quality Prediction**
#### **Model Used:** Random Forest Classifier

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the cleaned dataset
wine_df = pd.read_csv('cleaned_wine_data.csv')

# Splitting features and target
X = wine_df.drop(columns=['quality'])
y = wine_df['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=1))
```

## 📷 Screenshots
1.![Screenshot (289)](https://github.com/user-attachments/assets/0cbafc4d-06a4-44b4-8a3c-d30818023c5a)

2.![Screenshot (287)](https://github.com/user-attachments/assets/dc6b04d1-a3bf-46db-92d2-3e4be99e0cef)

3.![Screenshot (284)](https://github.com/user-attachments/assets/3993a606-026a-42c9-821e-9ab8fcac1862)

4.![image](https://github.com/user-attachments/assets/96e37ef3-64d9-40a7-8e29-b54ff01b1a3e)



## 🛠 Future Enhancements
✅ Apply more machine learning models (XGBoost, SVM, etc.)  
✅ Integrate additional datasets for better analysis  
✅ Build an interactive dashboard  
✅ Optimize feature selection and hyperparameters  


Happy coding! 🚀

