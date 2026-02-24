# Customer Churn Prediction using ANN

##  Project Overview
Customer churn occurs when customers stop doing business with a company. This project implements an **Artificial Neural Network (ANN)** to predict the likelihood of a customer leaving based on features like credit score, geography, gender, age, tenure, and balance.

##  Key Features
* **Multi-Layer Perceptron:** Optimized hidden layers with `ReLU` activation.
* **Data Preprocessing:** Handled categorical data using `OneHotEncoding` and feature scaling with `StandardScaler`.
* **Binary Classification:** Uses `Sigmoid` activation in the output layer to predict churn probability.

##  Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

##  Model Architecture
```python
# A snippet of the model structure
model = Sequential([
    Dense(units=6, activation='relu', input_dim=11), # Input + Hidden Layer 1
    Dense(units=6, activation='relu'),              # Hidden Layer 2
    Dense(units=1, activation='sigmoid')           # Output Layer
])
