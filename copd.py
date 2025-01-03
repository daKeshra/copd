import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Title of the App
st.title("Gas Flare and COPD Cases Prediction")

# Explanation Section
st.write("""
This app predicts **COPD cases** based on the input values of **Year** and **Gas Flare** using multiple regression models. 
You can also compare the performance of the models.
""")

# Input Section
st.subheader("Enter Input Variables")
year = st.number_input("Enter Year", min_value=2018, max_value=2040, step=1, value=2024)
gas_flare = st.number_input("Enter Gas Flare Value", min_value=300.0, step=100.0, value=500.0)


df = pd.read_csv('environmental_health_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
# st.write(df)

# Prepare Features and Target Variable
X = df[['year', 'gasflare']]
y = df['copd_cases']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Model
models = {
    "Linear Regression": LinearRegression(),
}


# Predict Button
if st.button("Predict"):
    # Model Training and Evaluation
    results = []
    predictions = {}

    # st.subheader("Model Performance")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "MSE": mse, "R²": r2})

        # Predict based on user input
        user_input = np.array([[year, gas_flare]])
        predictions[name] = model.predict(user_input)[0]

        # Display model performance
        st.write(f"**{name}**: Mean Squared Error = {mse:.4f}, R² = {r2:.4f}")


    st.subheader("Predicted COPD Cases Based on Your Input")
    for model_name, pred in predictions.items():
        st.write(f"**{model_name}** predicts: {round(pred)} COPD cases")
        # st.write(f"**{model_name}** predicts: {pred:.0f} COPD cases")


    # Results Table
    results_df = pd.DataFrame(results)
    
    
