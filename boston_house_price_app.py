import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the saved model
loaded_model = joblib.load("linear_regression_model.pkl")

# Define the features for input
# CRIM = per capita crime rate by town
# RM = average number of rooms per dwelling
# AGE = proportion of owner-occupied units built prior to 1940
# MEDV = Median value of owner-occupied homes in $1000's
features = ["CRIM", "RM", "AGE"]

# Define the Streamlit app
st.title("Boston House Price Prediction App")

# Add input fields for user to enter data
crim = st.slider("Per capita crime rate by town", min_value=0, max_value=100, value=5)
rm = st.slider(
    "Average number of rooms per dwelling", min_value=0, max_value=10, value=2
)
age = st.slider("Number of Age", min_value=0, max_value=120, value=15)

# Create a DataFrame with the user input
example_data = pd.DataFrame([[crim, rm, age]], columns=features)

# Make predictions
predicted_charges = loaded_model.predict(example_data)

# Display the prediction
st.write(f"Predicted Charges: ${predicted_charges[0]:.2f}")

# Add calculation of Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
actual_charges = 10000  # Replace with the actual charges if available
if actual_charges:
    mse = mean_squared_error([actual_charges], [predicted_charges[0]])
    rmse = np.sqrt(mse)
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
else:
    st.write("Actual charges not provided. Unable to calculate MSE and RMSE.")
