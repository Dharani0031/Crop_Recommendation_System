import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Load dataset
df = pd.read_csv('C:/Users/dhara/Downloads/Crop_recommendation.csv')

# Encode the target labels
c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes

# Prepare the features and target variable
y = df.target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the K-Nearest Neighbors model
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("Crop Recommendation System")
st.write("Enter the following details to get a crop recommendation:")

N = st.number_input("Nitrogen content (N):", min_value=0.0, max_value=100.0, value=0.0)
P = st.number_input("Phosphorous content (P):", min_value=0.0, max_value=100.0, value=0.0)
K = st.number_input("Potassium content (K):", min_value=0.0, max_value=100.0, value=0.0)
temperature = st.number_input("Temperature (in °C):", min_value=0.0, max_value=50.0, value=0.0)
humidity = st.number_input("Humidity (in %):", min_value=0.0, max_value=100.0, value=0.0)
ph = st.number_input("pH value of the soil:", min_value=0.0, max_value=14.0, value=0.0)
rainfall = st.number_input("Rainfall (in mm):", min_value=0.0, max_value=300.0, value=0.0)

if st.button("Recommend Crop"):
    # Create a DataFrame for the user input
    user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    # Scale the user input using the same scaler
    user_input_scaled = scaler.transform(user_input)

    # Predict the crop
    predicted_target = knn.predict(user_input_scaled)
    predicted_crop = targets[predicted_target[0]]
    
    st.success(f"The recommended crop to grow is: {predicted_crop}")
    
    # Visualize the feature importance or a relevant plot
    st.write("### Feature Distribution")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df['temperature'], color="purple", bins=15, ax=ax[0], kde=True)
    ax[0].axvline(x=temperature, color='red', linestyle='--', label='Your Input')
    ax[0].legend()
    ax[0].set_title('Temperature Distribution')

    sns.histplot(df['ph'], color="green", bins=15, ax=ax[1], kde=True)
    ax[1].axvline(x=ph, color='red', linestyle='--', label='Your Input')
    ax[1].legend()
    ax[1].set_title('pH Distribution')

    st.pyplot(fig)
