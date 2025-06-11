# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------
# Judul Aplikasi
# ------------------------------
st.title("Prediksi Gaji Berdasarkan Pengalaman Kerja")
st.write("Aplikasi ini memprediksi gaji karyawan berdasarkan pengalaman kerja menggunakan Regresi Linear Sederhana.")

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_csv('Salary.csv')  # Pastikan file Salary.csv ada di direktori yang sama
st.subheader("Dataset")
st.dataframe(df.head())

# ------------------------------
# Preprocessing
# ------------------------------
X = df[['YearsExperience']]
y = df['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ------------------------------
# Training Model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ------------------------------
# Evaluasi Model
# ------------------------------
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("Evaluasi Model")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**R-squared (R² Score):** {r2:.2f}")

# ------------------------------
# Plot: Residual Plot
# ------------------------------
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_test - predictions)
ax1.axhline(y=0, color='r', linestyle='-')
ax1.set_title('Residual Plot')
ax1.set_xlabel('Actual Salary')
ax1.set_ylabel('Residuals')
st.pyplot(fig1)

# ------------------------------
# Plot: Regresi Linear Seluruh Data
# ------------------------------
model_all = LinearRegression()
model_all.fit(X, y)
y_pred_all = model_all.predict(X)

fig2, ax2 = plt.subplots()
ax2.scatter(X, y, color='red', label='Data Asli')
ax2.plot(X, y_pred_all, color='blue', label='Garis Regresi')
ax2.set_title('Pengalaman Kerja vs Gaji')
ax2.set_xlabel('Years of Experience')
ax2.set_ylabel('Salary')
ax2.legend()
st.pyplot(fig2)

# ------------------------------
# Prediksi Berdasarkan Input Pengguna
# ------------------------------
st.subheader("Prediksi Gaji Anda")

# Input dari user
years_input = st.number_input("Masukkan pengalaman kerja Anda (dalam tahun):", min_value=0.0, max_value=50.0, value=1.0, step=0.1)

# Prediksi
predicted_salary = model_all.predict([[years_input]])[0]
st.success(f"Perkiraan gaji untuk {years_input:.1f} tahun pengalaman adalah: **${predicted_salary:,.2f}**")

# ------------------------------
# Info Koefisien Model
# ------------------------------
st.write("---")
st.subheader("Informasi Model")
st.write(f"**Intercept (B₀):** {model_all.intercept_:.2f}")
st.write(f"**Slope (B₁):** {model_all.coef_[0]:.2f}")
