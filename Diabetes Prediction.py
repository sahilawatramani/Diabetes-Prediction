import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# File for storing user data
USER_DATA_FILE = "users.csv"

# Load dataset
def load_data():
    dataset = pd.read_csv("C:\\Users\\Victus\\Downloads\\archive (6)\\diabetes.csv")
    dataset['Glucose'].replace(0, dataset['Glucose'].median(), inplace=True)
    dataset['BloodPressure'].replace(0, dataset['BloodPressure'].median(), inplace=True)
    dataset['BMI'].replace(0, dataset['BMI'].mean(), inplace=True)
    dataset['SkinThickness'].replace(0, dataset['SkinThickness'].mean(), inplace=True)
    dataset['Insulin'].replace(0, dataset['Insulin'].mean(), inplace=True)
    return dataset

dataset = load_data()

# Secure password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Save user registration
def save_user(username, password):
    hashed_password = hash_password(password)

    if os.path.exists(USER_DATA_FILE):
        df = pd.read_csv(USER_DATA_FILE)
    else:
        df = pd.DataFrame(columns=["Username", "Password"])

    if username in df["Username"].values:
        return False  # User already exists

    df = pd.concat([df, pd.DataFrame([{"Username": username, "Password": hashed_password}])], ignore_index=True)
    df.to_csv(USER_DATA_FILE, index=False)
    return True  # Registration successful

# Authenticate user login
def authenticate_user(username, password):
    if not os.path.exists(USER_DATA_FILE):
        return False

    df = pd.read_csv(USER_DATA_FILE)
    hashed_password = hash_password(password)

    return ((df["Username"] == username) & (df["Password"] == hashed_password)).any()

# Login/Register Page
def login_page():
    st.title("Login / Register")

    option = st.radio("Select an option:", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        if st.button("Register"):
            if save_user(username, password):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists!")

    elif option == "Login":
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()

            else:
                st.error("Invalid username or password!")

# Main App - Diabetes Prediction
def main_app():
    st.sidebar.header("Model Selection")
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "Decision Tree", "MLP"))

    st.write("## Diabetes Dataset Preview")
    st.write(dataset.head())

    # Split Data
    X = dataset.drop(['Outcome'], axis=1)
    y = dataset['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Model Training
    def train_model(classifier_name):
        if classifier_name == "KNN":
            knn = KNeighborsClassifier(n_neighbors=9)
            knn.fit(X_train, y_train)
            return knn
        elif classifier_name == "Decision Tree":
            dt = DecisionTreeClassifier(random_state=0, max_depth=3)
            dt.fit(X_train, y_train)
            return dt
        else:  # MLP Classifier
            sc = StandardScaler()
            X_train_scaled = sc.fit_transform(X_train)
            X_test_scaled = sc.transform(X_test)
            mlp = MLPClassifier(random_state=42)
            mlp.fit(X_train_scaled, y_train)
            return mlp

    model = train_model(classifier_name)

    # Model Performance
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    st.write(f"### {classifier_name} Model Performance")
    st.write(f"Training Accuracy: {train_acc:.2f}")
    st.write(f"Testing Accuracy: {test_acc:.2f}")

    # Correlation Heatmap
    st.write("## Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    # Pregnancy Distribution by Outcome
    st.write("## Pregnancy Distribution by Outcome")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==1], color='Red', shade=True, ax=ax)
    sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==0], color='Blue', shade=True, ax=ax)
    ax.set_xlabel("Pregnancies")
    ax.set_ylabel("Density")
    ax.legend(["Positive", "Negative"])
    st.pyplot(fig)

    # User Input for Diabetes Prediction
    st.write("## Enter Your Details for Prediction")
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 150)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI", 0.0, 67.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 1, 100)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1][0] * 100  # Convert to percentage

        if prediction[0] == 1:
            st.error(f"**High Risk of Diabetes! (Probability: {probability:.2f}%)**")
        else:
            st.success(f"**Low Risk of Diabetes! (Probability: {probability:.2f}%)**")

# Streamlit App Flow
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    main_app()
