# Diabetes Prediction Project

This project is a **web-based diabetes prediction system** using **machine learning** with **Streamlit**. It allows users to **register, log in, explore the dataset, visualize trends, and predict diabetes risk** based on their medical attributes. The dataset used is the **Pima Indians Diabetes Database**.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [User Authentication](#user-authentication)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
Ensure you have Python 3.x installed and the following libraries:
- `streamlit`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `hashlib`
- `os`

### Installing
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
```

## Features

✅ **User Authentication** (Register/Login system with hashed passwords)  
✅ **Diabetes Prediction** using **KNN, Decision Tree, and MLP classifiers**  
✅ **Dataset Visualization** (Heatmaps, KDE plots, and more)  
✅ **Real-time Model Selection**  
✅ **Custom User Inputs for Prediction**  

## Dataset

The dataset used is the **Pima Indians Diabetes Database** (`diabetes.csv`). It contains the following attributes:

- **Pregnancies**: Number of times pregnant  
- **Glucose**: Plasma glucose concentration  
- **BloodPressure**: Diastolic blood pressure  
- **SkinThickness**: Triceps skinfold thickness  
- **Insulin**: 2-Hour serum insulin  
- **BMI**: Body mass index  
- **DiabetesPedigreeFunction**: Diabetes likelihood based on family history  
- **Age**: Age in years  
- **Outcome**: 0 (Non-diabetic) or 1 (Diabetic)  

## Exploratory Data Analysis (EDA)

The app provides interactive **visualizations**, including:
- **Correlation Heatmap**: Displays relationships between attributes.
- **KDE Plot**: Density estimation of pregnancies by diabetes outcome.
- **Data Preview**: View the first five rows of the dataset.

## Preprocessing

Preprocessing steps include:
- **Handling missing values**: Replacing zeros in `Glucose`, `BloodPressure`, `BMI`, etc., with median or mean values.
- **Splitting data** into **training (67%)** and **testing (33%)** sets.

## Model Training and Evaluation

Three **ML models** are available:
1. **K-Nearest Neighbors (KNN)**:  
   - Uses `n_neighbors=9`
   - Computes accuracy on training/testing data

2. **Decision Tree**:  
   - A depth-limited (`max_depth=3`) model to avoid overfitting  

3. **Multi-Layer Perceptron (MLP)**:  
   - Uses a standard neural network (`MLPClassifier`)
   - Data is **standardized using StandardScaler** before training

Each model reports **training and testing accuracy**.

## User Authentication

- **New users** can register and have their passwords securely stored using **SHA-256 hashing**.
- **Existing users** can log in and access diabetes prediction features.
- **Session state** is used to maintain login status.

## Results

Each classifier provides **accuracy scores** based on training/testing data.  
The app also displays **the probability of diabetes risk** based on user inputs.

## Usage

To run the application:

```bash
streamlit run app.py
```

## Contributing

Contributions are welcome!  
- Fork the repository  
- Create a new branch (`git checkout -b feature-branch`)  
- Commit your changes (`git commit -m "Added feature XYZ"`)  
- Push to the branch (`git push origin feature-branch`)  
- Submit a **pull request**  

## License

This project is licensed under the **MIT License**.

---
