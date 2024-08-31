# Diabetes Prediction Project

This project utilizes various machine learning algorithms to predict whether a person has diabetes based on medical attributes. The dataset used in this project is the Pima Indians Diabetes Database.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn

### Installing
Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database. The dataset can be found in the file `diabetes.csv`.

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1) 0 means non-diabetic and 1 means diabetic

## Exploratory Data Analysis (EDA)

The dataset is explored through various visualizations:
- **Correlation heatmap**: Displays the correlations between different features.
- **KDE plot**: Kernel density estimation plots for pregnancies based on the outcome.
- **Violin plot**: Violin plots for glucose levels based on the outcome.

## Preprocessing

Before training the models, the following preprocessing steps are performed:
- Handling missing values by replacing zeros with the median or mean of the respective columns.
- Splitting the data into features (`X`) and labels (`y`).
- Splitting the dataset into training and testing sets using an 80-20 split.

## Model Training and Evaluation

Three different machine learning models are trained and evaluated:

1. **K-Nearest Neighbors (KNN)**
   - Evaluates accuracy for different values of `n_neighbors`.
   - Trains a final model with the best `n_neighbors`.

2. **Decision Tree**
   - Trains a decision tree classifier.
   - Also trains a decision tree with a maximum depth of 3 to prevent overfitting.

3. **Multi-Layer Perceptron (MLP)**
   - Trains a neural network without scaling.
   - Re-trains the neural network after scaling the features using `StandardScaler`.

### Training and Evaluation Script

To train and evaluate the models, run:

```python
python train_and_evaluate.py
```

## Results

The models are evaluated based on their accuracy on the training and testing sets:
- **KNN**: Reports the training and testing accuracy for different values of `n_neighbors`.
- **Decision Tree**: Reports the accuracy before and after limiting the tree depth.
- **MLP**: Reports the accuracy before and after scaling the data.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
