# Adult Income Classification using SVM and Logistic Regression

## Project Overview
This project aims to predict whether an individual's annual income exceeds $50,000 based on census data, also known as the "Adult Census Income" dataset. The classification is performed using two machine learning models: **Support Vector Machines (SVM)** and **Logistic Regression**. The project is implemented in Python using libraries such as scikit-learn, pandas, and seaborn for data processing, model training, and visualization.

The notebook (`Adult_Income_classification using SVM_LogisticRegression.ipynb`) includes data ingestion, exploratory data analysis (EDA), data preprocessing, model training, hyperparameter tuning, and performance evaluation using metrics like ROC AUC, confusion matrix, and classification reports.

## Dataset
The dataset used is the **Adult Census Income dataset** from the UCI Machine Learning Repository. It contains 32,561 records with 15 features, including:

- **Numerical features**: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- **Categorical features**: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Target variable**: income (binary: `<=50K` or `>50K`)

**Source**:
- [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Adult)
- [Kaggle](https://www.kaggle.com/overload10/adult-census-dataset/tasks)

## Requirements
To run the notebook, you need the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/adult-income-classification.git
   cd adult-income-classification
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   (Create a `requirements.txt` file with the above libraries if needed.)
3. Download the dataset (`adult_data.txt`) from the UCI repository or Kaggle and place it in the project directory.
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook Adult_Income_classification using SVM_LogisticRegression.ipynb
   ```

## Usage
1. Open the Jupyter notebook in your environment.
2. Ensure the dataset file (`adult_data.txt`) is in the same directory as the notebook.
3. Run the notebook cells sequentially to:
   - Load and preprocess the data
   - Perform exploratory data analysis (EDA)
   - Train and evaluate SVM and Logistic Regression models
   - Visualize model performance (e.g., ROC curves, confusion matrices)
4. Modify hyperparameters or experiment with other models by editing the relevant cells.

## Project Structure
- `Adult_Income_classification using SVM_LogisticRegression.ipynb`: Main notebook with the complete workflow.
- `adult_data.txt`: Dataset file (not included; download from UCI or Kaggle).
- `README.md`: Project documentation.

## Methodology
1. **Data Ingestion**: Load the dataset using pandas and assign appropriate column names.
2. **Exploratory Data Analysis (EDA)**: Analyze data distribution, check for missing values, and visualize income distribution.
3. **Data Preprocessing**:
   - Encode categorical variables using `LabelEncoder`.
   - Scale numerical features using `StandardScaler`.
   - Split data into training and testing sets.
4. **Model Training**:
   - Train Logistic Regression and SVM models.
   - Use `GridSearchCV` or `RandomizedSearchCV` for hyperparameter tuning.
5. **Evaluation**:
   - Compute ROC AUC scores, confusion matrices, and classification reports.
   - Compare model performance using visualizations.

## Results
- The notebook includes a comparison of ROC AUC scores for both models.
- Visualizations such as bar plots and ROC curves are provided to assess model performance.
- No missing values were found in the dataset, simplifying preprocessing.



## Acknowledgments
- UCI Machine Learning Repository for providing the dataset.
- Scikit-learn documentation for model implementation guidance.
- Kaggle community for insights and discussions.
