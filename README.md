# Apple Stock Data Analysis

This project analyzes Apple stock data using machine learning models to predict stock prices. It includes data preprocessing, exploratory data analysis (EDA), and the implementation of regression models such as Random Forest Regression and Linear Regression

## Features

- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA) with visualizations
- Implementation of machine learning models:
  - Random Forest Regression
- Evaluation metrics for model performance:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - R² Score

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/apple-stock-data-analysis.git
   ```
2. Install the required dependencies:
    ```sh
    pip install scikit-learn, pandas, numpy, seaborn, matplotlib
    ```

## Usage

1. Open the Jupyter Notebook

    ```sh
    jupyter notebook Apple Stock Data Analysis.ipynb
    ```
2. Run the cells sequentially to preprocess the data, train the model, and evaluate its performance.

## Project Structure

- `Apple Stock Data Analysis.ipynb`: Main notebook containing the code for data analysis and model training.

## Evaluation Metrics

The Random Forest Regression model is evaluated using the following metrics:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R² Score**: Coefficient of determination

## Results

The performance of the Random Forest Regression model and linear regression model is printed in the notebook, including metrics such as MSE, RMSE, MAE, MAPE, and R² Score.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (optional, for visualizations)
