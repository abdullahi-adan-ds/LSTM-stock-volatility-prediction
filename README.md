# LSTM-stock-volatility-prediction
A deep learning project for financial time-series forecasting. Predicts the rolling volatility of QQQ and SPY ETFs using separate LSTM models built with TensorFlow. Features live data fetching (yfinance), comprehensive data preprocessing, and rigorous model evaluation, achieving an R² score of over 77% on test data.

# LSTM-Based Stock Volatility Prediction for QQQ & SPY ETFs

## Project Overview

This project aims to predict the rolling volatility of two major stock Exchange Traded Funds (ETFs): **QQQ** (Invesco QQQ Trust, tracking the NASDAQ-100 index) and **SPY** (SPDR S&P 500 ETF Trust, tracking the S&P 500 index). Two separate Long Short-Term Memory (LSTM) models are built and trained using live market data fetched from Yahoo Finance.

The primary goal is to demonstrate the application of deep learning techniques in financial time-series forecasting, specifically for a regression task where the target variable, **volatility**, is continuous. The project is structured as a supervised learning problem, where the models learn from historical features to predict future outcomes.

---

## Features

* **Live Data Integration**: Fetches the latest 6 years of stock data for QQQ and 5 years for SPY directly from Yahoo Finance (`yfinance`).
* **Comprehensive Data Preprocessing**: Includes robust data cleaning, handling of missing values, removal of duplicates, and validation checks to ensure data quality.
* **Advanced Feature Engineering**: Creates a rich set of features to improve model performance, including:
    * **Returns**: Percentage change in closing prices.
    * **Rolling Volatility**: 5-day rolling standard deviation of returns.
    * **Simple Moving Averages (SMA)**: 10-day and 20-day SMAs.
    * **Momentum Indicators**: 5-day and 10-day price momentum.
    * **Lagged Features**: Lagged values of volatility and returns to capture temporal dependencies.
* **Dual LSTM Models**: Two independent LSTM models are built, trained, and evaluated—one for each ETF (QQQ and SPY).
* **Rigorous Model Evaluation**: The models are evaluated using standard regression metrics: **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R-squared (R²)**.
* **Overfitting Mitigation**: Dropout layers are incorporated into the LSTM architecture to prevent overfitting and improve generalization.
* **Real-Time Prediction Simulation**: A function simulates real-time volatility predictions to showcase the model's practical application.

---

## Methodology

### 1. Data Gathering and Preprocessing

Historical stock data (Open, High, Low, Close, Volume) is loaded using the `yfinance` library. The data undergoes a thorough cleaning process where duplicates are removed, chronological order is ensured, and missing values are forward-filled. Validation checks are performed to handle non-positive prices and zero trading volumes.

### 2. Feature Engineering

Additional features are engineered to provide the models with more context. These include returns, rolling volatility (the target variable), moving averages, price ratios, and lagged features. These features are designed to capture trends, momentum, and historical patterns that influence future volatility.

### 3. Model Selection: LSTM

An **LSTM (Long Short-Term Memory)** network was chosen for this task due to its effectiveness in handling time-series data. LSTMs are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies, making them ideal for financial forecasting where past events can influence future outcomes.

The problem is framed as a **supervised regression task**, where the model learns to map a sequence of historical features to a continuous volatility value.

### 4. Model Architecture

Two identical LSTM models are built, one for each stock. The architecture consists of:
* Two LSTM layers with 64 units each.
* Dropout layers with a rate of 0.2 to prevent overfitting.
* A Dense layer with 25 units and a ReLU activation function.
* An output Dense layer with a single unit for the regression output.

The models are compiled using the **Adam optimizer** with a learning rate of 0.001 and **Mean Squared Error (MSE)** as the loss function.

---

## Results and Evaluation

Both models demonstrated strong predictive performance on the test datasets. The final evaluation metrics are summarized below:

| Metric | QQQ Test | SPY Test |
| :--- | :---: | :---: |
| **RMSE** | 0.009667 | 0.008940 |
| **R²** | 0.7727 | 0.7876 |

The **R² values** indicate that the models can explain approximately **77%** of the variance in QQQ's volatility and **79%** of the variance in SPY's volatility, which is a strong result for financial forecasting.

### Visualizations

#### Feature Relationships
The scatter plots below show a mild positive correlation between returns and price ratio for both ETFs, with QQQ exhibiting slightly more variance.

---

#### Feature-Target Correlation
The heatmaps confirm that lagged volatility features are the strongest predictors for the current volatility.

---

#### Model Performance and Overfitting Check
The training and validation loss curves show good convergence without significant overfitting. The predicted vs. actual plots demonstrate that the models' predictions closely follow the actual volatility values.

---

#### Time Series Prediction
The models are effective at tracking the actual volatility trends over time, capturing both peaks and troughs with reasonable accuracy.

---

## How to Use

To run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/lstm-volatility-prediction.git](https://github.com/your-username/lstm-volatility-prediction.git)
    cd lstm-volatility-prediction
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebook:**
    The `LSTM_Final.ipynb` notebook contains the entire workflow, from data loading to model evaluation. You can run the cells sequentially to reproduce the results.

4.  **Exported Assets:**
    * `qqq_model.keras` & `spy_model.keras`: The trained LSTM models.
    * `qqq_feature_scaler.pkl`, `qqq_target_scaler.pkl`, etc.: The scalers used for preprocessing.
    * `model_metrics.json`: A JSON file containing the final evaluation metrics.
