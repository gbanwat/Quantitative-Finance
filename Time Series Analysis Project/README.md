# 📈 Time Series Stock Analysis Project

## 📂 Files

- **[`tsfunct.py`](https://github.com/gbanwat/Quantitative-Finance/blob/main/Time%20Series%20Analysis%20Project/tsfunct.py)**  
  Contains all the core functions for time series analysis, including:
  - Stock data fetching from Yahoo Finance  
  - Plotting stock price charts  
  - Time series decomposition (classical & STL)  
  - Stationarity checks and transformations  
  - ARIMA modeling and forecasting  
  These functions are designed for **direct reuse** in other projects.

- **[`Time Series Modelling Project.ipynb`](https://github.com/gbanwat/Quantitative-Finance/blob/main/Time%20Series%20Analysis%20Project/Time%20Series%20Modelling%20Project.ipynb)**  
  A Jupyter Notebook demonstrating the **practical application** of the functions defined in `tsfunct.py`.  
  It performs a complete **time series analysis of stock data**, including:
  - Data visualization  
  - Decomposition and stationarity checks  
  - Transformation to achieve stationarity  
  - ARIMA modeling, prediction, and forecasting  
  

## 📝 Overview
This project provides a comprehensive **Python-based framework for analyzing, visualizing, and forecasting stock prices** using time series methods. It integrates data retrieval, statistical analysis, decomposition, stationarity tests, transformations, and ARIMA modeling to support both exploratory analysis and predictive modeling.

The framework is ideal for **quantitative finance**, **algorithmic trading**, or **financial data analysis**.

---

## 🚀 Features

### 1. Data Acquisition
- Fetches historical stock data from Yahoo Finance (`yfinance`).
- Prints summary statistics, major holders, and institutional holders.
- Returns the stock ticker name with closing prices.

### 2. Visualization
- Plots stock closing price time series with properly formatted dates.
- Generates decomposition plots (trend, seasonal, residual) for time series analysis.
- Plots stationary time series and ACF/PACF diagrams for ARIMA modeling.

### 3. Time Series Decomposition
- Classical decomposition (`additive` or `multiplicative`).
- STL decomposition (seasonal-trend decomposition using Loess).
- Visualizes trend, seasonality, and residual components.

### 4. Stationarity Analysis
- Performs Augmented Dickey-Fuller (ADF) and KPSS tests.
- Checks stationarity type using Kolmogorov-Smirnov test.
- Provides recommendations for required transformations.
- Iteratively applies transformations to make series stationary (log, sqrt, Box-Cox, detrending, differencing, seasonal adjustment).

### 5. ARIMA Modeling & Forecasting
- Fits ARIMA models on training data.
- Predicts and visualizes test data against ARIMA forecasts.
- Computes RMSE for prediction accuracy.
- Forecasts future stock prices and plots them with recent actual values.

---

## 🛠️ Libraries Used
- `yfinance` – for stock data retrieval
- `pandas`, `NumPy` – for data manipulation
- `matplotlib` – for visualization
- `statsmodels` – for statistical tests and ARIMA modeling
- `scipy` – for Box-Cox transformation
- `sklearn` – for linear detrending

---

