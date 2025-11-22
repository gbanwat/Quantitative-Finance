
# 📈 Moving Average Crossover Backtesting System

🔗 **Project Notebook:**  
https://github.com/gbanwat/Quantitative-Finance/blob/master/MA-Crossover%20Backtesting%20Engine.ipynb


*A Python-based backtesting engine for multi-asset SMA crossover trading strategies.*

---

## 🚀 Overview

This project implements a complete **algorithmic trading backtester** using Python.
It downloads historical stock data, generates SMA crossover signals, simulates trades, computes portfolio performance metrics, and plots the equity curve.

The system supports:

* Multiple tickers
* SMA50 / SMA200 crossover strategy
* Trade signal generation (BUY/SELL)
* Portfolio value tracking
* Sharpe ratio, drawdowns & cumulative return
* Automated trade logs
* Equity curve visualization

---

## 📘 Features

### ✔ 1. Data Download

Historical stock prices are fetched using **Yahoo Finance (`yfinance`)**.

---

### ✔ 2. Trading Strategy — SMA Crossover

Uses a **Simple Moving Average (SMA)** crossover technique:

* **BUY** → When **SMA50** crosses above **SMA200**
* **SELL** → When **SMA50** crosses below **SMA200**

---

### ✔ 3. Backtesting Engine

The backtester computes:

* Daily holdings
* Cash balance
* Trade value
* Total portfolio value
* Net percentage return

---

### ✔ 4. Performance Metrics

Includes:

* **Sharpe Ratio**
* **Cumulative Return**
* **Maximum Drawdown**

---

### ✔ 5. Trade Log

Generates a clear BUY/SELL/HOLD log for all position changes.

---

### ✔ 6. Equity Curve Plot

Visualizes:

* Overall portfolio value
* Individual ticker price movements

---


