# 📈 BharatTrade AI

**Intelligent Stock Movement Prediction & Paper Trading System**

BharatTrade AI is an end-to-end quantitative trading and paper-trading platform built specifically for Indian Equities (NSE). It processes high-frequency intraday market data (15-minute candles) using a hybrid deep learning model (Conv1D + LSTM ensemble) to generate actionable signal classifications (Buy, Sell, Hold) accompanied by strict cost-aware risk management.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Project Workflow](#-project-workflow)
- [Evaluation Metrics](#-evaluation-metrics)
- [References](#-references)
- [License](#-license)

---

## 🔍 Overview

The Indian stock market generates large volumes of continuously changing price and volume data. Most retail traders either rely on static technical indicators or simple price prediction models without integrating realistic execution constraints.

**BharatTrade AI** bridges this gap by combining:
1. **Intraday Feature Extraction:** Processes 15-minute OHLCV candles enriched with technical indicators.
2. **Deep Learning Ensemble:** Combines Conv1D (spatial patterns) and LSTM (temporal dependencies).
3. **Execution Engine:** Evaluates trades using slippage, transaction costs, and automated risk management (stop-loss and position limits).
4. **Interactive Dashboard:** Real-time visual tracking of signals, virtual portfolios, and equity curves.

---

## 🌟 Key Features

* **Intraday Data Processing:** Preprocesses 15-minute historical candles from selected NSE equities (e.g., NIFTY 50 cash/futures).
* **Hybrid Neural Architecture:**
  * **Conv1D Layer:** Identifies local spatial patterns, including short-term momentum shifts and volume-price interactions.
  * **LSTM Layer:** Learns long-term temporal sequence dependencies across trading sessions.
  * **Weighted Ensemble:** Blends predicted class probabilities to yield high-confidence Buy, Sell, or Hold signals.
* **Feature Engineering:** Extracts technical metrics including RSI, MACD, Moving Averages, Average True Range (ATR), Price Range, and Volume Volatility.
* **Cost-Aware Backtesting:** Accounts for realistic transaction friction, including brokerage fees, taxes, and slippage.
* **Virtual Paper Trading:** Monitors position sizing, trailing stop-losses, virtual balance curves, and risk exposure in real time.

---

## 🏗️ System Architecture

                  +-----------------------------+
                  |   NSE Historical Data       |
                  |   (15-min OHLCV Candles)    |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |   Data Preprocessing &      |
                  |   Feature Engineering       |
                  | (RSI, MACD, ATR, MA, etc.)  |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |   Sliding Window Sequence   |
                  |   Train / Val / Test Split  |
                  +--------------+--------------+
                                 |
                +----------------+----------------+
                |                                 |
                v                                 v
      +-------------------+             +-------------------+
      |    Conv1D Model   |             |    LSTM Model     |
      | (Spatial Patterns)|             | (Temporal Memory) |
      +---------+---------+             +---------+---------+
                |                                 |
                +----------------+----------------+
                                 |
                                 v
                  +-----------------------------+
                  |  Weighted Ensemble Engine   |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  | Decision & Signal Generation|
                  |     (BUY / SELL / HOLD)     |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |  Risk Manager & Paper Trader|
                  | (Stop-loss, Fees, Equity)   |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  | Streamlit / Plotly Dashboard|
                  +-----------------------------+


## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Data Processing & Analytics:** Pandas, NumPy
* **Deep Learning Framework:** TensorFlow / Keras
* **Interactive Visualization:** Streamlit, Plotly
* **Storage:** CSV / PostgreSQL
* **Broker & Data Interfaces:** Zerodha Kite API / Upstox API / Yahoo Finance


## 💻 Installation & Setup

### Prerequisites

Ensure you have Python 3.10+ installed on your system.

### Step-by-Step Instructions

1. **Clone the Repository:**
   git clone [https://github.com/ojass404/BharatTrade-AI.git](https://github.com/ojass404/BharatTrade-AI.git)
   cd BharatTrade-AI
   
3. **Create a Virtual Environment:**
   python -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   .venv\Scripts\activate           # Windows

4. **Install Required Packages:**
   pip install -r requirements.txt

5. **Launch the Dashboard:**
   streamlit run app.py


## 🔄 Project Workflow

1. **Data Collection:** Retrieve historical 15-minute OHLCV candles from NSE brokers or financial APIs[cite: 1].
2. **Preprocessing:** Clean missing values, scale features using `MinMaxScaler` / `StandardScaler`, and compute technical indicators[cite: 1].
3. **Sequence Windowing:** Apply a sliding-window transform to frame market history as time-series sequence tensors[cite: 1].
4. **Model Training:** Train Conv1D and LSTM architectures on a chronological 70/15/15 train/validation/test split[cite: 1].
5. **Ensembling:** Combine class probabilities from both models using validation-optimized weights[cite: 1].
6. **Execution Simulation:** Route generated signals through the risk engine to simulate execution against transaction costs and stop-loss bounds[cite: 1].

---

## 📊 Evaluation Metrics

The system assesses performance using both machine learning and financial benchmarks[cite: 1]:

| Category | Metric | Description |
| :--- | :--- | :--- |
| **Model Performance**[cite: 1] | **Precision, Recall, F1-Score**[cite: 1] | Class-wise prediction accuracy for Buy/Sell/Hold signals[cite: 1]. |
| | **ROC-AUC & Confusion Matrix**[cite: 1] | Classification thresholding and confusion distribution[cite: 1]. |
| **Financial Backtest**[cite: 1] | **Net Return**[cite: 1] | Total profit/loss after accounting for fees and slippage[cite: 1]. |
| | **Sharpe Ratio**[cite: 1] | Risk-adjusted return profile[cite: 1]. |
| | **Max Drawdown**[cite: 1] | Largest peak-to-trough decline in account balance[cite: 1]. |
| | **Profit Factor**[cite: 1] | Ratio of gross profits to gross losses[cite: 1]. |
| | **Buy & Hold Comparison**[cite: 1] | Benchmarked against passive holding of the target asset[cite: 1]. |
