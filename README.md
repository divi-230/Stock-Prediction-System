# AI Stock Prediction System

An AI-powered stock prediction web application built using Python, Flask, Machine Learning, and Deep Learning. The system allows users to search for companies, fetch real-time stock market data, visualize trends, analyze technical indicators, and predict future stock prices using an LSTM (Long Short-Term Memory) neural network.

---

# Features

* Real-time stock data using Yahoo Finance
* Company name to stock symbol detection
* Stock price visualization using Chart.js
* Technical Indicators:

  * RSI (Relative Strength Index)
  * MA5 (5-Day Moving Average)
  * MA10 (10-Day Moving Average)
  * Bollinger Bands
  * Volatility
  * Volume Ratio
* LSTM-based stock prediction model
* Interactive Flask web interface
* Prediction accuracy evaluation
* Dynamic charts and graphs
* Multiple company support
* Modern responsive UI

---

# Technologies Used

## Backend

* Python
* Flask
* TensorFlow / Keras
* scikit-learn
* NumPy
* pandas
* yfinance

## Frontend

* HTML
* CSS
* JavaScript
* Chart.js

## Machine Learning

* LSTM (Long Short-Term Memory)
* Data Scaling using MinMaxScaler
* Model Evaluation Metrics

---

# Project Structure

```text
AI-Stock-Prediction-System/
│
├── app.py
├── requirements.txt
├── model_tcs_ns.keras
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── datasets/
│
└── README.md
```

---

# How the System Works


## 1. Symbol Detection 

The system detects the correct stock symbol.

## 2. Data Collection

Stock market data is fetched using:

```python
import yfinance as yf
```

The system downloads:

* Open Price
* Close Price
* High Price
* Low Price
* Volume
* Historical Trends

---

## 3. Data Preprocessing

The collected stock data is:

* Cleaned
* Normalized
* Converted into sequences
* Prepared for LSTM training

Using:

```python
from sklearn.preprocessing import MinMaxScaler
```

---

## 4. LSTM Model Training

The model learns patterns from historical stock data.

The LSTM network helps in:

* Time-series forecasting
* Trend analysis
* Future price prediction

Example:

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1))
```

---

## 5. Prediction

The trained model predicts future stock prices.

The predicted prices are displayed on:

* Line Charts
* Prediction Graphs
* Technical Analysis Dashboard

---

# Installation Guide

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AI-Stock-Prediction-System.git
```

---

## Step 2: Open Project Folder

```bash
cd AI-Stock-Prediction-System
```

---

## Step 3: Create Virtual Environment

```bash
python -m venv venv
```

---

## Step 4: Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```text
Flask
numpy
pandas
yfinance
scikit-learn
tensorflow
matplotlib
```

---

# Run the Application

```bash
python app.py
```

Flask server will start:

```text
http://127.0.0.1:5000
```

Open it in your browser.

---

# Example Workflow


## System detects symbol:

```text
AAPL
```

## Application:

* Downloads stock data
* Processes data
* Runs prediction model
* Displays charts
* Shows predicted price

---

# Technical Indicators Explained

## RSI (Relative Strength Index)

Used to identify:

* Overbought stocks
* Oversold stocks

---

## MA5 & MA10

Moving averages used to analyze short-term trends.

---

## Bollinger Bands

Used to measure:

* Volatility
* Market movement range

---

## Volatility

Measures how much stock prices fluctuate.

---

## Volume Ratio

Analyzes buying and selling activity.

---

# Model Evaluation Metrics

## RMSE

Root Mean Square Error.
Measures prediction error.

---

## MAPE

Mean Absolute Percentage Error.
Measures prediction accuracy in percentage.

---

## Directional Accuracy

Checks whether the model correctly predicts:

* Upward trend
* Downward trend

---

# Future Improvements

* Multi-stock comparison
* News sentiment analysis
* AI chatbot integration
* Portfolio management
* Live market updates
* Candlestick charts
* Cryptocurrency prediction
* User authentication system
* Database integration

---

# Screenshots

Add screenshots of:

* Home Page
![Home Page](screenshots/home.png)

* Prediction Results
![Dashboard](screenshots/result.png)


# Learning Outcomes

This project demonstrates:

* Flask Web Development
* Machine Learning Concepts
* Deep Learning with LSTM
* Data Preprocessing
* Financial Data Analysis
* Time-Series Forecasting
* Frontend + Backend Integration
* Real-Time Data Handling

---

# Author

Developed by Diva

---

# License

This project is developed for educational and learning purposes.

---

