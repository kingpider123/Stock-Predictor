# Stock-Predictor

Basically try to implement 6 different approaches from traditional regression to LSTM neural networks.

## Models Implemented

### 1. **Linear Regression**
- **Objective**: Predict exact closing price
- **Features**: Open, High, Low, Close, Trading Volume
- **Outputs**: MAE and R² score
- **Best for**: Baseline comparison

### 2. **Naive Bayes Classifier**
- **Objective**: Classify price direction (UP/DOWN)
- **Features**: Open, High, Low, Close, Trading Volume
- **Outputs**: Accuracy, classification report, confusion matrix
- **Best for**: Fast, probabilistic trend prediction

### 3. **Neural Network (MLP)**
- **Objective**: Trend classification with engineered features
- **Architecture**: Two hidden layers (64, 32 neurons)
- **Key Features**:
  - Daily Return (percentage change)
  - Simple Moving Averages (SMA_5, SMA_20)
  - Volatility Index
  - Volume Change
- **Special**: Scaled data using StandardScaler
- **Outputs**: Accuracy and classification report

### 4. **Autoregressive Linear Regression**
- **Objective**: Predict exact closing price using price history
- **Features**: Lag_1, Lag_2, Lag_3 (past 3 days of closing prices)
- **Approach**: Time-series specific model capturing sequential dependencies
- **Outputs**: MAE and R² score

### 5. **LSTM Model 1** (60-Day Window)
- **Objective**: Predict closing price with long-term memory
- **Memory Span**: 60-day sliding window
- **Features**: Univariate (close price only)
- **Architecture**: 
  - LSTM layer (50 units) + Dropout (0.2)
  - LSTM layer (50 units) + Dropout (0.2)
  - Dense output layer
- **Special Feature**: Early stopping to prevent overfitting
- **Outputs**: MAE (real dollars), prediction plot, MAPE & directional accuracy

### 6. **LSTM Model 2** (1-Day Window, Multivariate)
- **Objective**: Predict closing price with multiple features
- **Memory Span**: 1-day window
- **Features**: Open, High, Low, Close, Trading Volume (5 variables)
- **Architecture**:
  - LSTM layer (64 units, dropout 0.2, return_sequences=True)
  - LSTM layer (64 units, dropout 0.2)
  - Dense layer (16 units, ReLU activation)
  - Dense output layer
- **Fixed Epochs**: 10 (no early stopping)
- **Outputs**: Prediction plot with train/test split

## Dataset: TSMC, ticker 2330 daily trading data starting from 2000

**File**: `data/stock.csv`

**Columns**: 
- `date` - Trading date
- `open` - Opening price
- `max` - Highest price of the day
- `min` - Lowest price of the day
- `close` - Closing price
- `Trading_Volume` - Volume of shares traded

## Data Preprocessing

- **Date Handling**: Converted to datetime objects and sorted chronologically
- **Target Variables**:
  - `Target_Price`: Tomorrow's closing price (for regression)
  - `Target_Trend`: Binary classification (1=UP, 0=DOWN)
- **Train/Test Split**: 80/20 chronological split (preserves temporal order)
- **Feature Engineering**: Moving averages, volatility, returns, volume changes
- **Scaling**: MinMaxScaler (0-1 range) for LSTM, StandardScaler for neural networks

## Evaluation Metrics

### Regression Models
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **R² Score**: Variance explained by the model (range: 0-1)

### Classification Models
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives
- **Classification Report**: Precision, recall, and F1-score

### LSTM Models
- **MAPE (Mean Absolute Percentage Error)**: Percentage accuracy of price predictions
- **Directional Accuracy**: Percentage of correctly predicted price trends (UP/DOWN)

## Installation

```bash
pip install pandas scikit-learn tensorflow keras matplotlib seaborn numpy