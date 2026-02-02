import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import os

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_stock_prediction(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="6mo")

    if df.empty:
        raise Exception("Invalid stock symbol")

    # Indicators
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df.dropna(inplace=True)

    # ML data
    X = df[["Close", "MA20", "MA50"]]
    y = df["Close"].shift(-1)

    X = X[:-1]
    y = y[:-1]

    model = LinearRegression()
    model.fit(X, y)

    current_price = float(df["Close"].iloc[-1])
    current_rsi = float(df["RSI"].iloc[-1])

    # 🔮 7-DAY FORECAST
    last_row = X.iloc[-1].values.reshape(1, -1)
    predictions = []

    for _ in range(7):
        next_price = model.predict(last_row)[0]
        predictions.append(next_price)

        # update input for next iteration
        last_row = [[next_price, last_row[0][1], last_row[0][2]]]

    avg_predicted_price = sum(predictions) / len(predictions)

    # 📊 SIGNAL LOGIC
    if avg_predicted_price > current_price and current_rsi < 30:
        signal = "STRONG BUY"
    elif avg_predicted_price < current_price and current_rsi > 70:
        signal = "STRONG SELL"
    elif avg_predicted_price > current_price:
        signal = "BUY"
    elif avg_predicted_price < current_price:
        signal = "SELL"
    else:
        signal = "HOLD"

    # 📈 Plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(df.index, df["Close"], label="Close", linewidth=2)
    ax1.plot(df.index, df["MA20"], "--", label="MA20")
    ax1.plot(df.index, df["MA50"], "--", label="MA50")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.3)

    # BUY / SELL Arrow
    last_date = df.index[-1]
    last_price = df["Close"].iloc[-1]

    if "BUY" in signal:
        ax1.annotate("BUY", xy=(last_date, last_price),
                     xytext=(last_date, last_price * 0.97),
                     arrowprops=dict(facecolor="green", arrowstyle="->"),
                     color="green", fontweight="bold")
    elif "SELL" in signal:
        ax1.annotate("SELL", xy=(last_date, last_price),
                     xytext=(last_date, last_price * 1.03),
                     arrowprops=dict(facecolor="red", arrowstyle="->"),
                     color="red", fontweight="bold")

    # RSI
    ax2.plot(df.index, df["RSI"], color="purple")
    ax2.axhline(70, linestyle="--", color="red")
    ax2.axhline(30, linestyle="--", color="green")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("static/graph.png")
    plt.close()

    return (
        round(current_price, 2),
        round(avg_predicted_price, 2),
        round(current_rsi, 2),
        signal,
        "static/graph.png"
    )