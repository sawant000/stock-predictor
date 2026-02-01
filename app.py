from flask import Flask, render_template, request
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.linear_model import LinearRegression
from services.stock_service import get_stock_prediction
import numpy as np
import os
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    price = None
    predicted_price = None
    rsi = None
    signal = None
    graph_path = None

    if request.method == "POST":
        symbol = request.form["symbol"].upper()
        try:
            price, predicted_price, rsi, signal, graph_path = get_stock_prediction(symbol)
        except Exception as e:
            print("Error:", e)

    return render_template(
        "index.html",
        price=price,
        predicted_price=predicted_price,
        rsi=rsi,
        signal=signal,
        graph_path=graph_path
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)