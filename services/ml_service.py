import numpy as np
from sklearn.linear_model import LinearRegression

def predict_next_day(data):
    data = data.copy()
    data["Day"] = range(len(data))

    X = data[["Day"]]
    y = data["Close"]

    model = LinearRegression()
    model.fit(X, y)

    next_day = [[len(data)]]
    prediction = model.predict(next_day)[0]
    return round(prediction, 2)