# -*- coding: utf-8 -*-
import datetime
import json
import numpy as np
import pandas as pd
import requests
import time
import warnings

warnings.simplefilter("ignore")
import yfinance as yf
import pandas as pd
import talib as ta
from talib import MA_Type

"""## Data Processing"""

# Define the Dow 30 tickers as of October 2014
Ticker_List = [
    "MSFT",
    "AAPL",
    "AMZN",
    "JPM",
    "V",
    "WMT",
    "UNH",
    "PG",
    "JNJ",
    "HD",
    "MRK",
    "CVX",
    "CRM",
    "KO",
    "DIS",
    "MCD",
    "CSCO",
    "CAT",
    "AXP",
    "VZ",
    "IBM",
    "AMGN",
    "NKE",
    "GS",
    "INTC",
    "HON",
    "BA",
    "MMM",
    "TRV",
    "DOW",
]

# Define the date range
start_date = "2019-04-30"
end_date = "2024-04-30"

# Dictionary to hold the dataframes for each stock
Stock_Data = {}

# Download the historical data for each stock
for ticker in Ticker_List:
    # Using yfinance to download the data
    Stock_df = yf.download(ticker, start=start_date, end=end_date)
    # Select only the required columns
    Stock_Data[ticker] = Stock_df[["Open", "High", "Low", "Close", "Volume"]]

Stock_Data["AAPL"].head()  # Show an example dataframe for Apple Inc.

# 你要預測幾天以後的漲跌
f = 1
print("you want to forcast the stock go up or down in the next", f, "days")

# 我不確定這裡面有沒有look ahead bias的問題，還需要點時間figure it out
for i in Ticker_List:

    Stock_Data[i]["High Shifted"] = Stock_Data[i]["High"].shift(-f)
    Stock_Data[i]["Low Shifted"] = Stock_Data[i]["Low"].shift(-f)
    Stock_Data[i]["Close Shifted"] = Stock_Data[i]["Close"].shift(-f)

    (
        Stock_Data[i]["Upper BBand"],
        Stock_Data[i]["Middle BBand"],
        Stock_Data[i]["Lower BBand"],
    ) = ta.BBANDS(
        Stock_Data[i]["Close Shifted"],
        timeperiod=20,
    )

    Stock_Data[i]["RSI"] = ta.RSI(
        np.array(Stock_Data[i]["Close Shifted"]), timeperiod=14
    )

    Stock_Data[i]["Macd"], Stock_Data[i]["Macd Signal"], Stock_Data[i]["Macd Hist"] = (
        ta.MACD(
            Stock_Data[i]["Close Shifted"], fastperiod=12, slowperiod=26, signalperiod=9
        )
    )

    Stock_Data[i]["Momentum"] = ta.MOM(Stock_Data[i]["Close Shifted"], timeperiod=12)

    Stock_Data[i]["Returns"] = np.log(
        Stock_Data[i]["Open"] / Stock_Data[i]["Open"].shift(-f)
    )
    # 這裡的return為log return

for i in Ticker_List:
    Signal_List = []
    for j in Stock_Data[i]["Returns"]:

        if j >= 0:
            Signal_List.append("1")

        else:
            Signal_List.append("0")

    Stock_Data[i]["Signal"] = Signal_List

Stock_Data["AAPL"]

"""## Fitting the Model"""

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

max_abs_scaler = preprocessing.MaxAbsScaler()

Model_Dict = {}

for i in Ticker_List:
    Stock_Data[i].dropna(inplace=True)

    X = np.array(Stock_Data[i].drop(["Signal", "Returns"], axis=1))
    X = max_abs_scaler.fit_transform(X)
    Y = np.array(Stock_Data[i]["Signal"])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    Model_Dict[i] = {}
    Model_Dict[i]["X Train"] = X_train
    Model_Dict[i]["X Test"] = X_test
    Model_Dict[i]["Y Train"] = y_train
    Model_Dict[i]["Y Test"] = y_test

    model = svm.SVC(kernel="rbf", decision_function_shape="ovo")
    # model = svm.SVC(kernel='linear')
    # model = svm.SVC(kernel='linear',decision_function_shape='ovo')
    # model = svm.SVC(kernel='rbf',decision_function_shape='ovo')
    # model = svm.SVC(kernel='poly')
    # model = svm.SVC(kernel='poly',decision_function_shape='ovo')
    # model = svm.SVC(kernel='sigmoid')
    # model = svm.SVC(kernel='sigmoid',decision_function_shape='ovo')

    model.fit(Model_Dict[i]["X Train"], Model_Dict[i]["Y Train"])
    y_pred = model.predict(Model_Dict[i]["X Test"])

    Model_Dict[i]["Y Prediction"] = y_pred

    print("SVM Model Info for Ticker: " + i)
    print(
        "Accuracy:",
        metrics.accuracy_score(Model_Dict[i]["Y Test"], Model_Dict[i]["Y Prediction"]),
    )
    Model_Dict[i]["Accuracy"] = metrics.accuracy_score(
        Model_Dict[i]["Y Test"], Model_Dict[i]["Y Prediction"]
    )
    print(
        "Precision:",
        metrics.precision_score(
            Model_Dict[i]["Y Test"],
            Model_Dict[i]["Y Prediction"],
            pos_label=str(1),
            average="macro",
        ),
    )
    Model_Dict[i]["Precision"] = metrics.precision_score(
        Model_Dict[i]["Y Test"],
        Model_Dict[i]["Y Prediction"],
        pos_label=str(1),
        average="macro",
    )
    print(
        "Recall:",
        metrics.recall_score(
            Model_Dict[i]["Y Test"],
            Model_Dict[i]["Y Prediction"],
            pos_label=str(1),
            average="macro",
        ),
    )
    Model_Dict[i]["Recall"] = metrics.recall_score(
        Model_Dict[i]["Y Test"],
        Model_Dict[i]["Y Prediction"],
        pos_label=str(1),
        average="macro",
    )
    print("#################### \n")

"""## Evaluation"""

average_accuracy = np.mean([Model_Dict[ticker]["Accuracy"] for ticker in Ticker_List])
average_precision = np.mean([Model_Dict[ticker]["Precision"] for ticker in Ticker_List])
average_recall = np.mean([Model_Dict[ticker]["Recall"] for ticker in Ticker_List])

# Round the average metrics to two decimal places
average_accuracy_rounded = round(average_accuracy, 2)
average_precision_rounded = round(average_precision, 2)
average_recall_rounded = round(average_recall, 2)

average_accuracy_rounded, average_precision_rounded, average_recall_rounded

for i in Ticker_List:

    prediction_length = len(Model_Dict[i]["Y Prediction"])

    Stock_Data[i]["SVM Signal"] = 0
    Stock_Data[i]["SVM Returns"] = 0
    Stock_Data[i]["Total Strat Returns"] = 0
    Stock_Data[i]["Market Returns"] = 0

    Signal_Column = Stock_Data[i].columns.get_loc("SVM Signal")
    Strat_Column = Stock_Data[i].columns.get_loc("SVM Returns")
    Return_Column = Stock_Data[i].columns.get_loc("Total Strat Returns")
    Market_Column = Stock_Data[i].columns.get_loc("Market Returns")

    Stock_Data[i].iloc[-prediction_length:, Signal_Column] = list(
        map(int, Model_Dict[i]["Y Prediction"])
    )
    Stock_Data[i]["SVM Returns"] = Stock_Data[i]["SVM Signal"] * Stock_Data[i][
        "Returns"
    ].shift(-1)

    Stock_Data[i].iloc[-prediction_length:, Return_Column] = np.nancumsum(
        Stock_Data[i]["SVM Returns"][-prediction_length:]
    )
    Stock_Data[i].iloc[-prediction_length:, Market_Column] = np.nancumsum(
        Stock_Data[i]["Returns"][-prediction_length:]
    )

    Model_Dict[i]["Sharpe_Ratio"] = (
        Stock_Data[i]["Total Strat Returns"][-1] - Stock_Data[i]["Market Returns"][-1]
    ) / np.nanstd(Stock_Data[i]["Total Strat Returns"][-prediction_length:])

## 這裡的Sharpe Ratio 概念錯誤，參考就好

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

for i in Ticker_List:
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(
        Stock_Data[i][-prediction_length:].index.values,
        Stock_Data[i]["Total Strat Returns"][-prediction_length:].values,
        color="g",
        label="Strat Returns",
    )

    ax.plot(
        Stock_Data[i][-prediction_length:].index.values,
        Stock_Data[i]["Market Returns"][-prediction_length:].values,
        color="b",
        label="Market Returns",
    )

    ax.set(xlabel="Date", ylabel="Returns")
    plt.title(i, fontsize=15)
    ax.xaxis.set_major_locator(ticker.AutoLocator())

    plt.figtext(
        0.95, 0.78, s="Sharpe Ratio " + "{0:.5g}".format(Model_Dict[i]["Sharpe_Ratio"])
    )
    plt.figtext(
        0.95,
        0.75,
        s="Sum Total Strat Returns "
        + "{0:.5g}".format(Stock_Data[i]["Total Strat Returns"].sum()),
    )
    plt.figtext(
        0.95, 0.72, s="Model Accuracy " + "{0:.5g}".format(Model_Dict[i]["Accuracy"])
    )
    plt.figtext(
        0.95, 0.69, s="Model Precision " + "{0:.5g}".format(Model_Dict[i]["Precision"])
    )
    plt.figtext(
        0.95, 0.66, s="Model Recall " + "{0:.5g}".format(Model_Dict[i]["Recall"])
    )

    plt.legend(loc="best")
    plt.show()
## 這裡的Sharpe ratio, Sum total return並非我們印象中的cumulative return, 絕對值無法用來參考但相對值可以
