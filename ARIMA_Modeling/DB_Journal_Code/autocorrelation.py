import pandas as pd
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

path = "560_20170702_0130_2340_3pixels.csv"
X = list()
with open(path, 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        if(row[1] != "AWS"):
            X.append(float(row[1]))
csvFile.close()

# autocorrelation
# print(sm.graphics.tsa.acf(X, nlags=10))
# sm.graphics.tsa.plot_acf(X, lags=10)

# partial autocorrelation
print(sm.graphics.tsa.pacf(X, nlags=10, method="ywm"))
sm.graphics.tsa.plot_pacf(X, lags=10, method="ywm")

plt.show()