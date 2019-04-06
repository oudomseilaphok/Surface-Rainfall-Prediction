import csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import sklearn.metrics as mt
import scipy.stats as sp
import heapq
import numpy as np
import statsmodels.api as sm


def getExoArrayForTrain(length):



        train_r2 = R2[0:length]
        history_r2 = [x for x in train_r2]

        train_r3 = R3[0:length]
        history_r3 = [x for x in train_r3]

        train_r6 = R6[0:length]
        history_r6 = [x for x in train_r6]

        exo_array = list()

        for i in range(0, len(history_r2)):
            exo_array.append([history_r2[i], history_r3[i], history_r6[i]])
        return exo_array

def getExoArrayForTest(length):


        test_r2 = R2[length:length + 1]
        history_r2 = [x for x in test_r2]

        test_r3 = R3[length:length + 1]
        history_r3 = [x for x in test_r3]

        test_r6 = R6[length:length + 1]
        history_r6 = [x for x in test_r6]

        return [test_r2[0], test_r3[0], test_r6[0]]


X = [0.05,0.05,0.05,0.05,0.15,0.15,0.15,0.4,0.1,0.15,0.6,0.35,0.25,0.4,0.3,0.75,0.5]
R2 = [1, 0.1, 0.8, 0.6, 0.8, 5, 8, 7, 3, 3, 14, 3, 6, 4, 4, 9, 2]
R3 = [2, 0.2, 1, 2, 1, 6, 8, 5, 2, 4, 16, 5, 3, 6, 5, 10, 4]
R6 = [1, 0.6, 0.6, 4, 1.5, 5, 6, 5, 2, 6, 9, 6, 4, 2, 5, 10 , 6]

size = 16
#X = difference(X,2)

train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()



for t in range(len(test)):
    model = ARIMA(endog=history, order=(1, 0, 0), exog=getExoArrayForTrain(len(history)))
    print(getExoArrayForTrain(len(history)))
    model_fit = model.fit(disp=0, trend='nc', method='css')
    print(getExoArrayForTest(len(history)))
    output = model_fit.forecast(exog=getExoArrayForTest(len(history)))
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error_mse = mt.mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error_mse)

error_mae = mt.mean_absolute_error(test, predictions)
print('Test MAE: %.3f' % error_mae)

array1 = test
array2 = predictions
pyplot.plot(array1)
pyplot.axis([0, len(array1),0,1.5])
#pyplot.xticks(range(len(array1)), time[len(train):len(time)-1])
pyplot.plot(array2, color='red')
pyplot.show()