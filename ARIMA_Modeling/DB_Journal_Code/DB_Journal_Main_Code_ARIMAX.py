import csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import sklearn.metrics as mt
import scipy.stats as sp
import heapq
import numpy as np
import statsmodels.api as sm

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

def getExoArrayForTrain(length):

        mapCorrelation = list()
        for j in range(0,len(radarData)):
            mapCorrelation.append(radarData[j][0:length])
        corrValue = list()
        for i in range(0,len(mapCorrelation)):
            corrValue.append(sp.pearsonr(X[0:length], mapCorrelation[i])[0])

        corrValue = np.array(corrValue)
        top3 = heapq.nlargest(3, range(len(corrValue)), corrValue.take)

        train_r2 = mapCorrelation[top3[0]][0:length]
        history_r2 = [x for x in train_r2]

        train_r3 = mapCorrelation[top3[1]][0:length]
        history_r3 = [x for x in train_r3]

        train_r6 = mapCorrelation[top3[2]][0:length]
        history_r6 = [x for x in train_r6]

        exo_array = list()

        for i in range(0, len(history_r2)):
            exo_array.append([history_r2[i], history_r3[i], history_r6[i]])
        return exo_array

def getExoArrayForTest(length):

        mapCorrelation = list()
        for j in range(0, len(radarData)):
            mapCorrelation.append(radarData[j][0:length])

        corrValue = list()
        for i in range(0,len(mapCorrelation)):
            corrValue.append(sp.pearsonr(X[0:length], mapCorrelation[i])[0])
        corrValue = np.array(corrValue)
        top3 = heapq.nlargest(3, range(len(corrValue)), corrValue.take)

        spatialCorrelationValue_1.append(corrValue[top3[0]])
        spatialCorrelationValue_2.append(corrValue[top3[1]])
        spatialCorrelationValue_3.append(corrValue[top3[2]])
        print(top3)
        print(corrValue)

        test_r2 = radarData[top3[0]][length:len(radarData[top3[0]])]
        history_r2 = [x for x in test_r2]

        test_r3 = radarData[top3[1]][length:len(radarData[top3[1]])]
        history_r3 = [x for x in test_r3]

        test_r6 = radarData[top3[2]][length:len(radarData[top3[2]])]
        history_r6 = [x for x in test_r6]

        return [test_r2[0], test_r3[0], test_r6[0]]

path = "20170702_0130_2340_3pixels.csv"
X = list()
time = list()
radarData = list()

with open(path, 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        if(row[1] != "AWS"):
            time.append(row[0])
            X.append(float(row[1]))
            for i in range(2, len(row)):
                radarData[i-2].append(float(row[i]))
        else:
            for i in range(len(row)-2):
                radarData.append(list())
csvFile.close()

# size of train data. Exp 0.5 = 50% of the data is for train
size = int(len(X) * 0.5)

allTemporalCorrelationValue = list()
spatialCorrelationValue_1 = list()
spatialCorrelationValue_2 = list()
spatialCorrelationValue_3 = list()

#X = difference(X,2)

train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

predictions_arima = list()
predictions_arimax = list()
for t in range(len(test)):
    #c = constant include, #nc = no constant

    model_arima = ARIMA(endog=history, order=(1, 0, 0))
    model_fit_arima = model_arima.fit(disp=0, trend='nc', method='css')  # , Ŷt - ϕ1Yt-1 = μ - θ1et-1 Formula Main ARIMA
    output_arima = model_fit_arima.forecast()
    y_arima = output_arima[0]
    predictions_arima.append(y_arima)

    model_arimax = ARIMA(endog=history, order=(1, 0, 0), exog=getExoArrayForTrain(len(history)))
    model_fit_arimax = model_arimax.fit(disp=0, trend='nc', method='css') #, Ŷt - ϕ1Yt-1 = μ - θ1et-1 + β(Xt - ϕ1Xt-1) Formula Main ARIMAX
    output_arimax = model_fit_arimax.forecast(exog=getExoArrayForTest(len(history)))

    y_arimax = output_arimax[0]
    print(output_arimax)
    predictions_arimax.append(y_arimax)

    allTemporalCorrelationValue.append((sm.graphics.tsa.acf(history, nlags=1))[1])
    obs = test[t]
    history.append(obs)
    print('ARIMA predicted=%f, expected=%f' % (y_arima, obs))
    print('ARIMAX predicted=%f, expected=%f' % (y_arimax, obs))

error_mse = mt.mean_squared_error(test, predictions_arima)
print('ARIMA Test MSE: %.3f' % error_mse)
error_mae = mt.mean_absolute_error(test, predictions_arima)
print('ARIMA Test MAE: %.3f' % error_mae)

error_mse = mt.mean_squared_error(test, predictions_arimax)
print('ARIMAX Test MSE: %.3f' % error_mse)
error_mae = mt.mean_absolute_error(test, predictions_arimax)
print('ARIMAX Test MAE: %.3f' % error_mae)

print(sum(allTemporalCorrelationValue) / len(allTemporalCorrelationValue))
print(sum(spatialCorrelationValue_1) / len(spatialCorrelationValue_1))
print(sum(spatialCorrelationValue_2) / len(spatialCorrelationValue_2))
print(sum(spatialCorrelationValue_3) / len(spatialCorrelationValue_3))

# plot
# array1 = train + test
# array2 = train + predictions
array1 = test
array2 = predictions_arima
array3 = predictions_arimax
pyplot.plot(array1)
pyplot.plot(array2, color='orange')
# pyplot.axis([0, len(array1),0,1.5])
pyplot.axis([0, len(array1),0,1.5])
#pyplot.xticks(range(len(array1)), time[len(train):len(time)-1])
pyplot.plot(array3, color='red')
pyplot.show()