import csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import sklearn.metrics as mt
import scipy.stats as sp
import math
import heapq
import numpy as np
import statsmodels.api as sm


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

def getExoArrayForTrain(length, scanningRange):

        mapCorrelation = list()
        for j in range(0,len(radarData)):
            mapCorrelation.append(radarData[j][0:length])
        corrValue = list()
        for i in range(0,len(mapCorrelation)):
            corrValue.append(sp.pearsonr(X[0:length], mapCorrelation[i])[0])

        topSection = scanningForMaxArea(scanningRange, corrValue)
        topSectionDataArray = list()
        topSectionCorrValues = list()

        for topIndex in range(len(topSection)):
            train = mapCorrelation[topSection[topIndex]][0:length]
            history = [x for x in train]
            topSectionDataArray.append(history)
            topSectionCorrValues.append(corrValue[topSection[topIndex]])
        # print(topSectionCorrValues)
        spatialCorrelationValue.append(sum(topSectionCorrValues) / len(topSectionCorrValues))
        exo_array = list()
        for i in range(0, len(topSectionDataArray[0])):
            dataRow = list()
            for k in range(0, len(topSectionDataArray)):
                dataRow.append(topSectionDataArray[k][i])
            exo_array.append(dataRow)

        return exo_array

def getExoArrayForTest(length, scanningRange):

        mapCorrelation = list()
        for j in range(0, len(radarData)):
            mapCorrelation.append(radarData[j][0:length])

        corrValue = list()
        for i in range(0,len(mapCorrelation)):
            corrValue.append(sp.pearsonr(X[0:length], mapCorrelation[i])[0])

        topSection = scanningForMaxArea(scanningRange, corrValue)
        topSectionDataArray = list()

        for topIndex in range(len(topSection)):
            test = radarData[topSection[topIndex]][length:length + 1]
            test_future = [x for x in test]
            topSectionDataArray.append(test_future)
        dataRow = list()
        for k in range(len(topSectionDataArray)):
            dataRow.append(topSectionDataArray[k][0])
        return dataRow

def scanningForMaxArea(scanningWindow, corrValueDatas):
    #define exclusion
    allSectionCellIndex = list()
    allSectionCellCorrelation = list()
    totalCells = len(corrValueDatas) #total cell data
    exCells = list() #cell section that excluded from scanning

    for ex in range(scanningWindow - 1):

        for rowEx in range(int(math.sqrt(totalCells))):
            exCells.append((int(math.sqrt(totalCells)) * (rowEx + 1) - ex) - 1)
            #print((int(math.sqrt(totalCells)) * (rowEx + 1) - ex))

        for colsEx in range(int(math.sqrt(totalCells))):
            exCells.append((totalCells - (5 * (ex + 1)) + (colsEx + 1)) - 1)
            #print((totalCells - (5 * (ex + 1)) + (colsEx + 1)))

    for i in range(len(corrValueDatas)):
        if i not in (exCells):
            #print(i)
            totalCorr = 0
            sectionCellIndexs = list()
            for row in range(scanningWindow):
                #totalCorr = totalCorr + corrValueDatas[i + (5 * row)]
                for col in range(scanningWindow):
                    totalCorr = totalCorr + corrValueDatas[i + col  + (5 * row)]
                    sectionCellIndexs.append(i + col  + (5 * row))

            allSectionCellIndex.append(sectionCellIndexs)
            #print(sectionCellIndexs)

            allSectionCellCorrelation.append(totalCorr)
            #print(totalCorr)

    allSectionCellCorrelation = np.array(allSectionCellCorrelation)
    topSection = heapq.nlargest(1, range(len(allSectionCellCorrelation)), allSectionCellCorrelation.take)[0]

    #print(allSectionCellIndex[topSection])
    return  allSectionCellIndex[topSection]


path = "compiled_data/seoul/4_20170702_0130_2340_5pixels.csv"
scanningRange = 4
X = list()
time = list()
radarData = list()
allTemporalCorrelationValue = list()
spatialCorrelationValue = list()


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


    model_arimax = ARIMA(endog=history, order=(1, 0, 0), exog=getExoArrayForTrain(len(history), scanningRange))
    model_fit_arimax = model_arimax.fit(disp=0, trend='nc', method='css') #, Ŷt - ϕ1Yt-1 = μ - θ1et-1 + β(Xt - ϕ1Xt-1) Formula Main ARIMAX
    output_arimax = model_fit_arimax.forecast(exog=getExoArrayForTest(len(history), scanningRange))

    allTemporalCorrelationValue.append((sm.graphics.tsa.acf(history, nlags=1))[1])
    y_arimax = output_arimax[0]
    #print(output_arimax)
    predictions_arimax.append(y_arimax)

    obs = test[t]
    history.append(obs)
    # print('ARIMA predicted=%f, expected=%f' % (y_arima, obs))
    # print('ARIMAX predicted=%f, expected=%f' % (y_arimax, obs))

# error_mse = mt.mean_squared_error(test, predictions_arima)
# print('ARIMA Test MSE: %.3f' % error_mse)
error_mae = mt.mean_absolute_error(test, predictions_arima)
print('ARIMA Test MAE: %.3f' % error_mae)
#
# error_mse = mt.mean_squared_error(test, predictions_arimax)
# print('ARIMAX Test MSE: %.3f' % error_mse)
error_mae = mt.mean_absolute_error(test, predictions_arimax)
print('ARIMAX Test MAE: %.3f' % error_mae)

# print(sum(allTemporalCorrelationValue) / len(allTemporalCorrelationValue))
# print(sum(spatialCorrelationValue) / len(spatialCorrelationValue))

# error_mape = mean_absolute_percentage_error(test, predictions_arima)
# print('ARIMAX Test MSE: %.3f' % error_mape)
# error_mape = mt.mean_absolute_error(test, predictions_arimax)
# print('ARIMAX Test MAE: %.3f' % error_mape)

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