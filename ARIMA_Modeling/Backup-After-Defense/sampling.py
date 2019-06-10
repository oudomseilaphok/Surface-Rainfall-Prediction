import csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import sklearn.metrics as mt
import scipy.stats as sp
import math
import heapq
import numpy as np
import statsmodels.api as sm

# Seila : MAPE Calculation Function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Seila : Differencing Algorithm
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# Seila : Testing Accuracy of ARIMA p,d,q based on different configuration
def acurracyTestARIMA(data,p,d,q):
    predictions = list()
    data_size = int(len(data) * 0.5)
    train_arima, test_arima = data[0:data_size], data[data_size:len(data)]
    trainData = [x for x in train_arima]
    for t in range(len(test_arima)):
        model_arima = ARIMA(endog=trainData, order=(p, d, q))
        model_fit_arima = model_arima.fit(disp=0, trend='nc', method='css')
        output_arima = model_fit_arima.forecast()
        y_arima = output_arima[0]
        predictions.append(y_arima)
        actual = test_arima[t]
        trainData.append(actual)

    print('ARIMA Test MSE: %.3f' % mt.mean_squared_error(test_arima, predictions))
    print('ARIMA Test MAE: %.3f' % mt.mean_absolute_error(test_arima, predictions))


# Seila : Get Spatial Variable for Training Data
def getExoArrayForTrain(length, scanningRange):

        mapCorrelation = list()
        for j in range(0,len(radarData)):
            mapCorrelation.append(radarData[j][0:length])
        corrValue = list()
        for i in range(0,len(mapCorrelation)):
            corrValue.append(sp.pearsonr(X[0:length], mapCorrelation[i])[0])

        windDir = windDirection[length]

        print(time[length])
        print(windDir)
        topSection = scanningForMaxArea(scanningRange, corrValue, windDir)
        topSectionDataArray = list()
        topSectionCorrValues = list()

        for topIndex in range(len(topSection)):
            train = mapCorrelation[topSection[topIndex]][0:length]
            history = [x for x in train]
            topSectionDataArray.append(history)
            topSectionCorrValues.append(corrValue[topSection[topIndex]])
        print(topSectionCorrValues)
        spatialCorrelationValue.append(sum(topSectionCorrValues) / len(topSectionCorrValues))
        exo_array = list()
        for i in range(0, len(topSectionDataArray[0])):
            dataRow = list()
            for k in range(0, len(topSectionDataArray)):
                dataRow.append(topSectionDataArray[k][i])
            exo_array.append(dataRow)

        return exo_array

# Seila : Get Spatial Variable for Predicting Data
def getExoArrayForTest(length, scanningRange):

        mapCorrelation = list()
        for j in range(0, len(radarData)):
            mapCorrelation.append(radarData[j][0:length])

        corrValue = list()
        for i in range(0,len(mapCorrelation)):
            corrValue.append(sp.pearsonr(X[0:length], mapCorrelation[i])[0])

        windDir = windDirection[length]
        topSection = scanningForMaxArea(scanningRange, corrValue, windDir)
        topSectionDataArray = list()

        for topIndex in range(len(topSection)):
            test = radarData[topSection[topIndex]][length:length + 1]
            test_future = [x for x in test]
            topSectionDataArray.append(test_future)
        dataRow = list()
        for k in range(len(topSectionDataArray)):
            dataRow.append(topSectionDataArray[k][0])
        return dataRow

def matrixPruing(windDirection): #for 7 * 7 Matrix
    prunedCells = list()
    if (windDirection <= 22.5 or windDirection > 337.5): #N
        prunedCells.extend(range(0,7))
        prunedCells.extend(range(42,49))
        for i in range(5):
            prunedCells.extend([12 + (i*7), 13 + (i*7)])

    elif (windDirection > 22.5 and windDirection <=  67.5): #NE
        prunedCells.extend(range(0, 7))
        prunedCells.extend(range(7, 14))
        for i in range(5):
            prunedCells.extend([19 + (i * 7), 20 + (i * 7)])

    elif (windDirection > 67.5 and windDirection <=  112.5): #E
        prunedCells.extend(range(0, 7))
        prunedCells.extend(range(7, 14))
        for i in range(5):
            prunedCells.extend([14 + (i * 7), 20 + (i * 7)])

    elif (windDirection > 112.5 and windDirection <=  157.5): #SE
        prunedCells.extend(range(0, 7))
        prunedCells.extend(range(7, 14))
        for i in range(5):
            prunedCells.extend([14 + (i * 7), 15 + (i * 7)])

    elif (windDirection > 157.5 and windDirection <=  202.5): #S
        prunedCells.extend(range(0, 7))
        prunedCells.extend(range(42, 49))
        for i in range(5):
            prunedCells.extend([7 + (i * 7), 8 + (i * 7)])

    elif (windDirection > 202.5 and windDirection <=  247.5): #SW
        prunedCells.extend(range(35, 42))
        prunedCells.extend(range(42, 49))
        for i in range(5):
            prunedCells.extend([0 + (i * 7), 1 + (i * 7)])

    elif (windDirection > 247.5 and windDirection <=  292.5): #W
        prunedCells.extend(range(35, 42))
        prunedCells.extend(range(42, 49))
        for i in range(5):
            prunedCells.extend([0 + (i * 7), 6 + (i * 7)])

    elif (windDirection > 292.5 and windDirection <=  337.5): #NW
        prunedCells.extend(range(35, 42))
        prunedCells.extend(range(42, 49))
        for i in range(5):
            prunedCells.extend([5 + (i * 7), 6 + (i * 7)])

    return prunedCells


# Seila : SSM Scanning Algorithm, outputting index of max correlation value section
def scanningForMaxArea(scanningWindow, corrValueDatas, windDir):
    #define exclusion
    allSectionCellIndex = list()
    allSectionCellCorrelation = list()
    totalCells = len(corrValueDatas) #total cell data
    exCells = list() #cell section that excluded from scanning
    prunedCells = matrixPruing(windDir)
    for ex in range(scanningWindow - 1):

        for rowEx in range(int(math.sqrt(totalCells))):
            exCells.append((int(math.sqrt(totalCells)) * (rowEx + 1) - ex) - 1)

        for colsEx in range(int(math.sqrt(totalCells))):
            exCells.append((totalCells - (radarRange * (ex + 1)) + (colsEx + 1)) - 1)


    for i in range(len(corrValueDatas)):
        if i not in (exCells):
            totalCorr = 0
            sectionCellIndexs = list()
            for row in range(scanningWindow):
                #totalCorr = totalCorr + corrValueDatas[i + (5 * row)]
                for col in range(scanningWindow):
                    totalCorr = totalCorr + corrValueDatas[i + col  + (radarRange * row)]
                    sectionCellIndexs.append(i + col  + (radarRange * row))

            # pruning here in the code (If section has prunedCells)

            isPrunedSection = False
            for sectionCellIndex in sectionCellIndexs:
                if sectionCellIndex in prunedCells:
                    isPrunedSection = True

            if(isPrunedSection == False):
                allSectionCellIndex.append(sectionCellIndexs)
                allSectionCellCorrelation.append(totalCorr)

    allSectionCellCorrelation = np.array(allSectionCellCorrelation)
    topSection = heapq.nlargest(1, range(len(allSectionCellCorrelation)), allSectionCellCorrelation.take)[0]

    topSectionIndex =   allSectionCellIndex[topSection]
    topSectionCorrValues = list()
    # adding all correlationValue of top section into array
    for topIndex in range(len(topSectionIndex)):
        topSectionCorrValues.append(corrValueDatas[topSectionIndex[topIndex]])

    # select only n cells from that section
    topSectionCorrValues = np.array(topSectionCorrValues)
    topCellSelectionIndex = heapq.nlargest(nSelect, range(len(topSectionCorrValues)), topSectionCorrValues.take)

    resultTopIndex = list()
    for i in range(len(topCellSelectionIndex)):
        resultTopIndex.append(topSectionIndex[topCellSelectionIndex[i]])
    return  resultTopIndex

# MAIN CODE START HERE:
path = "compiled_data/3_20170702_0130_2340_7pixels_short.csv"
scanningRange = 3
radarRange = 7 # 7 = 7 * 7
nSelect = 3
X = list()
time = list()
radarData = list()
windSpeed = list()
windDirection = list()
allTemporalCorrelationValue = list()
spatialCorrelationValue = list()


with open(path, 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        if(row[1] != "AWS"):
            time.append(row[0])
            X.append(float(row[1]))
            windSpeed.append(float(row[2]))
            windDirection.append(float(row[3]))
            for i in range(4, len(row)):
                radarData[i-4].append(float(row[i]))
        else:
            for i in range(len(row)-4):
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
    predictions_arimax.append(y_arimax)

    obs = test[t]
    history.append(obs)
    print('ARIMA predicted=%f, expected=%f' % (y_arima, obs))
    print('ARIMAX predicted=%f, expected=%f' % (y_arimax, obs))

# error_mse = mt.mean_squared_error(test, predictions_arima)
# print('ARIMA Test MSE: %.3f' % error_mse)
error_mae = mt.mean_absolute_error(test, predictions_arima)
print('ARIMA Test MAE: %.3f' % error_mae)
#
# error_mse = mt.mean_squared_error(test, predictions_arimax)
# print('ARIMAX Test MSE: %.3f' % error_mse)
error_mae = mt.mean_absolute_error(test, predictions_arimax)
print('ARIMAX Test MAE: %.3f' % error_mae)

print('temporal corr:')
print(sum(allTemporalCorrelationValue) / len(allTemporalCorrelationValue))
print('spatial corr:')
print(sum(spatialCorrelationValue) / len(spatialCorrelationValue))

# error_mape = mean_absolute_percentage_error(test, predictions_arima)
# print('ARIMAX Test MSE: %.3f' % error_mape)
# error_mape = mt.mean_absolute_error(test, predictions_arimax)
# print('ARIMAX Test MAE: %.3f' % error_mape)

# plot
# array1 = train + test
# array2 = train + predictions
maxValue = list()
array1 = test
array2 = predictions_arima
array3 = predictions_arimax

maxValue.append(max(array1))
maxValue.append(max(array2))
maxValue.append(max(array3))

pyplot.plot(array1)
pyplot.plot(array2, color='orange')
# pyplot.axis([0, len(array1),0,1.5])
pyplot.axis([0, len(array1),0, (max(maxValue) + 0.1)])
#pyplot.xticks(range(len(array1)), time[len(train):len(time)-1])
pyplot.plot(array3, color='red')
pyplot.show()

