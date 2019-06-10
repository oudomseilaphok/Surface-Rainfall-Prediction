import csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
import sklearn.metrics as mt
import numpy
import matplotlib.pyplot as plt
from math import sqrt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from numpy import concatenate
import scipy.stats as sp
import math
import heapq
import numpy as np

import statsmodels.api as sm

# Seila : MAPE Calculation Function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Seila : Differencing Algorithm for testing
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# Seila : Get Spatial Variables and Wind Speed for Training Data
def getExoArrayForTrain(length, scanningRange, includeWindDirection, includeWindSpeed):

        mapCorrelation = list()
        for j in range(0,len(radarData)):
            mapCorrelation.append(radarData[j][0:length])
        corrValue = list()
        for i in range(0,len(mapCorrelation)):
            corrValue.append(sp.pearsonr(X[0:length], mapCorrelation[i])[0])
            # corrValue.append(sp.spearmanr(X[0:length], mapCorrelation[i])[0])

        #wind direction at the targeted time
        windDir = windDirection[length]
        # print(time[length])
        # print(corrValue)
        #extracted the targeted radar variables indexes by pruning and scanning logic
        topSection = scanningForMaxArea(scanningRange, corrValue, windDir, includeWindDirection)
        topSectionDataArray = list()
        topSectionCorrValues = list()

        #put all top variables data into array of arrays "topSectionDataArray"
        #put all top variables correlation value into array "topSectionCorrValues"
        for topIndex in range(len(topSection)):
            train = mapCorrelation[topSection[topIndex]][0:length]
            history = [x for x in train]
            topSectionDataArray.append(history)
            topSectionCorrValues.append(corrValue[topSection[topIndex]])

        #For getting average spatial value for showing in Figure
        spatialCorrelationValue.append(sum(topSectionCorrValues) / len(topSectionCorrValues))
        print(topSectionDataArray)
        #converting data into row by row arrays for training model
        exo_array = list()
        for i in range(0, len(topSectionDataArray[0])):
            dataRow = list()

            #include windspeed as variable
            if includeWindSpeed:
                dataRow.append(windSpeed[i])

            for k in range(0, len(topSectionDataArray)):
                dataRow.append(topSectionDataArray[k][i])
            exo_array.append(dataRow)

        return exo_array

# Seila : Get Spatial Variable for Predicting Data
def getExoArrayForTest(length, scanningRange, includeWindDirection, includeWindSpeed):

        mapCorrelation = list()
        for j in range(0, len(radarData)):
            mapCorrelation.append(radarData[j][0:length])

        corrValue = list()
        for i in range(0,len(mapCorrelation)):
            corrValue.append(sp.pearsonr(X[0:length], mapCorrelation[i])[0])

        #perform the same scan to extract the same value for the model predicting time
        windDir = windDirection[length]
        topSection = scanningForMaxArea(scanningRange, corrValue, windDir, includeWindDirection)
        topSectionDataArray = list()
        print(topSection)

        for topIndex in range(len(topSection)):
            test = radarData[topSection[topIndex]][length:length + 1]
            test_future = [x for x in test]
            topSectionDataArray.append(test_future)
        dataRow = list()
        # print(topSection)
        #include windspeed for prediction
        if includeWindSpeed:
            dataRow.append(windSpeed[length])

        for k in range(len(topSectionDataArray)):
            dataRow.append(topSectionDataArray[k][0])
        # print(time[length])
        # print(dataRow)
        return dataRow


#Seila: Pruning matrix based on wind direction
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


# Seila : SSM Scanning Algorithm, outputting index of top correlated pixels
def scanningForMaxArea(scanningWindow, corrValueDatas, windDir, includeWindDirection):

    #define exclusion
    allSectionCellIndex = list()
    allSectionCellCorrelation = list()

    # total cell data
    totalCells = len(corrValueDatas)

    # cell section that excluded from scanning
    exCells = list()

    # output the cells that will be pruned from scanning
    prunedCells = list()
    if includeWindDirection:
        prunedCells = matrixPruing(windDir)

    # scanning is based on top left cells of all sections, so excludes the cell at the right corners
    for ex in range(scanningWindow - 1):

        for rowEx in range(int(math.sqrt(totalCells))):
            exCells.append((int(math.sqrt(totalCells)) * (rowEx + 1) - ex) - 1)

        for colsEx in range(int(math.sqrt(totalCells))):
            exCells.append((totalCells - (radarRange * (ex + 1)) + (colsEx + 1)) - 1)

    # start scanning by considering excluded cells and pruned cells
    for i in range(len(corrValueDatas)):
        if i not in (exCells):
            totalCorr = 0
            sectionCellIndexs = list()
            for row in range(scanningWindow):

                
                for col in range(scanningWindow):
                    totalCorr = totalCorr + corrValueDatas[i + col  + (radarRange * row)]
                    sectionCellIndexs.append(i + col + (radarRange * row))

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

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# load dataset
dataset = read_csv('compiled_data/gangwon/4_20170710_0130_2340_7pixels.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[list(range(53, 104))], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 66
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# calculate RMSE
rmse = mt.mean_squared_error(inv_y, inv_yhat)
print('Test MSE: %.3f' % rmse)
mae = mt.mean_absolute_error(inv_y, inv_yhat)
print('Test MAE: %.3f' % mae)
pyplot.plot(inv_y, label='actual')
pyplot.plot(inv_yhat, label='predicted')
pyplot.legend()
pyplot.show()

# MAIN CODE START HERE:
# path = "compiled_data/gangwon/4_20170710_0130_2340_7pixels.csv"
path = "compiled_data/3_20170702_0130_2340_7pixels_short.csv"
main_includeWindDirection = True
main_includeWindSpeed = False
scanningRange = 3
radarRange = 7 # 7 = 7 * 7
nSelect = 3
#
X = list()
time = list()
radarData = list()
windSpeed = list()
windDirection = list()
allTemporalCorrelationValue = list()
spatialCorrelationValue = list()
actual_data_percentage = list()
predict_data_percentage = list()

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



