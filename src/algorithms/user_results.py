"""LSTM-based result reporting reconstructed from the project snippet."""

import math

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from django.conf import settings
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class UserFinaleports:
    def starProcess(self):
        filepath = settings.MEDIA_ROOT + "\\" + "covid_19_india.csv"
        df = pd.read_csv(filepath)
        dataframe = df[["Confirmed"]]
        dataset = dataframe.values.astype("float32")

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.67)
        train, test = dataset[0:train_size, :], dataset[train_size : len(dataset), :]

        def create_dataset(values, look_back=1):
            data_x, data_y = [], []
            for i in range(len(values) - look_back - 1):
                a = values[i : (i + look_back), 0]
                data_x.append(a)
                data_y.append(values[i + look_back, 0])
            return numpy.array(data_x), numpy.array(data_y)

        numpy.random.seed(7)
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

        trainPredictPlot = numpy.empty_like(dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[look_back : len(trainPredict) + look_back, :] = trainPredict

        testPredictPlot = numpy.empty_like(dataset)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) + 1 : len(dataset) - 1, :] = testPredict

        plt.plot(scaler.inverse_transform(dataset))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()

        return trainScore, testScore
