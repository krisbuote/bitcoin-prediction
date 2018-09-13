from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pickle

''''''
def scaleData(csvFileName, useCols=['Timestamp', 'BTC-USD-Daily']):

    def Scaler(_dataTrain):  # Fit the scaler to Training Data only
        _scaler = MinMaxScaler(feature_range=(0, 1))
        _scaler.fit(_dataTrain)
        return _scaler


    # Read CSV
    path = './data/' + csvFileName + '.csv'
    df = pd.read_csv(path, usecols=useCols)
    data_split_ix = int(0.90*df.shape[0]) #Split percentage of training/evaluation

    #Get BTC Data only
    data = df['BTC-USD-Daily'].values
    dataTrain = data[:data_split_ix].reshape(-1,1)
    dataEvaluate = data[data_split_ix:].reshape(-1,1)

    #Create scaler
    scaler = Scaler(dataTrain)

    #scale all data according to training values
    dataTrainScaled = scaler.transform(dataTrain)
    dataEvaluateScaled = scaler.transform(dataEvaluate)

    #Save to df
    dfTrain = pd.DataFrame(data=dataTrainScaled,columns=['BTC-USD-Daily'])
    dfEvaluate = pd.DataFrame(data=dataEvaluateScaled,columns=['BTC-USD-Daily'])

    # #Save scaled df as .csv
    dataTrainFile = 'BitcoinDayPrice_trainScaled'
    dataEvaluateFile = 'BitcoinDayPrice_evaluateScaled'

    dfTrain.to_csv('./data/BitcoinDayPrice_trainScaled.csv')
    dfEvaluate.to_csv('./data/BitcoinDayPrice_evaluateScaled.csv')

    return dataTrainFile, dataEvaluateFile, scaler
