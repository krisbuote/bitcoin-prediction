import pandas as pd

# Read CSV
csvFileName = 'C:/Users/Admin/PycharmProjects/bitcoin-data/bitcoin-historical-data/coinbaseUSD_4hr_data_2014-12-01_to_2018-06-27.csv'
useCols=['Close']

path = csvFileName
df = pd.read_csv(path, usecols=useCols)
data_split_ix = int(0.925 * df.shape[0])  # Split percentage of training/evaluation

# Get BTC Data only
data = df['Close'].values
dataTrain = data[:data_split_ix].reshape(-1, 1)
dataEvaluate = data[data_split_ix:].reshape(-1, 1)

# Save to df
dfTrain = pd.DataFrame(data=dataTrain, columns=['Close'])
dfEvaluate = pd.DataFrame(data=dataEvaluate, columns=['Close'])

# #Save scaled df as .csv
# dataTrainFile = 'coinbaseUSD_4hr_data_2014-12-01_to_2018-06-27_trainScaled'
# dataEvaluateFile = 'coinbaseUSD_4hr_data_2014-12-01_to_2018-06-27_evaluateScaled'

dfTrain.to_csv('C:/Users/Admin/PycharmProjects/bitcoin-data/bitcoin-historical-data/coinbaseUSD_4hr_data_2014-12-01_to_2018-06-27_train.csv')
dfEvaluate.to_csv('C:/Users/Admin/PycharmProjects/bitcoin-data/bitcoin-historical-data/coinbaseUSD_4hr_data_2014-12-01_to_2018-06-27_evaluate.csv')