import pandas as pd

data_raw = pd.read_csv('C:/Users/Admin/PycharmProjects/BTC-data/bitcoin-historical-data-kaggle/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv',usecols=['Open','High','Low','Close','Weighted_Price'])

data_raw.to_pickle('C:/Users/Admin/PycharmProjects/BTC-data/bitcoin-historical-data-kaggle/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.pkl')