# Using LSTM to predict tomorrow's price of Bitcoin

Stock-market traders supposedly see patterns within charts and make successful trades accordingly.
LSTMs seem to offer the best hope for predicting time series while maintaining a memory of the past trend.

This is a simple LSTM implemented with Keras. The general goal is to get the Net to somewhat successfully predict if the price will simply go above/below what it is today. If this trend direction was reliably predicted, profit could be turned, regardless of price change magnitude. 

The data used to train the model can be found in this repo's main folder.

It's the 'shampoo' bitcoin lstm because it's adapted from a shampoo time series forecasting tutorial found [here.](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
