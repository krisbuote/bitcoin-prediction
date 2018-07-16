'''
This code gets the bitcoin daily price over the last ~5 years
Author: Kristopher Buote
'''
import requests
import numpy as np

url = 'https://graphs2.coinmarketcap.com/currencies/bitcoin/' #coinmarketcap data for bitcoin marketcap, BTc price, usd price, etc.
all_data = requests.get(url).json()

#Only care about the USD-price data within all_data
price_usd = all_data["price_usd"] #pull string from dict. [[time, price],...]
time_price = np.zeros([len(price_usd),2])

for i in range(len(price_usd)):
    time_price[i,0] = price_usd[i][0]
    time_price[i,1] = price_usd[i][1]

np.save("BitcoinDayPrice", time_price)


