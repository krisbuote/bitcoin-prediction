import urllib
import requests
import json
import numpy as np

url = 'https://graphs2.coinmarketcap.com/currencies/bitcoin/' #coinmarketcap data for bitcoin marketcap, BTc price, usd price, etc.
all_data = requests.get(url).json()

#Only care about the USD-price data within all_data
price_usd = all_data["price_usd"] #pull string from dict. [[time, price],...]
price_usd_array = np.zeros([len(price_usd),2])

for i in range(len(price_usd)):
    price_usd_array[i,0] = price_usd[i][0]
    price_usd_array[i,1] = price_usd[i][1]

np.save("BitcoinDayPrice_April2013-June2018", price_usd_array)


