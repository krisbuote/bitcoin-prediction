import urllib
import requests
import json
import numpy as np

url = 'https://graphs2.coinmarketcap.com/currencies/bitcoin/' #coinmarketcap data for bitcoin marketcap, BTc price, usd price, etc.

all_data = requests.get(url).json()



#Only care about the USD-price data within all_data
price_usd = all_data["price_usd"] #pull string from dict. [[time, price],...]

# price_usd = np.asarray(price_usd)
print(price_usd)
print(len(price_usd))
print(price_usd[0])

# current_state = np.zeros(4)
# print(current_state)
# current_state[0] = 0
# current_state[1] = 1000
# current_state[2] = price_usd[2][0]
# current_state[3] = price_usd[2][1]
#
# print(current_state)

price_usd_array = np.zeros([len(price_usd),2])
print(price_usd_array)

for i in range(len(price_usd)):
    price_usd_array[i,0] = price_usd[i][0]
    price_usd_array[i,1] = price_usd[i][1]

print(price_usd_array)

np.save("BitcoinDayPrice_April2013-June2018", price_usd_array)


