import urllib
import requests
import json
import numpy as np
import pandas as pd

'''
This Script scrapes data for different cryptocurrencies.

NOTE: COINS HAVE EXISTED FOR DIFFERENT PERIODS OF TIME !

TODO: Pad arrays with NaN to match length of bitcoin array.
'''
# coins = ['bitcoin', 'ethereum', 'bitcoin-cash', 'litecoin', 'ripple', 'eos', 'stellar', 'monero', 'dash', 'ethereum-classic']
coins = ['bitcoin']
d = {}


for i in range(len(coins)):

    url = 'https://graphs2.coinmarketcap.com/currencies/' + (coins[i]) + '/' #coinmarketcap data
    all_data = requests.get(url).json()

    #Only care about the USD-price data within all_data
    price_usd = all_data["price_usd"] #pull string from dict. [[time, price],...]
    price_usd_array = []

    for j in range(len(price_usd)):
        price_usd_array.append(price_usd[j][1])
    # Need to pad arrays here to be of equal length for each coin

    d[str(coins[i])] = price_usd_array


coins_df = pd.DataFrame(data=d)
coins_df.to_csv("Daily-Coin-Prices-2013-2018.csv")


# np.save("Daily-Coin-Prices-2013-2018", price_usd_array)


