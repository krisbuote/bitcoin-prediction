import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]: #Ignore Header
        vec.append(float(line.split(",")[1])) #Using the second column of sheet (index starts at 0)

    return vec

# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = np.array(data[d:t + 1])
    res = []
    for i in range(n - 1): # Divided by 100 to prevent overflow errors. Big changes in btc prices!
        res.append(sigmoid((block[i + 1] - block[i])/100)) # The state is sigmoid(difference)
    return np.array([res])





