import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Load data. Drop timestamp
raw_data = pd.read_csv('./data/bitcoincharts_6hr_2017-08-28_to_2018-08-22.csv')

data = raw_data.drop(['Timestamp'], axis=1)

print(data.head())

''' Visualize Data'''
raw_data.hist()
plt.show()

raw_data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

raw_data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

scatter_matrix(raw_data)
plt.show()

correlations = raw_data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(list(raw_data))
ax.set_yticklabels(list(raw_data))
plt.show()


''' Transform data.'''
features = list(data)
for feature in features:
    # Standardize to zero mean, unit variance
    data[feature] = preprocessing.scale(data[feature])

# Review data after scaling
scatter_matrix(raw_data)
plt.show()

plt.figure(1)
# line plot
plt.subplot(211)
plt.plot(data['Close'])
# histogram
plt.subplot(212)
plt.hist(data['Close'])
plt.show()


def bollinger_bands(s, k, n):
    """get_bollinger_bands DataFrame
    s is series of values
    k is multiple of standard deviations
    n is rolling window (time unit)
    """

    b = pd.concat([s, s.rolling(n).agg([np.mean, np.std])], axis=1)
    b['BB' + str(n) + 'hr_upper'] = b['mean'] + b['std'] * k
    b['BB' + str(n) + 'hr_lower'] = b['mean'] - b['std'] * k
    b.rename(columns={'mean': 'BB' + str(n) + '_mean'}, inplace=True)

    return b.drop('std', axis=1)


# Include Bollinger Bands into dataframe
BB24 = bollinger_bands(data['Close'], k=2, n=24).drop(columns=['Close'])
BB120 = bollinger_bands(data['Close'], k=2, n=120).drop(columns=['Close'])
data = data.join(BB24)
data = data.join(BB120)
print(data.head())

# Drop any rows with NaN
data = data.dropna(axis=0, how='any')
print(data.head())

def check_null(data):
    print("Training Data:")
    print(pd.isnull(data).sum())


check_null(data)
data.to_csv('./data/preprocessed_6hr_data_lastYear_standardized_BB.csv', index=False)


'''
EXTRA INDICATORS:

def calculate_ichimoku(df):
    """get ichimoku cloud indicators
    input dataframe to calculate and include indicators in df
    """
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    high_prices = df['High']
    low_prices = df['Low']
    close_prices = df['Close']

    period9_high = pd.rolling_max(high_prices, window=9)
    period9_low = pd.rolling_min(low_prices, window=9)
    tenkan_sen = (period9_high + period9_low) / 2
    df['tenkan_sen'] = tenkan_sen

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = pd.rolling_max(high_prices, window=26)
    period26_low = pd.rolling_min(low_prices, window=26)
    kijun_sen = (period26_high + period26_low) / 2
    df['kijun_sen'] = kijun_sen

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    df['senkou_span_a'] = senkou_span_a

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = pd.rolling_max(high_prices, window=52)
    period52_low = pd.rolling_min(low_prices, window=52)
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    df['senkou_span_b'] = senkou_span_b

    return df


def calculate_rsi(df, window_length=21):
    # Get just the close
    close = data['Close']
    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = pd.stats.moments.ewma(up, window_length)
    roll_down1 = pd.stats.moments.ewma(down.abs(), window_length)

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    RSI1 /= 100 # Percentage -> [0,1]

    # Calculate the SMA
    roll_up2 = pd.rolling_mean(up, window_length)
    roll_down2 = pd.rolling_mean(down.abs(), window_length)

    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))
    RSI2 /= 100 # Percentage -> [0,1]

    # Add RS1, RS2, to dataframe
    df['RSI1'] = RSI1
    df['RSI2'] = RSI2

    return df
    
# Include ichimoku cloud into dataframe
# data = calculate_ichimoku(data)
print(data.head())

# Include RSI into dataframe
# data = calculate_rsi(data, window_length=21)
'''
