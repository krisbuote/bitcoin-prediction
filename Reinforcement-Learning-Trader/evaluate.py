import keras
from keras.models import load_model

from agent import Agent
from functions import *
from scaleData import scaleData
import matplotlib.pyplot as plt
import sys
''' Evaluate the trained agent on new data.'''

# Data files
coinbase_eval = 'coinbaseUSD_4hr_data_2014-12-01_to_2018-06-27_evaluate'
bit_stamp_eval = 'bitstampUSD_4hr_data_2017-07-04_to_2018-06-27'

stock_name, model_name = bit_stamp_eval,'./dense/model_ep200'
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, is_eval=True, model_name=model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, window_size, window_size + 1)
usd_start = 1000.0
usd_wallet = usd_start
btc_wallet = 0.0
action_history = []

for t in range(window_size, l):
    action = agent.act(state)

    current_price = data[t]

    # Action 0 is the prediction that trend will go down
    if action == 0 and btc_wallet != 0.0:
        usd_wallet = current_price * btc_wallet  # Trade btc for usd
        btc_wallet = 0.0

    # Action 1 is prediction that trend will go up
    if action == 1 and usd_wallet != 0.0:
        # Buy Bitcoin with entire USD fund
        btc_wallet = usd_wallet / current_price  # Trade usd for btc
        usd_wallet = 0.0

    next_state = getState(data, t + 1, window_size + 1)
    next_trend = next_state[0][-1]  # Will be > 0 if trend increases, will be < 0 if trend decreases

    done = True if t == l - 1 else False
    state = next_state
    action_history.append(action)

    if done:
        print("--------------------------------")
        net_worth = np.sum([usd_wallet, btc_wallet * current_price])
        agent.net_worth.append(net_worth)
        print("Total USD: " + formatPrice(usd_wallet))
        print("Total BTC: " + str(btc_wallet))
        print("Net worth in USD: " + formatPrice(net_worth))
        print("HODL net worth: " + formatPrice(usd_start * data[-1] / data[0]))
        print("--------------------------------")


        ## PLOT RESULTS ##
        index = np.arange(len(data))
        index_buys = []
        index_sells = []
        buys = []
        sells = []
        # for index, value in zip(index,data):
            # buys =
        for i in range(len(action_history)):
            if action_history[i] == 0:
                index_sells.append(i)
                sells.append(data[i])
            else:
                index_buys.append(i)
                buys.append(data[i])

        print(action_history)
        plt.plot(index, data, label='Data')
        plt.plot(index_buys, buys, 'go', label='Buys')
        plt.plot(index_sells, sells, 'ro', label='Sells')
        plt.legend()
        plt.show()


