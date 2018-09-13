from agent import Agent
from functions import *
from scaleData import scaleData

import sys
import pickle

''' For Command Line Control
# if len(sys.argv) != 4:
#       print("Usage: python train.py [stock] [window] [episodes]")
#       exit()
stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

'''

stock_name, window_size, episode_count = 'coinbaseUSD_4hr_data_2014-12-01_to_2018-06-27_train', 10, 500

agent = Agent(window_size)
data = getStockDataVec(stock_name)

l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, window_size, window_size + 1) # First sample is once window size can be used

    usd_start = 1000.0 # Start with $1000
    btc_wallet = 0.0
    usd_wallet = usd_start

    for t in range(window_size, l):
        action = agent.act(state)
        current_price = data[t]

        # Action 0 is the prediction that trend will go up
        if action == 0 and usd_wallet != 0.0:
            # Buy Bitcoin with entire USD fund
            btc_wallet = usd_wallet/current_price #Trade usd for btc
            usd_wallet = 0.0

        # Action 1 is the prediction that trend will go up
        if action == 1 and btc_wallet != 0.0:
            usd_wallet = current_price*btc_wallet # Trade btc for usd
            btc_wallet = 0.0

        next_state = getState(data, t + 1, window_size + 1)
        next_trend = next_state[0][-1] # Will be > 0 if trend increases, will be < 0 if trend decreases

        # Reward for trend prediction correctness
        if action == 0 and next_trend > 0.5: #correct up-trend prediction
            reward = 1.0
        elif action == 1 and next_trend < 0.5:  # correct down-trend prediction
            reward = 1.0
        else: # Incorrect predictions
            reward = -1.0

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            net_worth = np.sum([usd_wallet, btc_wallet * current_price])
            agent.net_worth.append(net_worth)
            print("Total USD: " + formatPrice(usd_wallet))
            print("Total BTC: " + str(btc_wallet))
            print("Net worth in USD: " + formatPrice(net_worth))
            print("HODL net worth: " + formatPrice(usd_start * data[-1] / data[0]))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 5 == 0:
        agent.model.save("models/dense-linear/model_ep" + str(e))
