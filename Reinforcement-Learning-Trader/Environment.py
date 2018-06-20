#!/usr/bin/env python

"""
Latest version of this code created by Kristopher Buote.
    Purpose: Program the bitcoin-USD price environment.

    Original skeleton code from:
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np
import requests

price_usd = np.load("./historical_data/BitcoinDayPrice_April2013-June2018.npy")
price_usd[:,0]=price_usd[:,0]/(10**12) #scale down timestamps

max_price = np.max(price_usd[:, 1])
num_total_states = int(max_price) + 1

def env_init():
    global current_state, max_timestep, btc_start, usd_start
    #Starting currencies
    btc_start = 0
    usd_start = 1000
    current_state = np.zeros(4)  # [amount BTC, amount USD, price_usd[time], [price_usd[price]]
    max_timestep = len(price_usd) - 1

def env_start():
    """ returns numpy array """
    global timestep, last_state, btc

    btc = btc_start
    timestep = 0
    current_state[0] = btc_start
    current_state[1] = usd_start
    current_state[2] = price_usd[timestep][0]
    current_state[3] = price_usd[timestep][1]
    last_state = current_state
    return current_state

def env_step(action):

    global btc, usd, price, timestep

    last_btc = int(last_state[0])
    last_usd = int(last_state[1])
    last_price = int(last_state[3])

    if action == 0: #BUY
        btc += last_usd/last_price
        usd = 0

    if action == 1: #SELL
        usd += last_btc*last_price
        btc = 0

    if action == 2: #DO NOTHING
        btc = last_btc
        usd = last_usd

    timestep += 1
    current_state[0] = btc
    current_state[1] = usd
    new_price = price_usd[timestep][1]
    current_state[2] = price_usd[timestep][0]
    current_state[3] = new_price

    if timestep == max_timestep:
        is_terminal = True
    else:
        is_terminal = False

    reward = (usd-last_usd) + ((new_price*btc)-(new_price*last_btc))

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
