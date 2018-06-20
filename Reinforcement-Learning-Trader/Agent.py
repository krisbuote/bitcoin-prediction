#!/usr/bin/env python

"""
Latest version of this code created by Kristopher Buote.
    Purpose: Program a bitcoin-USD trader agent.

    Original skeleton code from:
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

global usd, btc, price, timestamp

price_usd = np.load("./historical_data/BitcoinDayPrice_April2013-June2018.npy")
max_price = np.max(price_usd[:,1])
num_total_states = int(max_price) + 1
num_total_actions = 3 #Buy All, Sell All, Hold
epsilon = 0.1

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    global Q, pi, rewards, returns, episode_counter, actions

    Q = np.zeros([num_total_states,3]) #initialize action-value array. BUY, SELL, HOLD.
    #
    returns = np.zeros([num_total_states,3])
    #
    pi = np.zeros([num_total_states,3])
    actions = np.arange(3)
    #
    episode_counter = 1



def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global action_hist_old
    global action_hist_new
    global rewards

    action_hist_old = np.zeros([num_total_states,3]) #keep a tally of which actions are taken in this episode
    action_hist_new = np.zeros([num_total_states,3])

    price = int(state[3])
    action = 0 # BUY. $1000 -> BTC of equivalent value. EXPLORING START
    action_hist_new[price ,action] += 1  #add one to the action history array
    rewards = np.zeros([num_total_states, 3]) #set rewards for the episode back to zero

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global action_hist_old
    global action_hist_new, wallet

    btc = int(state[0])
    usd = int(state[1])
    price = int(state[3])

    last_action_array = action_hist_new[price,:] - action_hist_old[price,:] #this will leave a "1" where the last action was taken
    last_action = np.argmax(last_action_array) #the value of the last action taken (i,e money wagered)
    rewards[price, last_action] += reward #Keep adding the rewards
    action_hist_old = action_hist_new #set old history = new history of actions taken

    if rand_un()>epsilon: #act greedily with respect to pi
        num_same = np.where(pi[price,:] == np.max(pi[price,:]))  # choose equally between max actions
        action = int(np.random.choice(num_same[0]))

    else:  # exploratory action
        action = int(np.random.choice(actions))

    action_hist_new[price, action] +=1 #update action history array
    wallet = np.array([btc,usd])

    return action

def agent_end(reward, state):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi

    global state_action_hist
    global rewards
    global returns
    global episode_counter

    #UPDATE RETURNS#
    #the return for an episode, G, for a first-visit method, is the average reward experienced in a state-action pair
    for state in range(num_total_states):
        state_rewards = rewards[state,:]
        state_action_hist = action_hist_new[state,:]
        for action in range(num_total_actions):
            if state_action_hist[action] > 0:
                average_reward=state_rewards[action]/state_action_hist[action]
                returns[state,action] += average_reward  #the average reward of each state-action pair is the return for this episode
            else:
                average_reward = 0
                returns[state,action] += average_reward

    #UPDATE Q#
    for state in range(num_total_states):
        for action in range(num_total_actions):
            Q[state,action] = returns[state,action]/episode_counter #action-value equals averge returns

    episode_counter += 1

    #UPDATE PI#
    for state in range(num_total_states):
        num_same = np.where(Q[state, :] == np.max(Q[state, :]))  # choose equally between max actions
        if num_same == True:
            A_star = int(np.random.choice(num_same[0]))
        else:
            A_star = np.random.choice([1,2,3])

        for action in range(num_total_actions):
            if action == A_star:
                pi[state,action] = 1-epsilon+epsilon*abs(state)/max_price
            else:
                pi[state, action] = epsilon*abs(state)/max_price



    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    if (in_message == 'wallet'):
        return wallet
    else:
        return "I don't know what to return!!"

