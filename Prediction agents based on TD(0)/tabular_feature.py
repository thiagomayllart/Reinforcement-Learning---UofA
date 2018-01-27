#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for On-Policy Sarsa Control Agent
           for use on A4 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle
import decimal
import random


alpha = 0.5 #alpha
w = None #weights vector
feature_vectors = None #feature vector
last_state = None #last state
discount = None #discount
v = None #estimates


def agent_init():
    global w, feature_vectors, discount, v

    v = np.zeros(1000)
    discount = 1 #set discount = 1
    w = np.zeros(1000) #weight vector of 1000 weights
    feature_vectors = np.zeros((1000, 1000)) #1 feature vector for each state with 1000 spaces each

    i = 0
    #the space corresponding to the number of the state will have value set to 1 and the rest will continue 0
    while i < 1000:
        feature_vectors[i][i] = 1
        i = i + 1

# Function to take action
# It considers the possibility of going to the left and rigth (each with 0,5 chance)
# After deciding the direction, it selects a random state with equal probability
# on the next 100 states to that direction. If there are no 100 states to that direction, it selects
# one of the states between the actual state and the end with equal probability


def take_action(state):
    possibility = float(decimal.Decimal(random.randrange(1, 100)) / 100)
    if possibility > 0.5: #chance to go to the right
        if state + 100 > 999:
            quantity_of_states_right = 999 - state
            if quantity_of_states_right == 1:
                return state + 1
            random_action = decimal.Decimal(random.randrange(1, int(quantity_of_states_right))) # selects one state between actual and the end

            return state + random_action

        else:
            possibility2 = decimal.Decimal(random.randrange(1, 100)) # selects one state between the next 100

            return state + possibility2
    else: # 0,5 chance to go to the left
        if state - 100 < 0:
            quantity_of_states_left = state
            if int(quantity_of_states_left) == 1:
                return state - 1
            random_action = decimal.Decimal(random.randrange(1, int(quantity_of_states_left))) # selects one state between actual and the end
            return state - random_action

        else:
            possibility2 = decimal.Decimal(random.randrange(1, 100)) # selects one state between the last 100

            return state - possibility2


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global last_state, v
    v = np.zeros(1000)
    last_state = state #save last state
    action = take_action(state) #take an action

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global alpha, w, feature_vectors, last_state, discount

    v_last_state = 0
    v_state = 0
    i = 0

    #get estimate value function for the last state
    while i < len(feature_vectors):
        v_last_state = v_last_state + feature_vectors[int(last_state)][i]*w[i]
        i = i + 1

    i = 0
    # get estimate value function for the state
    while i < len(feature_vectors):
        v_state = v_state + feature_vectors[int(state)][i]*w[i]
        i = i + 1

    brackets = (reward + v_state - v_last_state)*alpha #calculation for the value between brackets in the formula in the book Reinforcement learning page 166

    gradient = feature_vectors[int(last_state)] #gradient is equal to the feature vector of the last state

    i = 0

    #perform update to the weight vector
    while i < 1000:
        value = w[i]
        w[i] = value + brackets * gradient[i]
        i = i + 1

    action = take_action(state) # take action
    last_state = state # save last state
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """

    global alpha, w, feature_vectors, last_state, discount, v

    v_last_state = 0
    v_state = 0
    i = 0

    # get estimate value function for the last state
    # estimate for terminal state = 0
    while i < len(feature_vectors):
        v_last_state = v_last_state + feature_vectors[int(last_state)][i] * w[i]
        i = i + 1

    brackets = (reward + v_state - v_last_state) * alpha #calculation for the value between brackets in the formula in the book Reinforcement learning page 166

    gradient = feature_vectors[int(last_state)] #gradient is equal to the feature vector of the last state

    i = 0

    #perform update
    while i < 1000:
        value = w[i]
        w[i] = value + brackets * gradient[i]
        i = i + 1

    i = 0
    j = 0

    while j < 1000:
        v_last_state = 0
        i = 0
        while i < 1000:
            v_last_state = v_last_state + feature_vectors[j][i] * w[i]
            i = i + 1
        v[j] = v_last_state
        j = j + 1

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
    if (in_message == 'v_value'):
        return v
    else:
        return "I don't know what to return!!"

