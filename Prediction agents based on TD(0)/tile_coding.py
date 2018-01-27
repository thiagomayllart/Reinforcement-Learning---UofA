#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for On-Policy Sarsa Control Agent
           for use on A4 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from tiles3 import tiles, IHT
from utils import rand_in_range, rand_un
import numpy as np
import pickle
import decimal
import random

maxSize = 1000*50 #big size for the hash vector (states*tilings)
iht = IHT(maxSize)
weights = None
numTilings = 50 #number of tilings
stepSize = 0.01/numTilings #alpha
feature_vectors = None
last_state = None #last state
tile_width = 0.2 #tile width
v = None #estimates

# function to return the tilings corresponding to an state (x)
def mytiles(x):
    global numTilings, tile_width
    return tiles(iht, numTilings, [x*tile_width])

# function to perform update on the weight vector
def learn(x, y, z):
    global weights, stepSize
    tiles1 = mytiles(x) #obtain tiles for the last state
    estimate = 0
    #get estimate for state x (last state)
    for tile in tiles1:
        estimate += weights[tile]                  #form estimate
    tiles2 = mytiles(y) #obtain tiles for the actual state
    estimate2 = 0
    #get estimate for state y (actual state)
    for tile in tiles2:
        estimate2 += weights[tile]
    if y == 0 or y == 999: #if is terminal, estimate of actual state = 0
        estimate2 = 0
    error = z + estimate2 - estimate #calculate error (reward + estimate of actual state - estimate last state)
    #perform update on weigths vector
    for tile in tiles1:
        weights[tile] += stepSize * error          #learn weights

#function to get estimates of a state (x)
def test(x):
    global weights
    tiles = mytiles(x)
    estimate = 0
    for tile in tiles:
        estimate += weights[tile]
    return estimate


def agent_init():
    global maxSize, weights, v
    v = np.zeros(1000)
    weights = [0] * maxSize #weight vector size = hash vector size

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
    else:
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
    action = take_action(state) #take action

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global weights, last_state

    learn(int(last_state), int(state), reward) #perform learning

    action = take_action(state) #take action
    last_state = state #save last state
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """

    global weights, iht, numTilings, tile_width, last_state, v

    i = 0
    if reward == 1: #if reward equals 1, state is 999
        state = 999
    else:
        #else, state is 0
        state = 0

    learn(int(last_state), int(state), reward) #perform update

    while i < 1000:
        tiles1 = mytiles(i)
        estimate = 0
        for tile in tiles1:
            estimate += weights[tile]
        v[i] = estimate
        i = i + 1

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

