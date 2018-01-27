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

maxSize = 4096 #big size for the hash vector (states*tilings)
iht = IHT(maxSize)
weights = None #weights vector
numTilings = 8 #number of tilings
stepSize = 0.1/numTilings #alpha
feature_vectors = None
last_state = None #last state
tile_width = 8 #tile width
lamb = 0.9 #lambda value
epsilon = 0 #epsilon value
gamma = 1 #gamma value
q = None #estimates
z = None #traces vector
last_action = None #variable to save last action
position_values_50_interval = None #vector to save all the positions are going to be plot on the graph
velocity_values_50_interval = None #vector to save all the velocities are going to be plot on the graph
maxqvector = None

# function to return the tilings corresponding to a state (x(position, y(velocity)
def mytiles(x, y, action):
    global numTilings, tile_width, iht
    scaleFactor1 = tile_width/(0.5 - (-1.2)) #scale factor for the position
    scaleFactor2 = tile_width/(0.07 - (-0.07)) #scale factor for the velocity
    return tiles(iht, numTilings, [x*scaleFactor1, y*scaleFactor2], [action])


#function to get q-value for the state(position(x), velocity(y)) and the action
def test(x, y, action):
    global weights
    tiles = mytiles(x, y, action)
    estimate = 0
    for tile in tiles:
        estimate += weights[tile]
    return estimate


def agent_init():
    global maxSize, weights, q, position_values_50_interval, velocity_values_50_interval, maxqvector
    position_values_50_interval = np.zeros(50) #init the array of the positions are going to be plot on the graph
    velocity_values_50_interval = np.zeros(50) #init the array of the velocities are goning to be plot on the graph
    i = 1
    position_values_50_interval[0] = -1.2 #first value of the positions array
    velocity_values_50_interval[0] = -0.07 #first value of the velocities array
    intervals_position = (0.5 - (-1.2))/50 #interval between each position on the graph
    intervals_velocity = (0.07 - (-0.07))/50 #interval between each velocity on the graph

    #iterate through each array and set the positions and velocities are going to be plot
    while i < 50:
        position_values_50_interval[i] = position_values_50_interval[i-1] + intervals_position
        velocity_values_50_interval[i] = velocity_values_50_interval[i-1] + intervals_velocity
        i = i + 1
    maxqvector = np.zeros((50, 50)) #init array of the -max Q vector
    weights = [0] * maxSize #weight vector size = hash vector size
    i = 0

    #set the weight vector values between 0 and -0.001
    while i < maxSize:
        weights[i] = -random.uniform(0, 0.001)
        i = i + 1



# Function to take action
# it always takes the best action since the epsilon is 0
# return the best action

def takeAction(state):
    i = 0
    maxValue = -10000
    action = -1
    while i < 3:
        estimate = test(state[0], state[1], i)
        if maxValue < estimate:
            maxValue = estimate
            action = i
        i = i + 1

    return action



def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global last_state, z, last_action, steps
    z = np.zeros(maxSize) #init array of traces
    last_state = state #save last state
    action = takeAction(state)#take action
    last_action = action
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global weights, last_state, last_action, z, gamma, lamb
    indexes = mytiles(last_state[0], last_state[1], last_action) #get tile indexes for the state and action
    delta = reward #set delta as the reward
    i = 0

    #calculate delta
    #and replace traces
    while i < len(indexes):
        delta = delta - float(weights[indexes[i]])
        z[indexes[i]] = 1 #replacing traces
        i = i + 1

    action = takeAction(state)#take action
    indexes = mytiles(state[0], state[1], action) #get tiles index for the actual state (S')
    i = 0

    #calculate delta for (S')
    while i < len(indexes):
        delta = delta + gamma*float(weights[indexes[i]])
        i = i + 1

    i = 0
    #update values of the weight vector
    while i < len(weights):
        weights[i] = float(weights[i]) + stepSize*delta*z[i]
        i = i + 1

    i = 0

    #update traces values
    while i < len(z):
        z[i] = gamma*lamb*z[i]
        i = i + 1

    last_state = state #save last state
    last_action = action
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """

    global weights, last_state, last_action, z, gamma, lamb

    indexes = mytiles(last_state[0], last_state[1], last_action) #get tile indexes for the last_state
    delta = reward #set delta as the reward

    i = 0
    #iterate and update delta
    # and replace traces
    while i < len(indexes):
        delta = delta - float(weights[indexes[i]])
        z[indexes[i]] = 1
        i = i + 1

    i = 0
    #update weight vector
    while i < len(weights):
        weights[i] = float(weights[i]) + stepSize*delta*z[i]
        i = i + 1

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


# Function to calculate -MaxQ for each combination of position and velocity (50 values of each with the same interval)
def calculateMaxQ():
    global position_values_50_interval, velocity_values_50_interval, maxqvector
    i = 0
    while i < 50:
        j = 0
        while j < 50:
            state = np.zeros(2)
            state[0] = position_values_50_interval[i]
            state[1] = velocity_values_50_interval[j]
            action = takeAction(state)
            maxqvector[i][j] = -(test(state[0], state[1], action))
            j = j + 1
        i = i + 1
    return {'position': position_values_50_interval, "velocity": velocity_values_50_interval, 'maxqvector': maxqvector}

def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'calculateMaxQ'):
        return calculateMaxQ()
    else:
        return "I don't know what to return!!"

