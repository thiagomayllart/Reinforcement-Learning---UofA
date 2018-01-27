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


alpha = 0.1 #alpha value
w = None #weight vector
feature_vectors = None #feature vectors array
last_state = None #last state
discount = None #discount value
v = None #estimates

def agent_init():
    global w, feature_vectors, discount, v
    v = np.zeros(1000)
    discount = 1 #set discount to 1
    w = np.zeros(10) #weight vector with size equal to quantity of groups (10). Groups: 0-99, 100-199, since index 0 corresponds to state 1 and 999 to 1000
    feature_vectors = np.zeros((1000, 10)) #1 feature vector for each one of 1000 states and 10 values on each feature vector

    i = 0
    j = 0

    #set 1 in the feature vector to the index equal to the correspondent group of the state
    while j < 1000:
        feature_vectors[j][i] = 1
        i = i + 1
        if i == 10:
            i = 0
        j = j + 1

# Function to take action
# It considers the possibility of going to the left and rigth (each with 0,5 chance)
# After deciding the direction, it selects a random state with equal probability
# on the next 100 states to that direction. If there are no 100 states to that direction, it selects
# one of the states between the actual state and the end with equal probability

def take_action(state):
    possibility = float(decimal.Decimal(random.randrange(1, 100)) / 100)  #chance to go to the right
    if possibility > 0.5:
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
        if state - 100 < 0: # 0,5 chance to go to the left
            quantity_of_states_left = state
            if int(quantity_of_states_left) == 1:
                return state - 1
            random_action = decimal.Decimal(random.randrange(1, int(quantity_of_states_left))) # selects one state between actual and the end
            return state - random_action

        else:
            possibility2 = decimal.Decimal(random.randrange(1, 100))  # selects one state between the last 100

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

    group1 = 0
    count_state = 100
    #find the correspondent group for the actual state
    while state > count_state:
        group1 = group1 + 1
        count_state = count_state + 100

    group2 = 0
    count_last_state = 100
    #find the correspondent group for the last state
    while last_state > count_last_state:
        group2 = group2 + 1
        count_last_state = count_last_state + 100

    v_last_state = w[group2] #the estimate for the last state equals to the weight value on the position of the index of its group
    v_state = w[group1] #the estimate for the state equals to the weight value on the position of the index of its group

    brackets = (reward + v_state - v_last_state)*alpha #calculation for the value between brackets in the formula in the book Reinforcement learning page 166

    gradient = np.zeros(10) #gradient has the size of quantity of groups

    gradient[group2] = 1 #position equal to the correspondent group of the last state equals to 1

    i = 0

    #perform update of the weight vector
    while i < 10:
        value = w[i]
        w[i] = value + brackets * gradient[i]
        i = i + 1

    action = take_action(state) #take action
    last_state = state #save last state
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """

    global alpha, w, feature_vectors, last_state, discount, v

    v_state = 0

    group2 = 0
    count_last_state = 100
    # find the correspondent group for the last state
    while last_state > count_last_state:
        group2 = group2 + 1
        count_last_state = count_last_state + 100

    v_last_state = w[group2] #the estimate for the last state equals to the weight value on the position of the index of its group

    brackets = (reward + v_state - v_last_state) * alpha #calculation for the value between brackets in the formula in the book Reinforcement learning page 166

    gradient = np.zeros(10) #gradient has the size of quantity of groups

    #if terminal state is on group 1, set position 0 to 1
    if reward == -1:
        gradient[0] = 1
    else:
        #if terminal state is on group 10 set position 9 to 1
        gradient[9] = 1
    i = 0

    # perform update of the weight vector
    while i < 10:
        value = w[i]
        w[i] = value + brackets * gradient[i]
        i = i + 1

    i = 0
    j = 0
    while j < 1000:
        group2 = 0
        count_last_state = 100
        while j > count_last_state:
            group2 = group2 + 1
            count_last_state = count_last_state + 100

        v_last_state = w[group2]
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

