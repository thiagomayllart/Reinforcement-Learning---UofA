#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
 
  agent does *no* learning, selects actions randomly from the set of legal actions
 
"""

from utils import rand_in_range
import numpy as np

#creating all variables that will be used by the agent

last_action = None # last_action: NumPy array
actions_estimative = None
alpha = None
epsilon = None
q_one = None
num_actions = 10


def agent_init():

    #initializing all the values for the agent.

    global last_action
    global alpha
    global epsilon
    global q_one
    global actions_estimative

    alpha = 0.1
    epsilon = 0.1
    q_one = 0

    actions_estimative = np.full(num_actions, q_one) #array with the first estimatives of rewards per action (first estimatives = q_one)
    last_action = np.zeros(1) # generates a NumPy array with size 1 equal to zero

#function to initialize the agent by taking the first action randomly (makes sense, since every action will have the same first estimatives)

def agent_start(this_observation): # returns NumPy array, this_observation: NumPy array
    global last_action

    last_action[0] = rand_in_range(num_actions)

    local_action = np.zeros(1)
    local_action[0] = last_action[0]

    return local_action[0]


#function that returns the entire array with the best estimatives (if more than one with the same value)
def is_best_estimative(estimatives):
    best = 0
    i = 1
    n = 0
    best_array = []
    best_array.append(estimatives[0])
    while i < num_actions:
        if estimatives[best] < estimatives[i]:
            best_array[:] = []
            best_array.append(best)
            best = i
        if estimatives[best] == estimatives[i]:
            best_array.append(i)
        i = i + 1
    return best_array

#function to obtain the best estimative
#if there are more than one best estimative (same values)
#these actions are store in a array until another best action be found
#if there is an action better than the old ones, the array is reinitialized and this best value is stored

def best_estimative(estimatives):
    best = 0
    i = 0
    n = 0
    best_array = []
    while i < num_actions:
        if estimatives[best] < estimatives[i]:
            best_array[:] = []
            best_array.append(i)
            best = i
        if estimatives[best] == estimatives[i]:
            best_array.append(i)
        i = i + 1
    best_random = rand_in_range(len(best_array))
    return best_array[best_random]

#function that makes the agent choose an action based on learning

def agent_step(reward, this_observation): # returns NumPy array, reward: floating point, this_observation: NumPy array
    global last_action
    global actions_estimative
    global epsilon
    global alpha

    #learning function (2.5 on book)
    actions_estimative[int(last_action[0])] = actions_estimative[int(last_action[0])] + alpha*(reward - actions_estimative[int(last_action[0])])
    local_action = np.zeros(1)
    #flip a coin = obtains a random value and check if this value is smaller than 1-epsilon(choose the action with the best estimative)
    #or bigger than 1-epsilon(choose a random action)
    flip_coin = np.random.random_sample()
    if flip_coin < (1-epsilon):
        local_action[0] = best_estimative(actions_estimative)
    else:
        local_action[0] = rand_in_range(num_actions)

    last_action = local_action

    return last_action


def agent_end(reward): # reward: floating point
    # final learning update at end of episode
    return

#cleans the agent variables

def agent_cleanup():
    global last_action
    global actions_estimative

    last_action = None
    actions_estimative = None
    # clean up
    return

def agent_message(inMessage): # returns string, inMessage: string
    # might be useful to get information from the agent
    global actions_estimative
    if inMessage == "what is your name?":
        return "my name is skeleton_agent!"

    #if someone sends a message to the agent (best_action?), returns the best action in that step
    if inMessage == "best action?":
        return is_best_estimative(actions_estimative)
    # else
    return "I don't know how to respond to your message"
