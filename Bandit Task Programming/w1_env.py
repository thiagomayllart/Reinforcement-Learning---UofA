#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian
  Purpose: for use of Reinforcement learning course University of Alberta Fall 2017
 
  env *ignores* actions: rewards are all random
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

#create all variables that will be used by the enviroment

this_reward_observation = (None, None, None) # this_reward_observation: (floating point, NumPy array, Boolean)
mean_q_actions_array = None #array that stores the mean value for each action
num_actions = None

def env_init():
    global this_reward_observation
    global mean_q_actions_array
    global num_actions

    #initialize the variables for the enviroment, dependending on the quantity of actions available

    num_actions = 10
    mean_q_actions_array = np.zeros(num_actions)

    #Initialize the mean values for each action utilizing a Gaussian distribution (mean = 0, variance = 1)
    for i in range(0, num_actions):
        mean_q_actions_array[i] = rand_norm(0.0, 1.0)

    local_observation = np.zeros(0) # An empty NumPy array

    this_reward_observation = (0.0, local_observation, False)


def env_start(): # returns NumPy array
    return this_reward_observation[1]


#returns the observation for the action taken (reward, state and if some terminal state was hit)
def env_step(this_action): # returns (floating point, NumPy array, Boolean), this_action: NumPy array
    global mean_q_actions_array
    global this_reward_observation
    action = int(this_action)

    #obtains a reward for the action taken
    #the value is based on a distribution of mean = mean value for that action and variance = 1
    the_reward = rand_norm(mean_q_actions_array[action], 1)
    this_reward_observation = (the_reward, this_reward_observation[1], False)

    #returns (reward, state and if some terminal state was hit)
    return this_reward_observation

#cleans the enviroment variables
def env_cleanup():
    global this_reward_observation
    global mean_q_actions_array
    global num_actions

    this_reward_observation = (None, None, None)  # this_reward_observation: (floating point, NumPy array, Boolean)
    mean_q_actions_array = None
    num_actions = None
    return


#function that returns the action with the best value (best mean q)
def optimal_action():
    global mean_q_actions_array
    i = 0
    best = 0
    while i < len(mean_q_actions_array):
        if mean_q_actions_array[best] < mean_q_actions_array[i]:
            best = i
        i = i + 1
    return best


def env_message(inMessage): # returns string, inMessage: string
    if inMessage == "what is your name?":
        return "my name is skeleton_environment!"

    if inMessage == "optimal_action?":
        return optimal_action()

    # else
    return "I don't know how to respond to your message"
