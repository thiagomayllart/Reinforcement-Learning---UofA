#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta 
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

current_state = None #variable current_state
wind_array = None #array of intensity of wind in each column
max_rows = None #maximum of rows in windy gridworld
max_columns = None #maximum of columns in windy gridworld
lst_state = None #last state
initialState = None # variable to remember the initial state
episode_count = None #variable to count number of episodes

def env_init():
    global current_state, wind_array, max_columns, max_rows, episode_count
    current_state = np.zeros(2) #create the first state (coordinates i,j)
    max_rows = 6 #maximum of rows = 7
    max_columns = 9 #maximum of columns = 10
    episode_count = -1 #initialize counting of episodes as -1

def env_start():
    """ returns numpy array """
    global current_state, lst_state, initialState, episode_count
    episode_count = episode_count + 1 #new episode, sum +1 on counting
    current_state = None
    lst_state = None
    current_state = np.zeros(2)
    #first_state will be row 3, column 0
    current_state[0] = 2
    current_state[1] = 0
    initialState = current_state
    lst_state = current_state
    #return first state
    return current_state

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state, lst_state, wind_array, max_rows, max_columns, initialState, episode_count

    #reward always 0, unless it is the goal state (1)
    reward = 0
    current_state = action #current state = action (coordinate planned to move by the action)
    is_terminal = False

    #if goal was found and is not the first episode, return reward 1 and terminate the episode
    if current_state[0] == 0 and current_state[1] == 8 and episode_count > 0:
        is_terminal = True
        current_state = None
        reward = 1
    else:
        # if goal was found but it is the first episode, don't initialize a new episode, continue in the same even goal being hit
        # we want to make the maximum steps per episode be reached in the first episode
        if current_state[0] == 0 and current_state[1] == 8:
            is_terminal = False
            reward = 1

    lst_state = action
    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}
    #returns the dictionary reward, state, isTerminal
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
