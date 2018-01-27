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

def env_init():
    global current_state, wind_array, max_columns, max_rows
    current_state = np.zeros(2) #create the first state (coordinates i,j)
    wind_array = [0,0,0,1,1,1,2,2,1,0] #array of intesity of the wind (fixed intensity)
    max_rows = 7 #maximum of rows = 7
    max_columns = 10 #maximum of columns = 10

def env_start():
    """ returns numpy array """
    global current_state, lst_state
    current_state = None
    lst_state = None
    current_state = np.zeros(2)
    #first_state will be row 3, column 0
    current_state[0] = 3
    current_state[1] = 0
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
    global current_state, lst_state, wind_array, max_rows, max_columns

    #reduce the desired position of the agent by the wind intensity
    action[0] = action[0] - wind_array[int(current_state[1])]

    #if the wind would take the agent off the grid, bring him back
    while action[0] < 0:
        action[0] = action[0] + 1

    #reward always -1, unless it is the goal state (0)
    reward = -1
    #the action (desired state), now modified by the wind intensity becomes the current_state
    current_state = action
    is_terminal = False
    if current_state[0] == 3 and current_state[1] == 7:
        is_terminal = True
        current_state = None
        reward = 0

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
