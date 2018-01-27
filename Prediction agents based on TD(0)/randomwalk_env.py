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
size = None
max_states = None

def env_init():
    global current_state, max_states
    current_state = np.zeros(1) #declare first state
    max_states = 1000 #number of states = 1000

def env_start():
    """ returns numpy array """
    global current_state
    current_state = None
    current_state = 0
    #first_state will be row 499 (starts in 0) = 500
    current_state = 499
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

    #reward always 0, unless it is the terminal state (999 = 1 or 0 = -1)
    reward = 0
    current_state = action #current state = action (horizontal block that agent moved)
    is_terminal = False

    #if terminal state was found return reward 1 if terminal = 1000 or return -1 if terminal = 0
    if current_state == 0:
        is_terminal = True
        current_state = None
        reward = -1
    else:
        if current_state == 999:
            is_terminal = True
            current_state = None
            reward = 1

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
