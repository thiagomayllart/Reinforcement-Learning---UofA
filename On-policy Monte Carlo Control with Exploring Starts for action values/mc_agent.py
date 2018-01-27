#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle
from random import randint

last_state = None  # global variable to remember last state before stake
last_action = None  # global variable to remember last action before stake
qsa = None  # Q array of the action values for each pair state, action
returnsa = None  # array of the sum of the returns for each pair state, action in each episode, and quantities of episodes the pair appeared
pis = None  # array of the final policies for each state
appearedpairslistboolean = None  # array to check if the pair state, action appeared in the episode
appearedpairslist = None
states_available = 99  # number of states available
episode_number = 0


def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    global qsa, pis, last_action, last_state, returnsa, appearedpairslist, states_available, appearedpairslistboolean, states_available, episode_number
    episode_number = 0
    # initialize variables already described
    # they will be reset before each run
    last_state = None
    last_action = None
    qsa = None
    returnsa = None
    pis = None
    appearedpairslist = None
    appearedpairslistboolean = None

    pis = []
    i = 0
    while i < states_available + 1:
        pis.append([])
        i = i + 1
    i = 1
    j = 1
    returnsa = np.zeros(shape=(100, 100, 2))
    qsa = np.zeros(shape=(100, 100))


    # initialize the policy array in a smart way


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # initialize variables described before
    # they will be reset before each episode
    global returnsa, last_state, appearedpairslist, appearedpairslistboolean, pis, last_action, episode_number
    episode_number = episode_number + 1
    # initialize array of final policies
    # make it a list of size states_available of lists(undefined size)

    i = 1
    j = 1
    # initialize array of pairs states, action to verify if the pair already appeared in the episode
    appearedpairslistboolean = np.zeros(shape=(states_available + 1, states_available + 1))

    # make all the pairs false (they haven't appeared in the episode)
    while i <= states_available:
        while j <= i:
            appearedpairslistboolean[i][j] = False
            j = j + 1
        i = i + 1

    appearedpairslist = None
    appearedpairslist = []

    last_state = state[0]
    # choose a random action for the state, going from 1 until the maximum that can grant the sum to 100
    action = randint(1, np.minimum(state[0], states_available + 1 - state[0]))
    last_action = action
    # pick the first action, don't forget about exploring starts
    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global returnsa, appearedpairslist, appearedpairslistboolean, qsa, last_state, last_action
    pair = [last_state, last_action]

    # check if the pair already appeared in the episode
    # if haven't appeared, mark as appeared, and increase the sum of episodes in which it appeared
    if appearedpairslistboolean[pair[0]][pair[1]] == False:
        appearedpairslist.append(pair)
        appearedpairslistboolean[last_state][last_action] = True
        returnsa[pair[0]][pair[1]][1] = returnsa[pair[0]][pair[1]][1] + 1

    i = 0
    # go through all the pairs that appeared in the episode and increase their returns with the new reward
    # update the Q value for each pair based on the returns for each pair divided by the quantity of episodes in which they appeared]
    while i < len(appearedpairslist):
        returnsa[appearedpairslist[i][0]][appearedpairslist[i][1]][0] = \
        returnsa[appearedpairslist[i][0]][appearedpairslist[i][1]][0] + reward
        qsa[appearedpairslist[i][0]][appearedpairslist[i][1]] = (returnsa[appearedpairslist[i][0]][
                                                                     appearedpairslist[i][1]][0]) / (
                                                                returnsa[appearedpairslist[i][0]][
                                                                    appearedpairslist[i][1]][1])
        i = i + 1

    action = 0
    # if we are not in the exploring starts anymore
    # we can select actions based on the policy already stablished
    if episode_number > 1:

        # check all the actions available for the actual state from 1 until the minimum that may grant 100
        # if the Q value for the pair state, action is bigger than the previous pair, save the action
        # if the value is the same than the previous, append the action on a list of actions with the same value
        if len(pis[state[0]]) > 2:
            action_state = randint(1, len(pis[state[0]]))
            action = pis[state[0]][action_state]
        else:
            action = pis[state[0]][0]

    # if we are in the first episode, the episode to explore,
    # we select actions randomly
    else:
        action = randint(1, np.minimum(state[0], states_available + 1 - state[0]))

    last_state = state[0]
    last_action = action
    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    # same procedure in the agent_step
    global returnsa, appearedpairslist, appearedpairslistboolean, qsa, last_state, last_action, pis, qsa
    pair = [last_state, last_action]
    # check if the pair already appeared in the episode
    # if haven't appeared, mark as appeared, and increase the sum of episodes in which it appeared
    if appearedpairslistboolean[pair[0]][pair[1]] == False:
        appearedpairslist.append(pair)
        appearedpairslistboolean[last_state][last_action] = True
        returnsa[pair[0]][pair[1]][1] = returnsa[pair[0]][pair[1]][1] + 1
    i = 0
    # check all the actions available for the actual state from 1 until the minimum that may grant 100
    # if the Q value for the pair state, action is bigger than the previous pair, save the action
    # if the value is the same than the previous, append the action on a list of actions with the same value
    while i < len(appearedpairslist):
        returnsa[appearedpairslist[i][0]][appearedpairslist[i][1]][0] = \
        returnsa[appearedpairslist[i][0]][appearedpairslist[i][1]][0] + reward
        qsa[appearedpairslist[i][0]][appearedpairslist[i][1]] = (returnsa[appearedpairslist[i][0]][
                                                                     appearedpairslist[i][1]][0]) / (
                                                                    returnsa[appearedpairslist[i][0]][
                                                                        appearedpairslist[i][1]][1])
        i = i + 1

    i = 1
    maxaction = []
    maxq = 0
    j = 1
    # procedure to updtate the policies
    # check which action for all the states available, grant the state the highest action value
    # the action that grant that state the highest action value is now the new policy
    # this is done on every episode
    while i < len(pis):
        j = 1
        maxq = -1
        maxaction[:] = []
        while j <= np.minimum(i, states_available + 1 - i):
            if qsa[i][j] > maxq:
                maxq = qsa[i][j]
                maxaction[:] = []
                maxaction.append(j)
            else:
                if qsa[i][j] == maxq:
                    maxaction.append(j)
            j = j + 1
        pis[i] = maxaction
        i = i + 1

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return


def agent_message(in_message):  # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(qsa, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"

