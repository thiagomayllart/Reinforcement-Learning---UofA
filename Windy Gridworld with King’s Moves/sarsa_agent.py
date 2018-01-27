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


#each position is identified by a number
up = 0
down = 1
left = 2
right = 3
up_left = 4
up_right = 5
down_left = 6
down_right = 7
epsilon = 0.1 #epsilon
alpha = 0.5 #alpha
Q = None
last_st = None #last state (one time step before)
last_move = None #last move (move of time step before
steps = 0
possible_moves = None
actions = 8 #quantity of actions, in case of wanting 9 actions, select 9


""" This function receives the actual state and the desired move and change the state
    according to the move (up, down, left, right)
"""
def get_position_action(move, state):
    global up, down, left, right, up_left, up_right, down_left, down_right, actions

    new_state = np.zeros(2)

    if move == up:
        new_state[0] = state[0] - 1
        new_state[1] = state[1]

    if move == down:
        new_state[0] = state[0] + 1
        new_state[1] = state[1]

    if move == left:
        new_state[1] = state[1] - 1
        new_state[0] = state[0]

    if move == right:
        new_state[1] = state[1] + 1
        new_state[0] = state[0]

    if move == up_left:
        new_state[0] = state[0] - 1
        new_state[1] = state[1] - 1

    if move == up_right:
        new_state[0] = state[0] - 1
        new_state[1] = state[1] + 1

    if move == down_left:
        new_state[0] = state[0] + 1
        new_state[1] = state[1] - 1

    if move == down_right:
        new_state[0] = state[0] + 1
        new_state[1] = state[1] + 1

    #uncomment in case of 9 actions
    #if move == 8:
        #return new_state

    return new_state

"""
    This function set the array of possible actions for all the possible states
    in the windy gridworld.
    
    States in the laterals will have some actions set to 0, so, the agent won't be able
    to take invalid actions
"""
def setImpossibleActions():
    global Q, possible_moves, actions

    i = 0
    while i<7:
        j = 0
        while j<10:
            k = 0
            state = [i, j]
            while k < actions:
                new_state = get_position_action(k, state)
                if new_state[0] < 0 or new_state[0] > 6 or new_state[1] < 0 or new_state[1] > 9:
                    possible_moves[i][j][k] = 0 #if some move leads to a state outside the gridworld, the move is invalidated, 0 == False
                else:
                    possible_moves[i][j][k] = 1 #if the move leads to a position inside the gridworld, the move is valid, 1 == True
                k = k + 1
            j = j + 1
        i = i + 1

def agent_init():

    global Q, last_st, last_move, steps, possible_moves, actions
    last_move = None
    last_st = None
    steps = 0
    Q = np.zeros((7, 10, actions))
    possible_moves = np.zeros((7, 10, actions))
    setImpossibleActions()


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global last_st, last_move, steps, epsilon, alpha, possible_moves
    possibility = float(decimal.Decimal(random.randrange(0, 100)) / 100) #get a random number from 0.00 until 1.00
    maxmove = [] #array of movements with best Q
    maxvalue = -100000 #initialize the best minimum Q with a large negative number
    if possibility <= 1 - epsilon:
        #if the random number is <= 1 - epsilon, select the move with the best Q
        i = 0
        while i < actions:
            #check if there are moves with best Q until the last possible move
            if possible_moves[int(state[0])][int(state[1])][i] == 1:
                if Q[int(state[0])][int(state[1])][i] > maxvalue:
                    maxmove[:] = []
                    maxvalue = Q[int(state[0])][int(state[1])][i]
                    maxmove.append(i)
                else:
                    #if there are more moves with the same highest Q, add them to the array of best moves
                    if Q[int(state[0])][int(state[1])][i] == maxvalue:
                        maxmove.append(i)
            i = i + 1
        if len(maxmove) == 1:
            #if there is only one best move, select it and get the next position after doing the move
            move = maxmove[0]
            action = get_position_action(move, state)
        else:
            #if there are more best moves, select one of them randomly
            move = maxmove[random.randint(0, len(maxmove)-1)]
            action = get_position_action(move, state)


    else:
        #if the random number is > 1- epsilon select one action randomly
        move = random.randint(0, actions - 1)
        action = get_position_action(move, state)
        #select another action while the one selected before is an invalid move
        while possible_moves[int(state[0])][int(state[1])][move] == 0:
            move = random.randint(0, actions - 1)
            action = get_position_action(move, state)

    last_st = state #save the last state
    last_move = move #save the last move
    steps = steps + 1

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global Q, epsilon, alpha,steps, last_st, last_move, steps, epsilon, alpha, possible_moves
    possibility = float(decimal.Decimal(random.randrange(0, 100)) / 100) #get a random number from 0.00 until 1.00
    maxvalue = -100000 #initialize the best minimum Q with a large negative number
    maxmove = [] #array of movements with best Q
    if possibility <= 1 - epsilon:
        # if the random number is <= 1 - epsilon, select the move with the best Q
        i = 0
        while i < actions:
            # check if there are moves with best Q until the last possible move
            if possible_moves[int(state[0])][int(state[1])][i] == 1:
                if Q[int(state[0])][int(state[1])][i] > maxvalue:
                    maxmove[:] = []
                    maxvalue = Q[int(state[0])][int(state[1])][i]
                    maxmove.append(i)
                else:
                    # if there are more moves with the same highest Q, add them to the array of best moves
                    if Q[int(state[0])][int(state[1])][i] == maxvalue:
                        maxmove.append(i)
            i = i + 1

        if len(maxmove) == 1:
            # if there is only one best move, select it and get the next position after doing the move
            move = maxmove[0]
            action = get_position_action(move, state)
        else:
            move = maxmove[random.randint(0, len(maxmove) - 1)]
            action = get_position_action(move, state)

    else:
        # if the random number is > 1- epsilon select one action randomly
        move = random.randint(0, actions - 1)
        action = get_position_action(move, state)
        # select another action while the one selected before is an invalid move
        while possible_moves[int(state[0])][int(state[1])][move] == 0:
            move = random.randint(0, actions - 1)
            action = get_position_action(move, state)

    #upgrade Q based on on-policy Sarsa for updating Q and the last action and the last state saved before
    Q[int(last_st[0])][int(last_st[1])][last_move] = Q[int(last_st[0])][int(last_st[1])][last_move] + alpha*(reward + Q[int(state[0])][int(state[1])][move] - Q[int(last_st[0])][int(last_st[1])][last_move])
    steps = steps + 1 #count the step
    last_st = state #save the last state
    last_move = move #save the last move
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global Q, last_move, last_st, alpha
    # upgrade Q based on on-policy Sarsa for updating Q and the last action and the last state saved before
    # reward for the terminal state is 0
    Q[int(last_st[0])][int(last_st[1])][last_move] = Q[int(last_st[0])][int(last_st[1])][last_move] + alpha * (
    reward + 0 - Q[int(last_st[0])][int(last_st[1])][last_move])

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
    if (in_message == 'Steps'):
        return steps
    else:
        return "I don't know what to return!!"

