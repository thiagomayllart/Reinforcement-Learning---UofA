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
discount = 0.95
epsilon = 0.1 #epsilon
alpha = 1 #alpha
Q = None
last_st = None #last state (one time step before)
last_move = None #last move (move of time step before
steps = 0
possible_moves = None
actions = 4 #quantity of actions, in case of wanting 9 actions, select 9
blocked_positions = None
model = None
n_times = None
observed_st_act = None
list_observed = []


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

    return new_state

def isblocked(new_state):
    i = 0
    blocked = 0
    while i < len(blocked_positions):
        if new_state[0] == blocked_positions[i][0] and new_state[1] == blocked_positions[i][1]:
            blocked = 1
            break
        i = i + 1
    return blocked


"""
    This function set the array of possible actions for all the possible states
    in the gridworld.

    States in the laterals or states next to obstacles will have some actions set to 0, so, the agent won't be able
    to take invalid actions
"""

def setImpossibleActions():
    global Q, possible_moves, actions, blocked_positions

    i = 0
    while i < 6:
        j = 0
        while j < 9:
            k = 0
            state = [i, j]
            while k < actions:
                new_state = get_position_action(k, state)
                if isblocked(new_state) == 1: #if the move leads to a wall, mark as invalid
                    possible_moves[i][j][k] = 0
                else:
                    if new_state[0] < 0 or new_state[0] > 5 or new_state[1] < 0 or new_state[1] > 8:
                        possible_moves[i][j][k] = 0 #if some move leads to a state outside the gridworld, the move is invalidated, 0 == False
                    else:
                        possible_moves[i][j][k] = 1 #if the move leads to a position inside the gridworld, the move is valid, 1 == True
                k = k + 1
            j = j + 1
        i = i + 1


def agent_init():

    global Q, last_st, last_move, steps, possible_moves, actions, blocked_positions, model, observed_st_act, list_observed, n_times
    last_move = None
    last_st = None
    steps = 0

    #set all the walls on the array of blocked_positions
    blocked_positions = np.zeros((7, 2))
    blocked_positions[0][0] = 1
    blocked_positions[0][1] = 2
    blocked_positions[1][0] = 2
    blocked_positions[1][1] = 2
    blocked_positions[2][0] = 3
    blocked_positions[2][1] = 2
    blocked_positions[3][0] = 4
    blocked_positions[3][1] = 5
    blocked_positions[4][0] = 0
    blocked_positions[4][1] = 7
    blocked_positions[5][0] = 1
    blocked_positions[5][1] = 7
    blocked_positions[6][0] = 2
    blocked_positions[6][1] = 7

    #initialize Q
    Q = np.zeros((6, 9, actions))

    #initialize array of observed state, actions
    observed_st_act = np.zeros((6, 9, actions))

    #initialize Model
    model = np.zeros((6, 9, actions, 3))

    #initialize array of possible moves (0 or 1 for each state,action)
    possible_moves = np.zeros((6, 9, actions))

    #set invalid and valid actions for each state, action
    setImpossibleActions()
    list_observed[:] = [] #each run, erase the observed state, actions
    n_times = 5 #variable to set how much planning

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global last_st, last_move, steps, epsilon, alpha, possible_moves, list_observed, observed_st_act
    steps = 0
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

    #if the state was never observed before, mark as observed and add to observed list
    if observed_st_act[int(state[0])][int(state[1])][move] == 0:
        observed_st_act[int(state[0])][int(state[1])][move] = 1
        observed = [state, move]
        list_observed.append(observed)

    return action


# Function to get the action the grants the maximum Q for a state

def maxAction(state):
    global Q, actions

    action = [] #array of maximum actions
    maxvalue = -1 #maximum value observed
    i = 0

    #iterate through all the actions
    while i < actions:
        if Q[int(state[0])][int(state[1])][i] > maxvalue and possible_moves[int(state[0])][int(state[1])][i] == 1:
            #if there is an action with higher value than the one observed and is a possible move
            #create a new list of maximum actions, add that one and save its value
            maxvalue = Q[int(state[0])][int(state[1])][i]
            action[:] = []
            action.append(i)
        else:
            #if another action is observed with the same highest value, append that action to the list of best actions
            if Q[int(state[0])][int(state[1])][i] == maxvalue and possible_moves[int(state[0])][int(state[1])][i] == 1:
                action.append(i)
        i = i + 1

    #if there is only one best action, return that one
    if len(action) == 1:
        return action[0]
    else:
        #if there are more than 1 action, select one randomly
        act = action[random.randint(0, len(action) - 1)]
        return act



def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    global Q, epsilon, alpha,steps, last_st, last_move, steps, epsilon, alpha, possible_moves, discount, model, n_times

    #if it is the initial state (initial state restarted without terminating the episode)
    if state[0] == 2 and state[1] == 0:
        #save number of steps and obtain an action using the same function to start the agent(agent_start)
        save_steps = steps
        init_act = agent_start(state)
        #increment steps
        steps = save_steps + 1
        return init_act
    else:
        #else, if it is the goal state (episode not terminated on goal) call
        #same function to end the agent (agent_end) and make next state be the first state
        if state[0] == 0 and state[1] == 8:
            agent_end(reward)
            initialState = [2, 0]
            return initialState
        else:
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

            # if the state was never observed before, mark as observed and add to observed list
            if observed_st_act[int(state[0])][int(state[1])][move] == 0:
                observed_st_act[int(state[0])][int(state[1])][move] = 1
                observed = [state, move]
                list_observed.append(observed)

            maxact = maxAction(state) #obtain best action for the specific state, highest Q
            #upgrade Q based on Q-learning
            Q[int(last_st[0])][int(last_st[1])][last_move] = Q[int(last_st[0])][int(last_st[1])][last_move] + alpha*(reward + discount*Q[int(state[0])][int(state[1])][maxact] - Q[int(last_st[0])][int(last_st[1])][last_move])

            #update model, Model(S,A) = reward, S'
            model[int(last_st[0])][int(last_st[1])][last_move][0] = reward
            model[int(last_st[0])][int(last_st[1])][last_move][1] = state[0]
            model[int(last_st[0])][int(last_st[1])][last_move][2] = state[1]

            steps = steps + 1 #count the step
            last_st = state #save the last state
            last_move = move #save the last move

            n_i = 0
            #do planning based on variable n
            #n defines how much planning

            #iterate n times
            while n_i < n_times:
                #select a random already observed state, action
                num = random.randint(0, len(list_observed)-1)
                obs_lst_state = list_observed[num][0]
                obs_lst_act = list_observed[num][1]

                #R, S' <- Model(S,A)
                reward = model[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act][0]
                obs_state = [model[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act][1], model[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act][2]]
                maxact = maxAction(obs_state) #get the best action for S'

                #update Q(S,A)
                Q[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act] = Q[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act] + alpha * (
                reward + discount * Q[int(obs_state[0])][int(obs_state[1])][maxact] - Q[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act])
                n_i = n_i + 1 #iterate n

            return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global Q, epsilon, alpha,steps, last_st, last_move, steps, epsilon, alpha, possible_moves, discount, model, n_times

    # upgrade Q based on Q-learning
    Q[int(last_st[0])][int(last_st[1])][last_move] = Q[int(last_st[0])][int(last_st[1])][last_move] + alpha * (
    reward + discount * 0 - Q[int(last_st[0])][int(last_st[1])][last_move])

    # update model, Model(S,A) = reward, S'
    model[int(last_st[0])][int(last_st[1])][last_move][0] = reward
    model[int(last_st[0])][int(last_st[1])][last_move][1] = 0
    model[int(last_st[0])][int(last_st[1])][last_move][2] = 8
    steps = steps + 1  # count the step

    n_i = 0
    # do planning based on variable n
    # n defines how much planning

    # iterate n times
    while n_i < n_times:
        # select a random already observed state, action
        num = random.randint(0, len(list_observed) - 1)
        obs_lst_state = list_observed[num][0]
        obs_lst_act = list_observed[num][1]

        # R, S' <- Model(S,A)
        reward = model[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act][0]
        obs_state = [model[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act][1], model[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act][2]]
        maxact = maxAction(obs_state) #get the best action for S'

        # update Q(S,A)
        Q[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act] = Q[int(obs_lst_state[0])][int(obs_lst_state[1])][
                                                                           obs_lst_act] + alpha * (
            reward + discount * Q[int(obs_state[0])][int(obs_state[1])][maxact] -
            Q[int(obs_lst_state[0])][int(obs_lst_state[1])][obs_lst_act])
        n_i = n_i + 1 #iterate n

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

