#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Andrew Jacobsen, Victor Silva, Mohammad M. Ajallooeian
  Last Modified on: 16/9/2017

  Experiment runs 2000 runs, each 1000 steps, of an n-armed bandit problem
"""

from rl_glue import *  # Required for RL-Glue
RLGlue("w1_env", "w1_agent") #setting the file names of .py of the environment and the agent

import numpy as np
import sys


def save_results(data, data_size, filename): # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))

if __name__ == "__main__":
    num_runs = 2000 #number of runs
    max_steps = 1000 #number of steps

    # array to store the results of each step
    optimal_action = np.zeros(max_steps) #array to store the sum of the boolean value (1 or 0) if the optimal action was chosen and later obtain the mean of each step
    cumulative_actions_step = np.zeros(max_steps)#array to store the sum of the rewards obtained in each step and each run and later obtain the mean of these rewards per step

    print "\nPrinting one dot for every run: {0} total Runs to complete".format(num_runs)
    for k in range(num_runs):
        RL_init() #initializing the RL-glue algorithm
        RL_start() #starting RL-glue algorithm
        for i in range(max_steps):

            # RL_step returns (reward, state, action, is_terminal); we need only the
            # action in this problem
            #best_action = (RL_agent_message("best action?")) #obtain the best actions array(k=1,2,3,...,10) in that step
            step = RL_step() #take an action and obtain (reward, state, action, is_terminal)
            cumulative_actions_step[i] = cumulative_actions_step[i] + step[0] #sum the reward value obtained in this step to the reward-actions array
            best_action = RL_env_message("optimal_action?")

            # check if the best action was chosen by verifying if the best action is one of the best actions available
            if best_action == step[2]:
                optimal_action[i] += 1  # true


            '''
            check if action taken was optimal

            you need to get the optimal action; see the news/notices
            announcement on eClass for how to implement this
            '''
            # update your optimal action statistic here

        RL_cleanup() #clean the algorithm to start a new run
        print ".",
        sys.stdout.flush()

    #obtain the mean value in each position of the array that store the sum of all rewards in each step
    for n in range(max_steps):
        cumulative_actions_step[n] = cumulative_actions_step[n]/num_runs

    #obtain the mean value in each position of the array that store boolean value for the action chosen (1 = optimal action chosen, 0 = optimal action not chosen)
    for n in range(max_steps):
        optimal_action[n] = optimal_action[n]/num_runs


    save_results(cumulative_actions_step, max_steps, "RL_EXP_OUT.dat") #save mean rewards results into RL_EXP_OUT.dat
    save_results(optimal_action, max_steps, "RL_EXP_BOOLEAN_OPTIMAL_OUT.dat") #save percentage of optimal actions obtained per step into RL_EXP_BOOLEAN_OPTIMAL_OUT.dat
    print "\nDone"
