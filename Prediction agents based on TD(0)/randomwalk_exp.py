#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("randomwalk_env", "tabular_feature")
import math
import numpy as np
import pickle

if __name__ == "__main__":
    num_episodes = 5000 #quantity of episodes to simulate figure in the book
    max_steps = 2000
    num_runs = 10 #averaged over 10 runs
    v_values = np.zeros(1000)
    true_values = np.zeros(1000)
    rmse = np.zeros(5000)
    i = 0
    count = 0.002
    true_values[499] = 0
    true_values[0] = -1
    true_values[0] = 1
    while i < 500:
        true_values[i] = -1 + count
        count = count + 0.002
        i = i + 1

    i = 500
    count = 0.002
    while i < 1000:
        true_values[i] = 0 + count
        count = count + 0.002
        i = i + 1



    for run in range(num_runs):
      counter = 0
      print "run number: ", run
      RL_init()
      print "\n"
      for episode in range(num_episodes):
        RL_episode(max_steps)
        v_values = RL_agent_message("v_value")
        a = 0
        sum = 0
        while a < 1000:
            sum = sum + (math.pow(true_values[a] - v_values[a], 2))
            a = a + 1
        new_value = math.sqrt(1/1000*sum)
        rmse[episode] = rmse[episode] + new_value
      RL_cleanup()

      # create a .txt with the RMSE
      filename = "Estimates"
      with open(filename, "w") as data_file:
          j = 0
          while j < len(rmse):
              data_file.write("{0}\n".format(rmse[j])/num_runs)
              j = j + 1