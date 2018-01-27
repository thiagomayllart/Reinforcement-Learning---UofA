#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue. 
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("windygridworld_env", "sarsa_agent")

import numpy as np
import pickle

if __name__ == "__main__":
    num_episodes = 170 #quantity of episodes to simulate figure in the book
    max_steps = 10000
    num_runs = 1
    steps_episodes = np.zeros(num_episodes) #array of sum of number of steps in each episode
    # dict to hold data for key episodes

    for run in range(num_runs):
      counter = 0
      print "run number: ", run
      RL_init()
      print "\n"
      for episode in range(num_episodes):
        RL_episode(max_steps)
        steps = RL_agent_message('Steps')
        steps_episodes[episode] = steps #save the sum of number of steps in the array of steps per episode
      RL_cleanup()

    #create a .txt with the information inside steps_episodes array
    filename = "EpisodeXSumSteps"
    with open(filename, "w") as data_file:
        j = 0
        while j < len(steps_episodes):
            data_file.write("{0}\n".format(steps_episodes[j]))
            j = j + 1



