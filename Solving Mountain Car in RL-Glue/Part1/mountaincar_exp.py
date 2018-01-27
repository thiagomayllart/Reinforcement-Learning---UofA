#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent")

import numpy as np

if __name__ == "__main__":
    num_episodes = 200
    num_runs = 50
    steps_per_episode = np.zeros(num_episodes) #array for the number of steps per episode
    steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            steps_per_episode[e] = steps_per_episode[e] + RL_num_steps() #sum the number of steps on the correspondent episode
    np.save('steps',steps)

    #save number of steps per episode averaged through 50 runs on a .txt
    filename = "Steps_per_Episode"
    with open(filename, "w") as data_file:
        j = 0
        while j < num_episodes:
            data_file.write("{0}\n".format(steps_per_episode[j] / num_runs)) #calculate the average and save on the .txt
            j = j + 1