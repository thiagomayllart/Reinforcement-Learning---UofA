#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent_part3")

import numpy as np

if __name__ == "__main__":
    num_episodes = 1000
    num_runs = 1
    steps_per_episode = np.zeros(num_episodes)
    steps = np.zeros([num_runs,num_episodes])
    dict = None
    for r in range(num_runs):
        print "run number : ", r
        RL_init()
        for e in range(num_episodes):
            print "episode: ", e
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            steps_per_episode[e] = steps_per_episode[e] + RL_num_steps()
        dict = RL_agent_message("calculateMaxQ") #call function calculateMaxQ
    np.save('steps',steps)
    position_array = dict["position"] #get array of positions
    velocity_array = dict["velocity"] #get array of velocities
    maxqvector = dict["maxqvector"] #get array of -MaxQ for each combination of position and velocity

    #save array of positions on .txt
    filename = "position_intervals"
    with open(filename, "w") as data_file:
        j = 0
        while j < 50:
            k = 0
            while k < 50:
                data_file.write("{0}\n".format(position_array[j]))
                k = k + 1
            j = j + 1

    # save array of velocities on .txt
    filename = "velocity_intervals"
    with open(filename, "w") as data_file:
        k = 0
        while k < 50:
            j = 0
            while j < 50:
                data_file.write("{0}\n".format(velocity_array[j]))
                j = j + 1
            k = k + 1

    #save array of -MaxQ on .txt
    filename = "maxqvalues"
    with open(filename, "w") as data_file:
        j = 0
        i = 0
        while i < 50:
            j = 0
            while j < 50:
                data_file.write("{0}\n".format(maxqvector[i][j]))
                j = j + 1
            i = i + 1