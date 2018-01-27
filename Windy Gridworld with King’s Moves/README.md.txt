# Description of the Problem and Plotting:

Windy Gridworld with King’s Moves. Re-solve the windy gridworld
task assuming eight possible actions, including the diagonal moves, rather than the usual four. (1)
How much better can you do with the extra actions? (2) Can you do even better by including a ninth
action that causes no movement at all other than that caused by the wind? Be sure to answer the two
questions posed above and submit evidence for your answers in the form of additional plots. In
addition describe the parameter settings used in your experiment (alpha & epsilon).
You are required to use RL-Glue for this exercise. Use the rl_glue.py and utils.py code from
assignment #3. You can either take gambler_exp.py, mc_agent.py, and gambler_env.py and modify
them to implement the windy gridworld, the Sarsa agent, and the appropriate experiment
respectively. Or you can write your own environment, agent and experiment programs from scratch.
Use whatever plotting software that is convenient for you.
Please submit your agent (one-step Sarsa), environment (windy-gridworld with king’s moves), and
experiment program and any additional scripts and graphing code. You will submit at least two
plots. The first plot shows the performance of your Sarsa agent with eight actions. This will be a a
learning curve like figure 6.4 in the book: Episodes vs time steps. The second plot will be the
performance of your Sarsa agent with nine actions; again a learning curve like Figure 6.4
