#Description of the problem and plotting:

You will plot the Monte Carlo agent’s estimate of v*(s) during training. To do this: plot on the y-axis V(s)

for each state on the x-axis. The Monte Carlo Exploring Starts algorithm iteratively updates an action-
value function Q(s,a). You will compute V(s) using the following relation: V(s) = maxa Q(s, a) for all s ∈

S. You will plot V(s) after 100 episodes, 1000 episodes, and 8000 episodes. The plotted V should be
produced by averaging over 10 independent runs. That is, your plot should contain 3 lines: V after 100
episodes averaged over 10 runs, V after 1000 episodes averaged over 10 runs, and V after 8000 episodes
averaged over 10 runs. The experiment program you have been given does the averaging for you. You just
have to make sure agent_message in mc_agent.py does the correct thing. You may experiment with more
than 8000 episodes if you like. Our implementation takes 3 to 5 minutes to run 8000 episodes and 10
runs. Please perform at least 10 runs, but feel free to test more runs (typically leading to more clear
results). Number of runs and averaging is done in the experiment program which is provided for you.
