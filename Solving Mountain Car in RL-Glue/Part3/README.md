# Description of the plotting:

In this part you will produce a 3D plot of the cost-to-go function learned by your

agent after 1000 episodes like the bottom right subplot in Figure 10.1. Make a 3D plot of minus the state-
values used by your agent after 1 run of 1000 episodes. That is plot:



as a function of the state, over the range of allowed positions and velocities as given in the book. That is
for 50 equally spaced allowed values of the position, for 50 equally spaced allowed values of the velocity
query the tile coder for the corresponding state-action features, then use those features to compute the
negative of the best action-value according to your agent’s learned action-value function after 1000
episodes.
