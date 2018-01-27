# Description of the problem and plotting:

Experiment with one of the key parameters of your Dyna-Q agent. Perform a
systematic parameter sweep of the alpha parameter. You will test 6 different alpha values in:
{0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0},
recording the performance of your agent for each setting of alpha.
Then you will plot the performance of your agent for each value of alpha in the set. Specifically plot
alpha-value on the x-axis, and the average number of steps per episode over the first 50 episodes

(averaged over 10 runs) on the y-axis. That is, each point on the graph reports the performance of Dyna-
Q, for one specific setting of alpha, averaged over 10 independent runs.

In order to get good performance you may have to experiment with the exploration rate parameter
(epsilon). Start with the values used in the book. You are not required for systematically sweep epsilon,
but you are free to do so. Use number of planning steps equal to 5.
This exercise requires implementing one agent program (Dyna-Q), an environment program
(implementing the gridworld), and two experiment programs—the first for generating your version of
Figure 8.3, and the second to run your parameter sweep of alpha.
