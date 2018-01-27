# Description of the code and agents:

- Tabular feature encoding: 

Agent one will implement Semi-gradient TD(0) (described on page 166) with a
tabular or “one-hot” state encoding. A tabular encoding of the current state uses a unique feature vector
for each state, where the number of components in the feature vector is equal the number of states. For
example imagine an MDP with 4 states. The feature vectors corresponding to each state would be:
[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1] for states 1,2,3,4 respectively. Test your semi-gradient TD(0) agent
with alpha=0.5, tabular features on your 1000 state random walk environment.

- Tile coding features: 

Agent two will implement Semi-gradient TD(0) with tile coding. You will use the
supplied tiling coding software (tiles3.py). In this case we treat the state number as a continuous number
and tile code it producing a list of active features equal to the number of tilings. You will use number of
tilings = 50, and tile width equal to 0.2. For TD(0) use alpha = 0.01/50.
Tile coder documentation can be found here: http://www.incompleteideas.net/sutton/tiles/tiles3.html

- State aggregation: 

Agent three will implement Semi-gradient TD(0) with state aggregation. You can
implement state aggregation, as described in Example 9.1. “For the state aggregation, the 1000 states
were partitioned into 10 groups of 100 states each (i.e., states 1–100 were one group, states 101–200 were
another, and so on).” Or you can use the tile coder with num tilings equal to one and a tile width of 0.1 to
achieve state aggregation. For semi-gradient TD(0) use alpha = 0.1.
