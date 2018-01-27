# Description of the code and agent:

You will implement the Sarsa Control algorithm, with replacing eligibility traces and
tile coding function approximation. This algorithm is described on page 250 of book609.pdf in the course
folder.

The mountain car problem is described in Example 10.1 page 198 of book366.pdf. The three actions
(decelerate, coast, and accelerate) are represented by the integers 0, 1, and 2. The states are represented by
a numpy array of doubles corresponding to the position and velocity of the car. Please look over
mountaincar.py to better understand the task.

Your agent program should use the following parameter settings:
memorySize = 4096 (for the tile coder)
num tilings = 8
shape/size of tilings = 8x8
a = 0.1/(numTilings)
? = 0.9
e = 0.0
initial weights = random numbers between 0 and -0.001
? = 1