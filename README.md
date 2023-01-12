# Ghosts_In_The_Maze
An agent navigates through a maze with moving ghosts and walls to reach a goal.

## Environment Setup
The environment consists of mazes of the dimension 51x51. Each cell in the maze is filled with either 1s or 0s. The 1s denote a cell through which the agent could move and the 0s denote cells which are blocked due to walls.
The 1s are populated with a probability of 0.72 and the 0s are populated with a probability of 0.28. Once the maze is populated, a couple of conditions are checked before using it as part of the simulation for the agents.
Condition 1 –
The source and destinations are unblocked cells.
Condition 2 –
There is a viable path between the source and the destination.



