# Ghosts_In_The_Maze
An agent navigates through a maze with moving ghosts and walls to reach a goal.

## Environment Setup
The environment consists of a maze of the dimension 51x51. Each cell in the maze is filled with either 1s or 0s. The 1s denote a cell through which the agent could move and the 0s denote cells which are blocked due to walls.
The 1s are populated with a probability of 0.72 and the 0s are populated with a probability of 0.28. Once the maze is populated, a couple of conditions are checked before using it as part of the simulation for the agents.
Condition 1 –
The source and destinations are unblocked cells.
Condition 2 –
There is a viable path between the source and the destination.

## Agent 
The agent is going to start in the upper left corner, and attempt to navigate to the lower right corner. The agent canmove in the cardinal directions (up/down/left/right), but only between unblocked squares, and cannot move outsidethe 51x51 grid. At any time, the agent can ‘see’ the entirety of the maze, and use this information to plan a path.

## Ghosts
Unfortunately for the agent, the maze is full of ghosts. Each ghost starts at a random location in the maze that is reachable from the upper left corner (so that no ghost gets walled off). 

If the agent enters a cell with a ghost (or a ghost enters the agent’s cell), the agent dies. This is to be avoided. Each time the agent moves, the ghosts will also move. This means that whatever plan the agent initially generated to traverse the maze may at any point become blocked or invalid. This may mean the agent needs to re-plan its path through the maze to try to avoid the ghosts and may have to repeatedly re-plan as the ghosts move.

The ghosts move according to simple rules: at every timestep, a ghost picks one of its neighbors (up/down/left/right); if the picked neighbor is unblocked, the ghost moves to that cell; if the picked neighbor is blocked, the ghost either stays in place with probability 0.5, or moves into the blocked cell with probability 0.5. (These rules apply even if the ghost is currently within a blocked cell.) Every time the agent moves, the ghosts move according to the above rule. If the agent touches a ghost, the agent dies.

## Different strategy implementations

## Agent 1
Agent 1 plans the shortest path through the maze and executes it, ignoring the ghosts. This agent is incredibly efficient - it only has to plan a path once - but it makes no adjustments or updates due to a changing environment.

## Agent 2
Agent 2 re-plans. At every timestep, Agent 2 recalculates a new path to the goal based on the current information and executes the next step in this new path. Agent 2 is constantly updating, and readjusting based on new information about the ghosts. 
Note, however, Agent 2 makes no projections about the future. If all paths to the goal are currently blocked, Agent 2 attempts to move away from the nearest visible ghost (not occupying a blocked cell).

## Agent 3
Agent 3 forecasts. At every timestep, Agent 3 considers each possible move it might take (including staying in place), and ‘simulates’ the future based on the rules of Agent 2 past that point. For each possible move, this future is simulated some number of times, and then Agent 3 chooses among the moves with greatest success rates in these simulations. Agent 3 can be thought of as Agent 2, plus the ability to imagine the future.

## Agent 4
Agent 4 is an agent that executes a free strategy agent that maximizes performance.









