# monte_carlo_gridworld

## Solves a gridworld problem using the Monte-Carlo method

This solution is currently set up to solve a gridworld problem using Monte-Carlo exploration. Solution can be extended to solve arbitrary graph problems given a proper Agent/Environment combination. Grid is implemented as a graph with edges looping to self. 

Actions are be selected to be potential steps from a node in the graph to another node. Rewards/Nodes must be set up by Environment __init__ function.