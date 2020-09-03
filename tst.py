# -*- coding: utf-8 -*-
# @Author: Charlie Gallentine
# @Date:   2020-08-12 11:18:16
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2020-09-02 17:56:59

import numpy as np 

class GridworldEnv:

	"""
	Initialization -
		Return: init, creates object

		Param1: integer, number of gridworld rows
		Param2: integer, number of gridworld columns
	"""
	def __init__(self,rows,cols,wall_percent=0.20):
		self.rows = rows
		self.cols = cols

		# Array containing id of each state
		self.grid = np.array([i for i in range(rows * cols)])

		# Reward in state is given upon transition to that state
		self.rewards = np.full(rows*cols, -1.0, dtype=float)
		
		mask = np.random.choice(a=[False, True], size=rows*cols, p=[wall_percent, 1.0-wall_percent])

		self.rewards = np.where(mask, self.rewards, -1000.0)

		self.terminal = np.random.randint(0,rows*cols)
		self.rewards[self.terminal] = 1000.0


	def step(self,state,action):

		next_state = -1
		reward = 0.0
		terminated = False

		if (((state % self.cols) == (self.cols - 1)) and action == 0) or \
			((state < self.cols) and action == 1) or \
			(((state % self.cols) == 0) and action == 2) or \
			((state >= (self.rows - 1) * self.cols) and action == 3):
			
			# print("Right on right")
			next_state = state
			reward = -1.0
		else:
			if action == 0: # right
				next_state = state + 1
			elif action == 1: # up 
				next_state = state - self.cols
			elif action == 2: # left
				next_state = state - 1		
			elif action == 3: # down
				next_state = state + self.cols

			reward += self.rewards[next_state]

		if next_state == self.terminal:
			terminated = True

		return next_state,reward,terminated

	def print_grid(self):
		# char_grid = np.full((self.rows * self.cols), ' ')
		padding = 5

		state_str = [str(i).rjust(padding) for i in range(self.rows * self.cols)]

		walls = np.where(np.absolute(self.rewards) >= 1000.0)

		char_grid = np.array(state_str)
		char_grid[walls] = '|'.rjust(padding)
		char_grid[self.terminal] = 'T'.rjust(padding)

		return char_grid
		# print(np.reshape(env.grid,(env.rows,env.cols)))

class Agent:


	"""
	Initialization -
		Return: init, creates agent

		Param1: 

	"""
	def __init__(self,rows,cols,grid):
		self.rows = rows
		self.cols = cols 

		self.row = None
		self.col = None

		self.state = None

		self.policy_pi = None

		# 0 : right, 1 : up, 2 : left, 3 : down
		self.actions = [i for i in range(4)]

	def set_policy(self):
		self.policy_pi = np.random.randint(0,4,(self.rows*self.cols),dtype=int)
		
	def set_state(self,state):

		self.state = state

		self.row = state // self.rows
		self.col = state % self.cols

	def set_starting_state(self):
		self.state = np.random.randint(0,self.rows) * self.cols + np.random.randint(0,self.cols)

		return self.state

	def get_starting_action(self):
		return np.random.randint(0,4)

	def get_action_from_policy(self):
		return self.policy_pi[self.state]

	def print_policy(self):
		right = np.where(self.policy_pi == 0)
		up = np.where(self.policy_pi == 1)
		left = np.where(self.policy_pi == 2)
		down = np.where(self.policy_pi == 3)

		arrow_policy = np.full((self.rows * self.cols),'     ')

		arrow_policy[right] = '>'.rjust(5)
		arrow_policy[up] = '^'.rjust(5)
		arrow_policy[left] = '<'.rjust(5)
		arrow_policy[down] = 'v'.rjust(5)

		return arrow_policy


def follow_policy(agent,env,discount=0.8):

	acts = {0:'right',1:'up',2:'left',3:'down'}

	steps = 0
	states_visited = []
	state_action = {}
	terminated = False
	new_state,reward = None,0.0

	total_state_rewards = {}
	avg_state_rewards = {}

	for j in range(100):

		# Each episode:
		# 	Put agent in random state and perform random action
		# 	Follow policy until repeated state or terminal state
		# 		For each state visited, add reward * discount ^ times looped through
		for i in range(200): # Number of episodes

			state_action = {}
			states_visited = []

			# Random starting state/action pair
			current_state = agent.set_starting_state()
			action = agent.get_starting_action()

			# Append starting state to list
			states_visited.append((current_state,action))

			# Take initial step, random state/action
			new_state,reward,terminated = env.step(current_state,action)

			# Add reward into state
			state_action[(current_state,action)] = [0,reward]

			# 'Move' to next state
			current_state = new_state
			agent.set_state(current_state)

			# Until repeat cell and while not in terminal state
			while ((current_state,action) not in states_visited) and not terminated:
				# Mark as visited
				states_visited.append((current_state,action))

				# Find action from policy
				action = agent.get_action_from_policy()

				# Take step according to policy, determine reward and if in terminal state
				new_state,reward,terminated = env.step(current_state,action)

				for s_a in state_action:
					power = state_action[s_a][0] + 1

					state_action[s_a][1] += reward * np.power(discount,power)
					state_action[s_a][0] += 1

				# Add reward into state list
				if (current_state,action) not in state_action:
					state_action[(current_state,action)] = [0,reward]
					
				# 'Move' to new state
				agent.set_state(new_state)
				current_state = new_state

			for s_a in state_action:

				if s_a not in total_state_rewards:
					total_state_rewards[s_a] = []

				total_state_rewards[s_a].append(state_action[s_a][1])


		for s_a,lst in total_state_rewards.items():
			avg_state_rewards[s_a] = np.mean(np.array(lst))
			# print(s_a,total_state_rewards[s_a])

		# Update Policy
		for state in env.grid:
			state_vals = []
			for i in range(len(acts)):
				try:
					state_vals.append(avg_state_rewards[(state,i)])
				except:
					state_vals.append(-100.0)

			# print(state,state_vals,np.amax(np.argmax(np.array(state_vals))))

			agent.policy_pi[state] = np.amax(np.argmax(np.array(state_vals)))



env = GridworldEnv(6,6, wall_percent=0.2)
agent = Agent(env.rows,env.cols,env.grid)

char_grid = env.print_grid()

print("Terminal: ",env.terminal)
print(np.reshape(char_grid,(env.rows,env.cols)))


agent.set_policy()
follow_policy(agent,env)
print()
char_policy = agent.print_policy()

char_policy[np.where(char_grid == '|'.rjust(5))] = '|'.rjust(5)

print(np.reshape(char_policy,(agent.rows,agent.cols)))















