# -*- coding: utf-8 -*-
# @Author: Charlie Gallentine
# @Date:   2020-09-02 16:39:13
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2020-09-03 16:16:31

import numpy as np 
import operator 

class Env:

	# Builds 'graph' shaped like grid
	def __init__(self,rows,cols):
		self.states = {}

		self.rows = rows
		self.cols = cols
		
		self.terminal = np.random.randint(rows * cols)

		for i in range(rows*cols):

			self.states[i] = {}

			self.states[i]['right'] = i + 1
			self.states[i]['up'] = i - cols
			self.states[i]['left'] = i - 1
			self.states[i]['down'] = i + cols

			if i % cols == cols - 1:
				self.states[i]['right'] = i

			if i < cols:
				self.states[i]['up'] = i
	
			if i % cols == 0:
				self.states[i]['left'] = i

			if i >= (rows - 1) * cols:
				self.states[i]['down'] = i


	def step(self,state,action):
		new_state,reward,terminated = -1,-1.0,False

		new_state = self.states[state][action]

		if new_state == self.terminal:
			reward = 1000.0
			terminated = True

		return new_state,reward,terminated


class Agent:

	def __init__(self):
		self.state = None
		self.actions = ['right','up','left','down']

		# {Key: State, Value: Action}
		self.policy = {}

		# {
		# 	Key: state, 
		# 	Value: {
		# 		Key: Action, 
		# 		Value: [times visited in episode, reward for the episode]
		# 	}
		# }
		# self.episodic_policy[state][action][0] = times visited in episode
		# self.episodic_policy[state][action][1] = total reward for episode
		self.episodic_policy = {}

		self.episode_states_visited = []

		self.all_rewards = {}
		self.average_rewards = {}


def learn_policy(agent,env,discount=0.8):
	num_iterations = 10
	num_episodes = 100


	for i in range(num_iterations):

		for j in range(num_episodes):

			agent.episodic_policy = {}

			# Random initial state/action
			agent.state = np.random.randint(len(env.states))
			action = agent.actions[np.random.randint(4)]

			# Take step in environment using random state/action
			new_state,reward,terminated = env.step(agent.state,action)

			# Add episode to path taken
			agent.episodic_policy[agent.state] = {} 
			agent.episodic_policy[agent.state][action] = [0,reward]

			# Move to next state
			agent.state = new_state

			while (agent.state not in agent.episodic_policy) and not terminated:

				agent.episodic_policy[agent.state] = {}

				# If present in policy, follow policy, else random
				if agent.state not in agent.policy:
					action = agent.actions[np.random.randint(len(agent.actions))]
				else:
					action = agent.policy[agent.state]

				# Take step according to policy, determine reward and if in terminal state
				new_state,reward,terminated = env.step(agent.state,action)

				# Adjust discounted rewards on each square visited
				for s in agent.episodic_policy:
					for a in agent.episodic_policy[s]:
						power = agent.episodic_policy[s][a][0] + 1

						agent.episodic_policy[s][a][0] = power
						agent.episodic_policy[s][a][1] += reward * np.power(discount,power)


				# Add reward into state list
				if agent.state not in agent.episodic_policy:
					agent.episodic_policy[agent.state] = {}

				if action not in agent.episodic_policy[agent.state]:
					agent.episodic_policy[agent.state][action] = [0,reward]
							
				# 'Move' to new state
				agent.state = new_state

			# Append reward to list of results for given state/action
			for s in agent.episodic_policy:
				for a in agent.episodic_policy[s]:
					if s not in agent.all_rewards:
						agent.all_rewards[s] = {}
					if a not in agent.all_rewards[s]:
						agent.all_rewards[s][a] = []
					
					agent.all_rewards[s][a].append(agent.episodic_policy[s][a][1])


			# Calculate average reward for state/action pair
			for s in agent.all_rewards:
				for a in agent.all_rewards[s]:
					if s not in agent.average_rewards:
						agent.average_rewards[s] = {}

					agent.average_rewards[s][a] = np.mean( np.array(agent.all_rewards[s][a]) )

			# Get maximum average rewarded value for each state
			for s in agent.average_rewards:
				agent.policy[s] = max(agent.average_rewards[s].items(), key=operator.itemgetter(1))[0]


rows = 6
cols = 6

env = Env(rows,cols)
agent = Agent()

print("Terminal: %d" % (env.terminal))
learn_policy(agent,env)

dirs = np.array([agent.policy[i].rjust(5) for i in range(rows*cols)])
states = np.array([str(i).rjust(5) for i in range(rows*cols)])
print(np.reshape(states,(rows,cols)))
print(np.reshape(dirs,(rows,cols)))










