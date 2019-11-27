from torch.distributions import Categorical
from torch import device, cuda, from_numpy, clamp, exp, min, squeeze, mean, optim, tensor, FloatTensor, stack
from flatland_model import Decoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')


device = device("cuda:0" if cuda.is_available() else "cpu")


class FlatlandPPO():

	def __init__(self, state_space, action_space, hidden_size, agent_num, num_epochs, num_layers, gamma=0.9, learning_rate=0.001, clip_epsilon=0.2, td_lambda=10, c1=0.5, c2=0.01, betas=(0.9, 0.999)):

		self.model = Decoder(state_space, hidden_size, action_space, num_layers)
		self.model_old = Decoder(state_space, hidden_size, action_space, num_layers)
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.clip_epsilon = clip_epsilon
		self.td_lambda  = td_lambda
		self.num_epochs = num_epochs
		self.c1 = c1
		self.c2 = c2
		self.betas = betas
		self.agent_num = agent_num
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=self.betas)

		self.init_reward_dict_of_list(agent_num)
		self.init_value_dict_of_list(agent_num)

		self.step_counter = 0
		self.prev_value = 0
		self.curr_value = 0
		self.prev_policy = 0
		self.curr_policy = 0



	def init_reward_dict_of_list(self, agent_num):

		self.reward_dict_of_list = {}
		for agent in range(agent_num):
			self.reward_dict_of_list[agent] = np.zeros(self.td_lambda)


	def init_value_dict_of_list(self, agent_num):

		self.value_dict_of_list = {}
		for agent in range(agent_num):
			self.value_dict_of_list[agent] = np.zeros(self.td_lambda)



	def add_rewards_to_list(self, rewards):
		"""
		Keeps the last [td_lambda] amount of rewards for every agent.
		Update after update
		"""

		for agent_index in list(rewards.keys()):
			self.reward_dict_of_list[agent_index] = np.delete(self.reward_dict_of_list[agent_index], 0)
			self.reward_dict_of_list[agent_index] = np.append(self.reward_dict_of_list[agent_index], rewards[agent_index])


	def add_values_to_list(self, values):
		"""
		Keeps the last [td_lambda] amount of values for every agent.
		Update after update
		"""
		for agent_index in range(len(values)):
			self.value_dict_of_list[agent_index] = np.delete(self.value_dict_of_list[agent_index], 0)
			self.value_dict_of_list[agent_index] = np.append(self.value_dict_of_list[agent_index], values[agent_index])


	def calculate_advantage(self):
		"""
		Calcualtes the advatage using value_dict_of_list and reward_dict_of_list.
		I wanted to use TD lambda for advantage calculation rather than Monte Carlo or TD(0)

		advantage = V[0] - (R[1] + gamma*R[2] + gamma**2*R[3] + ... + gamma**(len(list) - 1)*R[-1] + gamma**(len(list) - 1)*V[-1])

		Remember to .detach() value before appending to the list
		"""

		advantage = []
		gamma_power = [i for i in range(len(self.reward_dict_of_list[0])-1)]
		powerful_gamma = np.power(self.gamma, gamma_power)

		for agent in range(len(self.reward_dict_of_list)):
			agent_advantage = self.value_dict_of_list[agent][0] - np.sum([np.sum(np.multiply(powerful_gamma, self.reward_dict_of_list[agent][1:])), self.value_dict_of_list[agent][-1]])
			advantage.append(agent_advantage)


		return advantage


	def obs_into_right_shape(self, obs):
		"""
		Gets as input, (array or list) the observation in the shape of (n, m) and returns in the shape of (n, 1, m)
		Also changes all -inf values to -10

		call this function before predict
		"""
		tmp_obs = np.array(obs)
		tmp_obs[tmp_obs==-np.inf] = -10
		tmp_obs[tmp_obs==np.inf] = -10
		states = tensor(tmp_obs)
		state = tensor(states)
		state = state.view(state.size()[0], 1, state[0].size()[0])

		return state.detach()


	def step(self, obs, model, agent_num, memory=None, greedy=False):
		"""
		Does a step and also appends needed data to memory
		obs is a numpy array
		"""
		agent_actions = {}
		np_obs = np.array(obs)
		torch_obs = tensor(obs).float().to(device)
		value, action_probs = model.forward(torch_obs, model.init_hidden(1))
		dist = Categorical(action_probs)

		if greedy:
			actions = action_probs.max(-1)[1]

		else:
			actions = dist.sample()

		actions_for_dict = squeeze(actions)
		if agent_num > 1:
			for agent in range(agent_num):
				agent_actions[agent] = int(actions_for_dict[agent])

		else:
			agent_actions[0] = int(actions_for_dict.item())



		if memory:

			np_actions = np.array(actions)
			np_logprobs = dist.log_prob(actions).detach().numpy()
			np_advantages = np.array(self.calculate_advantage())

			np_mem_states = np.array(memory.states)
			np_mem_actions = np.array(memory.actions)
			np_mem_logprobs = np.array(memory.logprobs)
			np_mem_advantages = np.array(memory.advantages)

			if not memory.states:
				new_mem_states = np_obs
				new_mem_actions = np_actions
				new_mem_logprobs = np_logprobs
				new_mem_advantages = np_advantages


			else:
				new_mem_states = np.column_stack((np_mem_states, np_obs))
				new_mem_actions = np.column_stack((np_mem_actions, np_actions))
				new_mem_logprobs = np.column_stack((np_mem_logprobs, np_logprobs))
				new_mem_advantages = np.column_stack((np_mem_advantages, np_advantages))

			memory.states = list(new_mem_states)
			memory.actions = list(new_mem_actions)
			memory.logprobs = list(new_mem_logprobs)
			memory.advantages = list(new_mem_advantages)

		return agent_actions, value


	def evaluate(self, state, action, model, batch_size):

		"""
		Based on the policy (model) returns that model's policy information,
		log probs for actions, state values and policy action distribution entropy
		"""

		value, action_probs = model.forward(state, model.init_hidden(batch_size))
		dist = Categorical(action_probs)

		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()


		return action_logprobs, squeeze(value), dist_entropy


	def train(self, memory, agent_num):

		old_states = tensor(memory.states).float().to(device).detach()
		old_actions = tensor(memory.actions).float().to(device).detach()
		old_logprobs = tensor(memory.logprobs).float().to(device).detach()
		advantages = tensor(memory.advantages).float().to(device).detach()

		np_states = np.array(old_states)
		np_actions = np.array(old_actions)
		np_logprobs = np.array(old_logprobs)
		np_advantages = np.array(advantages)


		np_states_pop = np_states[:, :-self.td_lambda]
		np_actions_pop = np_actions[:, :-self.td_lambda]
		np_logprobs_pop = np_logprobs[:, :-self.td_lambda]
		np_advantages_pop = np_advantages[:, self.td_lambda:]


		old_states = tensor(np_states_pop).float().to(device).detach()
		old_actions = tensor(np_actions_pop).float().to(device).detach()
		old_logprobs = tensor(np_logprobs_pop).float().to(device).detach()
		advantages = tensor(np_advantages_pop).float().to(device).detach()


		batch_size = old_states.size()[1]



		for _ in range(self.num_epochs):

			logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions, self.model, batch_size)

			# Calculate ratios
			ratios = exp(logprobs - old_logprobs.detach())

			# The Loss
			part1 = ratios * advantages
			part2 = clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages

			stacked_part1_part2 = stack([part1, part2])

			loss = -min(stacked_part1_part2, 0)[0] + self.c1*mean(advantages**2, 1).view(-1, 1) - self.c2*dist_entropy
			

			# take gradient step
			self.optimizer.zero_grad()
			loss_mean = loss.mean()
			loss_mean.backward()

			# for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
			# 	print(p.grad.data.norm(2).item())


			self.optimizer.step()

		# pdate old model's weights with the current one

		self.model_old.load_state_dict(self.model.state_dict())

		# Zero the memory for advantage

		self.init_reward_dict_of_list(agent_num)
		self.init_value_dict_of_list(agent_num)