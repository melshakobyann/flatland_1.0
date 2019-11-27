import PPO
import numpy as np
from numpy import inf
import time
import matplotlib.pyplot as plt
import math
from torch import device, cuda, from_numpy, clamp, exp, min, squeeze, mean, optim, tensor, save
from flatland_model import Memory
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator # Round 1
from flatland.envs.schedule_generators import sparse_schedule_generator # Round 2
from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool
import warnings
warnings.filterwarnings('ignore')


continue_training=False

##################################
# Hyper parameters
state_space=231
num_layers=6
action_space=5
hidden_size=256
agent_num=1
num_epochs=5
gamma=0.99
learning_rate=0.01
clip_epsilon=0.2
td_lambda=15
c1=0.3
c2=0.002
update_timestep=3560
update_episode=30
random_seed=2
episode_num=400
scheduled_learning=True
env_update_time=30
agent_num_update_time=2
env_update_decay=1
hardness_lvl=1
##################################

##################################
# Env parameters
env_width=10
env_height=10
rail_generator=complex_rail_generator(
								nr_start_goal=20,
								nr_extra=1,
								min_dist=9,
								max_dist=99999,
								seed=0)
schedule_generator=complex_schedule_generator()
env = RailEnv(
	    width=env_width,
	    height=env_height,
	    rail_generator=rail_generator,
	    schedule_generator=schedule_generator,
	    number_of_agents=agent_num)
env_renderer = RenderTool(env)
render = False
render_sleep_time = 0.0
stuck_break_pont = 20
max_timesteps_in_episode = update_timestep/1.2
##################################


memory = Memory()
ppo = PPO.FlatlandPPO(state_space=state_space,
				  action_space=action_space,
				  hidden_size=hidden_size,
				  agent_num=agent_num, 
				  num_epochs=num_epochs,
				  num_layers=num_layers, 
				  gamma=gamma, 
				  learning_rate=learning_rate, 
				  clip_epsilon=clip_epsilon, 
				  td_lambda=td_lambda, 
				  c1=c1, 
				  c2=c2)

if continue_training:
	ppo.model.load_state_dict(load('/home/vache/ML_projects/rl/flatland/saved_model/PPO_flatland.pth'))


def env_gradual_update(input_env, decay, agent=False, hardness_lvl=1):

	agent_num = input_env.number_of_agents
	env_width = input_env.width + 4
	env_height = input_env.height + 4

	if agent:
		agent_num += 1
		env_width = math.floor(env_width/decay)
		env_height = math.floor(env_height/decay)

	if hardness_lvl == 1:

		rail_generator=complex_rail_generator(
										nr_start_goal=20,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator=complex_schedule_generator()
	else:

		rail_generator=sparse_rail_generator(
										nr_start_goal=9,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator= sparse_schedule_generator()

	global env, env_renderer, render


	if render:
		env_renderer.close_window()

	env = RailEnv(
	    width=env_width,
	    height=env_height,
	    rail_generator=rail_generator,
	    schedule_generator=schedule_generator,
	    number_of_agents=agent_num)

	env_renderer = RenderTool(env)


def env_random_update(input_env, decay, agent=False, hardness_lvl=1):

	agent_num = np.random.randint(1, 5)
	env_width = (agent_num + 2) * 5
	env_height = (agent_num + 2) * 5


	if hardness_lvl == 1:

		rail_generator=complex_rail_generator(
										nr_start_goal=20,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator=complex_schedule_generator()
	else:

		rail_generator=sparse_rail_generator(
										nr_start_goal=9,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator= sparse_schedule_generator()

	global env, env_renderer, render


	if render:
		env_renderer.close_window()

	env = RailEnv(
	    width=env_width,
	    height=env_height,
	    rail_generator=rail_generator,
	    schedule_generator=schedule_generator,
	    number_of_agents=agent_num)

	env_renderer = RenderTool(env)


def main():

	timesteps = 0
	episode_rewards = []

	for episode in range(episode_num):
		all_done = False
		episode_timesteps = 0
		states_dict = env.reset()
		print(states_dict)
		states = list(states_dict.values())
		states = ppo.obs_into_right_shape(states)
		states_list = list(np.zeros(stuck_break_pont))
		# last states to test later that if the are the same, it means that the enviromnetn is stuck
		# so we need to reset
		print('Episode:', episode)
		reward_sum = 0

		if render:
			env_renderer.reset()

		while not all_done:
			timesteps += 1
			episode_timesteps += 1
			actions, values = ppo.step(obs=states, memory=memory, model=ppo.model_old, agent_num=env.number_of_agents)

			states, rewards, done, _ = env.step(actions)
			
			states_list.append(states)
			states_list.pop(0)

			states = ppo.obs_into_right_shape(list(states.values()))

			all_done = done['__all__']

			# check if the environment is stuck, reset the game and give a big negative reward
			if all(x == states_list[0] for x in states_list) or episode_timesteps >= max_timesteps_in_episode:
				all_done = True
				rewards = {}
				for i in range(agent_num):
					rewards[i] = -99999


			ppo.add_rewards_to_list(rewards)
			ppo.add_values_to_list(values.detach().numpy())

			if render:
				env_renderer.render_env(show=True, frames=False, show_observations=False)
				time.sleep(render_sleep_time)

			reward_sum += sum(list(rewards.values()))

			if update_timestep - timesteps == 0:

				print('---------TRAINING---------')
				ppo.train(memory, agent_num=env.number_of_agents)
				memory.clear_memory()
				timesteps = 0

			

		if scheduled_learning and episode % (env_update_time*agent_num_update_time) == 0 and episode != 0:
			env_gradual_update(env, decay=env_update_decay, agent=True, hardness_lvl=hardness_lvl)
			print('---------TRAINING---------')
			ppo.train(memory, agent_num=env.number_of_agents)
			memory.clear_memory()
			timesteps = 0
			if render:
				env_renderer.reset()
			
			

		elif scheduled_learning and episode % env_update_time == 0 and episode != 0:
			env_gradual_update(env, decay=env_update_decay, hardness_lvl=hardness_lvl)
			print('---------TRAINING---------')
			ppo.train(memory, agent_num=env.number_of_agents)
			memory.clear_memory()
			timesteps = 0
			if render:
				env_renderer.reset()
			

		episode_rewards.append(reward_sum)

		save(ppo.model.state_dict(), '/home/vache/ML_projects/rl/flatland/saved_model/PPO_flatland.pth')
		print('Reward sum:', reward_sum)


	plt.plot(episode_rewards)
	plt.ylabel('reward')
	plt.show()





if __name__ == '__main__':
    main()