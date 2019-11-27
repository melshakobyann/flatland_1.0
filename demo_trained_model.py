from main import state_space, action_space, hidden_size, agent_num, num_epochs, num_layers, gamma, learning_rate, clip_epsilon, td_lambda, c1, c2
from PPO import FlatlandPPO
import numpy as np
from numpy import inf
import time
from torch import load
import math
from flatland_model import Memory
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator # Round 1
from flatland.envs.schedule_generators import sparse_schedule_generator # Round 2
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

ppo = FlatlandPPO(state_space=state_space,
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


ppo.model.load_state_dict(load('/home/vache/ML_projects/rl/flatland/saved_model/PPO_flatland.pth'))


episode_num = 10
agent_num=2
env_width=15
env_height=15
rail_generator=complex_rail_generator(
									nr_start_goal=9,
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
render = True
render_sleep_time = 0.03
stuck_break_pont = 20
max_timesteps_in_episode = 1000



for i in range(episode_num):
	all_done = False
	episode_timesteps = 0

	reward_sum = 0
	states_dict = env.reset()
	states = list(states_dict.values())
	states = ppo.obs_into_right_shape(states)
	states_list = list(np.zeros(stuck_break_pont))

	if render:
		env_renderer.reset()

	while not all_done:
		episode_timesteps += 1
		actions, values = ppo.step(obs=states, model=ppo.model, agent_num=agent_num, greedy=True)

		states, rewards, done, _ = env.step(actions)
		
		
		states_list.append(states)
		states_list.pop(0)
		states = ppo.obs_into_right_shape(list(states.values()))

		all_done = done['__all__']

		# check if the environment is stuck, reset the game and give a big negative reward
		if all(x == states_list[0] for x in states_list) or episode_timesteps >= max_timesteps_in_episode:
			all_done = True

		reward_sum += sum(list(rewards.values()))


		if render:
			env_renderer.render_env(show=True, frames=False, show_observations=False)
			time.sleep(render_sleep_time)

	print('Episode:{} reward:{}'.format(i, reward_sum))
